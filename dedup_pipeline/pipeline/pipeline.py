"""The 10-stage deduplication orchestrator.

:class:`DedupPipeline` wires the package's components into the canonical
MinHash + LSH near-duplicate removal flow [Broder 1997; Indyk & Motwani 1998;
Lee et al. 2022]. Each of the ten stages is an independently callable method that
logs its start, completion, key counts, and elapsed time; :meth:`run` chains them
and supports checkpoint-based resume.

Stages:
    1. :meth:`stream_documents` -> 2. :meth:`normalize_batch` ->
    3. :meth:`shingle_batch` -> 4. :meth:`compute_signatures` ->
    5. :meth:`build_bucket_index` -> 6. :meth:`enumerate_candidate_pairs` ->
    7. :meth:`verify_pair` -> 8. :meth:`cluster_duplicates` ->
    9. :meth:`select_representatives` -> 10. :meth:`write_deduplicated`.

Responsibility:
    * Own the stage components and orchestrate a resumable end-to-end run.

Inputs:
    * A source (path/glob/list/dataset id) and a destination path.

Outputs:
    * A deduplicated corpus file plus a statistics dict/JSON.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from dedup_pipeline.clustering.union_find import UnionFind
from dedup_pipeline.config import PipelineConfig
from dedup_pipeline.lsh.banding import BandingScheme
from dedup_pipeline.lsh.bucket_index import BucketIndex, build_bucket_index
from dedup_pipeline.lsh.candidate_pairs import enumerate_candidate_pairs
from dedup_pipeline.minhash.minhash import MinHasher
from dedup_pipeline.pipeline.checkpointer import StageCheckpointer
from dedup_pipeline.pipeline.reader import Document, DocumentReader
from dedup_pipeline.pipeline.writer import DeduplicatedWriter
from dedup_pipeline.text_processing.normalizer import TextNormalizer
from dedup_pipeline.text_processing.shingler import Shingler
from dedup_pipeline.text_processing.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

# Checkpoint artifact names (also used to log which stages a resume skips).
_CK_DOCUMENTS = "documents"  # covers stages 1-2
_CK_SIGNATURES = "signatures"  # covers stages 3-4
_CK_BUCKET_INDEX = "bucket_index"  # covers stage 5
_CK_CANDIDATE_PAIRS = "candidate_pairs"  # covers stages 6-7
_CK_CLUSTERS = "clusters"  # covers stage 8
_CK_KEEP = "keep_indices"  # covers stage 9


class DedupPipeline:
    """End-to-end MinHash/LSH deduplication pipeline.

    Thread-safety:
        A pipeline instance holds per-run mutable state (timings, counts) and is
        intended to be used by a single run at a time. Its component objects are
        individually thread-safe where documented, but do not share one
        :class:`DedupPipeline` across concurrent runs.

    Args:
        config: The validated :class:`PipelineConfig`.

    Example:
        >>> import tempfile, pathlib
        >>> cfg = PipelineConfig(num_hash_functions=32, lsh_bands=8, lsh_rows=4,
        ...                      shingle_mode="word", shingle_size=2,
        ...                      jaccard_threshold=0.5)
        >>> pipe = DedupPipeline(cfg)
        >>> src = [{"id": str(i), "text": "the quick brown fox"} for i in range(3)]
        >>> dest = pathlib.Path(tempfile.mkdtemp()) / "out.jsonl"
        >>> stats = pipe.run(src, dest)
        >>> stats["output_count"]  # 3 identical docs -> 1 kept
        1
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._normalizer = TextNormalizer(config.cjk_ratio_threshold)
        self._tokenizer = Tokenizer()
        self._shingler = Shingler(
            config.shingle_size,
            config.shingle_mode,
            config.random_seed,
            self._tokenizer,
            self._normalizer,
        )
        self._minhasher = MinHasher(
            config.num_hash_functions, config.random_seed, config.use_numba
        ).fit()
        self._banding = BandingScheme.from_config(config)
        self._reader = DocumentReader(config)
        self._writer = DeduplicatedWriter(config, self._reader)
        self._checkpointer = StageCheckpointer(config.checkpoint_dir)
        # Per-run state populated during run() / individual stage calls.
        self._runtime: dict[str, float] = {}
        self._input_count: int = 0
        self._cluster_size_histogram: dict[int, int] = {}

    @property
    def config(self) -> PipelineConfig:
        """The pipeline configuration."""
        return self._config

    @property
    def runtime_per_stage(self) -> dict[str, float]:
        """Elapsed seconds recorded per stage during the last run."""
        return self._runtime

    @contextmanager
    def _record_time(self, stage: str) -> Iterator[None]:
        """Time an eager stage body, logging completion and storing elapsed.

        Args:
            stage: The stage name (used as the runtime key and in logs).

        Yields:
            Control to the timed body.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._runtime[stage] = elapsed
            logger.info("Stage %s: completed in %.3fs", stage, elapsed)

    # ------------------------------------------------------------------ #
    # Stage 1
    # ------------------------------------------------------------------ #
    def stream_documents(
        self, source: str | Path | list[dict[str, Any]]
    ) -> Iterator[Document]:
        """Stage 1: stream documents from any supported source.

        Args:
            source: A JSONL/Parquet path, a glob string, a HuggingFace dataset
                id, or an in-memory ``list[dict]``.

        Yields:
            :class:`Document` objects. Timing and the final count are logged when
            the stream is exhausted.

        Example:
            >>> pipe = DedupPipeline(PipelineConfig())
            >>> len(list(pipe.stream_documents([{"text": "a"}, {"text": "b"}])))
            2
        """
        logger.info("Stage stream_documents: start")
        start = time.perf_counter()
        count = 0
        for doc in self._reader.stream(source):
            count += 1
            yield doc
        elapsed = time.perf_counter() - start
        self._runtime["stream_documents"] = elapsed
        self._input_count = count
        logger.info(
            "Stage stream_documents: streamed %d documents in %.3fs", count, elapsed
        )

    # ------------------------------------------------------------------ #
    # Stage 2
    # ------------------------------------------------------------------ #
    def normalize_batch(self, docs: list[Document]) -> list[Document]:
        """Stage 2: normalize a batch of documents in place.

        Applies NFKC -> lowercase -> HTML strip -> whitespace collapse -> strip,
        and logs how many documents are detected as CJK (which downstream
        shingling will handle with character n-grams).

        Args:
            docs: Documents to normalize (their ``text`` is replaced).

        Returns:
            The same list, with normalized text.

        Example:
            >>> pipe = DedupPipeline(PipelineConfig())
            >>> d = Document("1", "  <b>Hi</b>  THERE ", {})
            >>> pipe.normalize_batch([d])[0].text
            'hi there'
        """
        logger.info("Stage normalize_batch: start (%d docs)", len(docs))
        cjk_count = 0
        with self._record_time("normalize_batch"):
            for doc in docs:
                doc.text = self._normalizer.normalize(doc.text)
                if self._config.shingle_mode == "word" and self._normalizer.is_cjk(
                    doc.text
                ):
                    cjk_count += 1
        if cjk_count:
            logger.info(
                "Stage normalize_batch: %d/%d CJK docs -> char shingling",
                cjk_count,
                len(docs),
            )
        return docs

    # ------------------------------------------------------------------ #
    # Stage 3
    # ------------------------------------------------------------------ #
    def shingle_batch(self, docs: list[Document]) -> list[set[int]]:
        """Stage 3: convert documents to integer shingle sets.

        Args:
            docs: Normalized documents.

        Returns:
            One ``set[int]`` of 64-bit shingle ids per document.

        Example:
            >>> cfg = PipelineConfig(shingle_mode="word", shingle_size=2)
            >>> pipe = DedupPipeline(cfg)
            >>> out = pipe.shingle_batch([Document("1", "a b c", {})])
            >>> len(out[0])
            2
        """
        logger.info("Stage shingle_batch: start (%d docs)", len(docs))
        with self._record_time("shingle_batch"):
            result = self._shingler.shingle_batch([doc.text for doc in docs])
        total = sum(len(s) for s in result)
        logger.info("Stage shingle_batch: produced %d total shingles", total)
        return result

    # ------------------------------------------------------------------ #
    # Stage 4
    # ------------------------------------------------------------------ #
    def compute_signatures(self, shingle_sets: list[set[int]]) -> npt.NDArray[Any]:
        """Stage 4: compute the MinHash signature matrix.

        Complexity:
            ``O(num_hash_functions * T)`` where ``T`` is the total number of
            shingles across the batch (vectorized over hash functions; no Python
            loop over documents).

        Args:
            shingle_sets: One shingle set per document.

        Returns:
            A ``(n_docs, num_hash_functions)`` ``uint32`` signature matrix.

        Example:
            >>> pipe = DedupPipeline(PipelineConfig(num_hash_functions=16,
            ...                      lsh_bands=4, lsh_rows=4))
            >>> sig = pipe.compute_signatures([{1, 2, 3}, {1, 2, 3}])
            >>> sig.shape
            (2, 16)
        """
        logger.info("Stage compute_signatures: start (%d docs)", len(shingle_sets))
        with self._record_time("compute_signatures"):
            signatures = self._minhasher.batch_transform(shingle_sets)
        logger.info(
            "Stage compute_signatures: matrix %s dtype=%s",
            signatures.shape,
            signatures.dtype,
        )
        return signatures

    # ------------------------------------------------------------------ #
    # Stage 5
    # ------------------------------------------------------------------ #
    def build_bucket_index(self, signatures: npt.NDArray[Any]) -> BucketIndex:
        """Stage 5: build the inverted LSH bucket index.

        Args:
            signatures: The ``(n_docs, num_hash_functions)`` signature matrix.

        Returns:
            A :class:`BucketIndex` holding only multi-document buckets.

        Example:
            >>> import numpy as np
            >>> pipe = DedupPipeline(PipelineConfig(num_hash_functions=4,
            ...                      lsh_bands=2, lsh_rows=2))
            >>> sig = np.array([[1, 1, 2, 2], [1, 1, 9, 9]], dtype=np.uint32)
            >>> idx = pipe.build_bucket_index(sig)
            >>> len(idx) >= 1
            True
        """
        logger.info(
            "Stage build_bucket_index: start (%d docs, %d bands)",
            signatures.shape[0],
            self._banding.num_bands,
        )
        with self._record_time("build_bucket_index"):
            index = build_bucket_index(signatures, self._banding)
        logger.info("Stage build_bucket_index: %d non-empty buckets", len(index))
        return index

    # ------------------------------------------------------------------ #
    # Stage 6
    # ------------------------------------------------------------------ #
    def enumerate_candidate_pairs(
        self, index: BucketIndex
    ) -> Iterator[tuple[int, int]]:
        """Stage 6: yield unique candidate pairs ``(i, j)`` with ``i < j``.

        Args:
            index: The inverted bucket index.

        Yields:
            Deduplicated candidate pairs. Timing and the final count are logged
            when the generator is exhausted.

        Example:
            >>> pipe = DedupPipeline(PipelineConfig())
            >>> idx = BucketIndex()
            >>> idx.add_bucket(1, [0, 2])
            >>> list(pipe.enumerate_candidate_pairs(idx))
            [(0, 2)]
        """
        logger.info("Stage enumerate_candidate_pairs: start")
        start = time.perf_counter()
        count = 0
        for pair in enumerate_candidate_pairs(
            index,
            self._config.use_bloom_filter,
            self._config.bloom_expected_pairs,
            self._config.bloom_false_positive_rate,
        ):
            count += 1
            yield pair
        elapsed = time.perf_counter() - start
        self._runtime["enumerate_candidate_pairs"] = elapsed
        logger.info(
            "Stage enumerate_candidate_pairs: %d unique pairs in %.3fs",
            count,
            elapsed,
        )

    # ------------------------------------------------------------------ #
    # Stage 7
    # ------------------------------------------------------------------ #
    def verify_pair(self, sig_a: npt.NDArray[Any], sig_b: npt.NDArray[Any]) -> bool:
        """Stage 7: re-verify a candidate pair against the Jaccard threshold.

        Only used when ``config.high_precision_mode`` is ``True``. Logs at DEBUG
        because it is called once per candidate pair.

        Args:
            sig_a: Signature of the first document.
            sig_b: Signature of the second document.

        Returns:
            ``True`` if the signature-estimated Jaccard meets the threshold.

        Example:
            >>> import numpy as np
            >>> pipe = DedupPipeline(PipelineConfig(jaccard_threshold=0.5))
            >>> a = np.array([1, 2, 3, 4], dtype=np.uint32)
            >>> pipe.verify_pair(a, a)
            True
        """
        estimate = MinHasher.estimate_jaccard(sig_a, sig_b)
        passed = estimate >= self._config.jaccard_threshold
        logger.debug("verify_pair: J_est=%.3f -> %s", estimate, passed)
        return passed

    # ------------------------------------------------------------------ #
    # Stage 8
    # ------------------------------------------------------------------ #
    def cluster_duplicates(
        self, pairs: Iterable[tuple[int, int]]
    ) -> list[list[int]]:
        """Stage 8: cluster candidate pairs into connected components.

        Args:
            pairs: Candidate (verified) pairs.

        Returns:
            Connected components of size >= ``config.min_cluster_size`` (each a
            list of document indices).

        Example:
            >>> pipe = DedupPipeline(PipelineConfig())
            >>> sorted(sorted(c) for c in pipe.cluster_duplicates([(0, 1), (1, 2)]))
            [[0, 1, 2]]
        """
        logger.info("Stage cluster_duplicates: start")
        with self._record_time("cluster_duplicates"):
            union_find = UnionFind.from_pairs(pairs)
            clusters = [
                sorted(component)
                for component in union_find.clusters(self._config.min_cluster_size)
            ]
        logger.info("Stage cluster_duplicates: %d duplicate clusters", len(clusters))
        return clusters

    # ------------------------------------------------------------------ #
    # Stage 9
    # ------------------------------------------------------------------ #
    def select_representatives(
        self, clusters: list[list[int]], docs: list[Document]
    ) -> set[int]:
        """Stage 9: choose which document index to keep from each cluster.

        Args:
            clusters: Duplicate clusters (lists of doc indices).
            docs: All documents (used for the ``longest`` strategy and to know
                the total count).

        Returns:
            The set of indices to **keep**: one representative per cluster plus
            every non-duplicate document.

        Example:
            >>> pipe = DedupPipeline(PipelineConfig(representative_strategy="first"))
            >>> docs = [Document(str(i), "x" * (i + 1), {}) for i in range(4)]
            >>> sorted(pipe.select_representatives([[1, 3]], docs))
            [0, 1, 2]
        """
        logger.info("Stage select_representatives: start (%d clusters)", len(clusters))
        total = len(docs)
        rng = random.Random(self._config.random_seed)
        with self._record_time("select_representatives"):
            duplicate_indices: set[int] = set()
            representatives: set[int] = set()
            for cluster in clusters:
                duplicate_indices.update(cluster)
                representatives.add(self._pick_representative(cluster, docs, rng))
            keep = (set(range(total)) - duplicate_indices) | representatives
        logger.info(
            "Stage select_representatives: keeping %d/%d documents", len(keep), total
        )
        return keep

    def _pick_representative(
        self, cluster: list[int], docs: list[Document], rng: random.Random
    ) -> int:
        """Pick one index from a cluster per the configured strategy.

        Args:
            cluster: The cluster's document indices.
            docs: All documents.
            rng: A seeded RNG (used only by the ``random`` strategy).

        Returns:
            The index to keep.

        Raises:
            ValueError: If the strategy is unrecognised (guarded by config).
        """
        strategy = self._config.representative_strategy
        if strategy == "first":
            return min(cluster)
        if strategy == "random":
            return rng.choice(sorted(cluster))
        if strategy == "longest":
            # Longest text wins; ties broken by smallest index for determinism.
            return max(cluster, key=lambda i: (len(docs[i].text), -i))
        raise ValueError(f"unknown representative_strategy: {strategy!r}")

    # ------------------------------------------------------------------ #
    # Stage 10
    # ------------------------------------------------------------------ #
    def write_deduplicated(
        self,
        doc_ids_to_keep: set[int],
        source: str | Path | list[dict[str, Any]],
        dest: Path,
    ) -> dict[str, Any]:
        """Stage 10: write surviving documents and the statistics JSON.

        Args:
            doc_ids_to_keep: Indices (stage-1 order) to keep.
            source: The original source, re-streamed to fetch verbatim text.
            dest: Output corpus path.

        Returns:
            The statistics dict (also written to a ``*_stats.json`` sidecar).

        Example:
            >>> import tempfile, pathlib
            >>> pipe = DedupPipeline(PipelineConfig())
            >>> dest = pathlib.Path(tempfile.mkdtemp()) / "out.jsonl"
            >>> src = [{"text": "a"}, {"text": "b"}]
            >>> pipe._input_count = 2
            >>> stats = pipe.write_deduplicated({0}, src, dest)
            >>> stats["output_count"]
            1
        """
        logger.info("Stage write_deduplicated: start (keep %d)", len(doc_ids_to_keep))
        input_count = self._input_count
        if input_count == 0:
            # Independent call without a prior run: count by a cheap stream pass.
            input_count = sum(1 for _ in self._reader.stream(source))
        with self._record_time("write_deduplicated"):
            stats = self._writer.write(
                doc_ids_to_keep,
                source,
                dest,
                input_count,
                self._cluster_size_histogram,
                self._runtime,
            )
        return stats

    def detect_clusters(
        self, source: str | Path | list[dict[str, Any]]
    ) -> tuple[list[list[int]], int]:
        """Run stages 1-8 and return duplicate clusters without writing output.

        Useful for evaluation and tuning, where the predicted clusters (not a
        written corpus) are compared against ground truth.

        Args:
            source: The input corpus source.

        Returns:
            A tuple ``(clusters, n_docs)`` where ``clusters`` are duplicate
            components (lists of stage-1 indices) and ``n_docs`` is the corpus
            size.

        Example:
            >>> cfg = PipelineConfig(num_hash_functions=16, lsh_bands=4,
            ...                      lsh_rows=4, shingle_mode="word",
            ...                      shingle_size=2, jaccard_threshold=0.5)
            >>> pipe = DedupPipeline(cfg)
            >>> clusters, n = pipe.detect_clusters([{"text": "a b c"}] * 3)
            >>> n, len(clusters)
            (3, 1)
        """
        documents = self.normalize_batch(list(self.stream_documents(source)))
        n_docs = len(documents)
        signatures = self.compute_signatures(self.shingle_batch(documents))
        index = self.build_bucket_index(signatures)
        pairs_iter = self.enumerate_candidate_pairs(index)
        if self._config.high_precision_mode:
            pairs: list[tuple[int, int]] = [
                (i, j)
                for (i, j) in pairs_iter
                if self.verify_pair(signatures[i], signatures[j])
            ]
        else:
            pairs = list(pairs_iter)
        return self.cluster_duplicates(pairs), n_docs

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run(
        self,
        source: str | Path | list[dict[str, Any]],
        dest: str | Path,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Run the full pipeline end to end, optionally resuming from checkpoints.

        Args:
            source: The input corpus source.
            dest: The output corpus path.
            resume: If ``True`` and a ``checkpoint_dir`` is configured, completed
                stages are loaded from disk and skipped.

        Returns:
            The statistics dict from stage 10.

        Raises:
            CheckpointError: If a checkpoint is corrupt during resume.
            IOError: On source/destination I/O errors.

        Example:
            >>> import tempfile, pathlib
            >>> cfg = PipelineConfig(num_hash_functions=16, lsh_bands=4, lsh_rows=4,
            ...                      shingle_mode="word", shingle_size=2,
            ...                      jaccard_threshold=0.5)
            >>> pipe = DedupPipeline(cfg)
            >>> dest = pathlib.Path(tempfile.mkdtemp()) / "out.jsonl"
            >>> src = [{"text": "alpha beta gamma"}] * 4
            >>> pipe.run(src, dest)["output_count"]
            1
        """
        dest_path = Path(dest)
        self._runtime = {}
        run_start = time.perf_counter()
        use_ck = resume and self._checkpointer.enabled
        logger.info(
            "Pipeline run start (resume=%s, checkpoints=%s)",
            resume,
            self._checkpointer.enabled,
        )

        documents = self._load_or_compute_documents(source, use_ck)
        self._input_count = len(documents)
        signatures = self._load_or_compute_signatures(documents, use_ck)
        index = self._load_or_compute_bucket_index(signatures, use_ck)
        pairs = self._load_or_compute_pairs(index, signatures, use_ck)
        clusters = self._load_or_compute_clusters(pairs, use_ck)

        self._cluster_size_histogram = self._histogram(clusters)

        keep = self._load_or_compute_keep(clusters, documents, use_ck)
        stats = self.write_deduplicated(keep, source, dest_path)

        total_elapsed = time.perf_counter() - run_start
        logger.info(
            "Pipeline run complete in %.3fs: %d -> %d docs (dedup_rate=%.4f)",
            total_elapsed,
            stats["input_count"],
            stats["output_count"],
            stats["dedup_rate"],
        )
        return stats

    # ------------------------------------------------------------------ #
    # Resume helpers (load-or-compute per artifact)
    # ------------------------------------------------------------------ #
    def _load_or_compute_documents(
        self, source: str | Path | list[dict[str, Any]], use_ck: bool
    ) -> list[Document]:
        """Load normalized documents from checkpoint or run stages 1-2."""
        if use_ck and self._checkpointer.has_stage(_CK_DOCUMENTS):
            logger.info(
                "Resume: loaded %r checkpoint; skipping stages stream_documents, "
                "normalize_batch",
                _CK_DOCUMENTS,
            )
            return list(self._checkpointer.load(_CK_DOCUMENTS))
        documents = self.normalize_batch(list(self.stream_documents(source)))
        self._checkpointer.save(_CK_DOCUMENTS, documents)
        return documents

    def _load_or_compute_signatures(
        self, documents: list[Document], use_ck: bool
    ) -> npt.NDArray[Any]:
        """Load signatures from checkpoint or run stages 3-4."""
        if use_ck and self._checkpointer.has_stage(_CK_SIGNATURES):
            logger.info(
                "Resume: loaded %r checkpoint; skipping stages shingle_batch, "
                "compute_signatures",
                _CK_SIGNATURES,
            )
            return np.asarray(self._checkpointer.load(_CK_SIGNATURES))
        shingle_sets = self.shingle_batch(documents)
        signatures = self.compute_signatures(shingle_sets)
        self._checkpointer.save(_CK_SIGNATURES, signatures)
        return signatures

    def _load_or_compute_bucket_index(
        self, signatures: npt.NDArray[Any], use_ck: bool
    ) -> BucketIndex:
        """Load the bucket index from checkpoint or run stage 5."""
        if use_ck and self._checkpointer.has_stage(_CK_BUCKET_INDEX):
            logger.info(
                "Resume: loaded %r checkpoint; skipping stage build_bucket_index",
                _CK_BUCKET_INDEX,
            )
            loaded: BucketIndex = self._checkpointer.load(_CK_BUCKET_INDEX)
            return loaded
        index = self.build_bucket_index(signatures)
        self._checkpointer.save(_CK_BUCKET_INDEX, index)
        return index

    def _load_or_compute_pairs(
        self, index: BucketIndex, signatures: npt.NDArray[Any], use_ck: bool
    ) -> list[tuple[int, int]]:
        """Load candidate pairs from checkpoint or run stages 6-7."""
        if use_ck and self._checkpointer.has_stage(_CK_CANDIDATE_PAIRS):
            logger.info(
                "Resume: loaded %r checkpoint; skipping stages "
                "enumerate_candidate_pairs, verify_pair",
                _CK_CANDIDATE_PAIRS,
            )
            return [tuple(p) for p in self._checkpointer.load(_CK_CANDIDATE_PAIRS)]
        pairs_iter = self.enumerate_candidate_pairs(index)
        if self._config.high_precision_mode:
            logger.info("high_precision_mode: verifying candidate pairs")
            pairs = [
                (i, j)
                for (i, j) in pairs_iter
                if self.verify_pair(signatures[i], signatures[j])
            ]
        else:
            pairs = list(pairs_iter)
        self._checkpointer.save(_CK_CANDIDATE_PAIRS, pairs)
        return pairs

    def _load_or_compute_clusters(
        self, pairs: list[tuple[int, int]], use_ck: bool
    ) -> list[list[int]]:
        """Load clusters from checkpoint or run stage 8."""
        if use_ck and self._checkpointer.has_stage(_CK_CLUSTERS):
            logger.info(
                "Resume: loaded %r checkpoint; skipping stage cluster_duplicates",
                _CK_CLUSTERS,
            )
            return [list(c) for c in self._checkpointer.load(_CK_CLUSTERS)]
        clusters = self.cluster_duplicates(pairs)
        self._checkpointer.save(_CK_CLUSTERS, clusters)
        return clusters

    def _load_or_compute_keep(
        self, clusters: list[list[int]], documents: list[Document], use_ck: bool
    ) -> set[int]:
        """Load the keep-set from checkpoint or run stage 9."""
        if use_ck and self._checkpointer.has_stage(_CK_KEEP):
            logger.info(
                "Resume: loaded %r checkpoint; skipping stage select_representatives",
                _CK_KEEP,
            )
            return set(self._checkpointer.load(_CK_KEEP))
        keep = self.select_representatives(clusters, documents)
        self._checkpointer.save(_CK_KEEP, sorted(keep))
        return keep

    @staticmethod
    def _histogram(clusters: list[list[int]]) -> dict[int, int]:
        """Compute a cluster-size -> count histogram.

        Args:
            clusters: Duplicate clusters.

        Returns:
            A mapping from cluster size to the number of clusters of that size.
        """
        histogram: dict[int, int] = {}
        for cluster in clusters:
            size = len(cluster)
            histogram[size] = histogram.get(size, 0) + 1
        return histogram
