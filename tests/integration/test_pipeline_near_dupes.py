"""End-to-end integration test: near-duplicate detection.

Injects 100 near-duplicate pairs (realized shingle Jaccard ~0.85) into a
1,000-document corpus, runs the pipeline at threshold 0.8, and checks F1, output
uniqueness, and sub-quadratic scaling.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pytest

from dedup_pipeline.config import PipelineConfig
from dedup_pipeline.evaluation.metrics import pairs_from_clusters, precision_recall_f1
from dedup_pipeline.evaluation.synthetic_injector import inject_near_duplicates
from dedup_pipeline.pipeline.pipeline import DedupPipeline

# Word-set target whose realized word-3-gram shingle Jaccard lands near 0.85.
_NEAR_DUP_WORD_TARGET = 0.96


def _unique_records(n: int, seed: int, words: int = 80, vocab: int = 4000) -> list[dict[str, str]]:
    """Build n mutually-distinct documents from a large random vocabulary."""
    rng = random.Random(seed)
    return [
        {"id": f"u{i}", "text": " ".join(f"w{rng.randint(0, vocab)}" for _ in range(words))}
        for i in range(n)
    ]


@pytest.fixture
def near_dup_config() -> PipelineConfig:
    """A config tuned for near-duplicate detection at threshold 0.8."""
    return PipelineConfig(
        num_hash_functions=128,
        lsh_bands=16,
        lsh_rows=8,
        shingle_mode="word",
        shingle_size=3,
        jaccard_threshold=0.8,
        random_seed=42,
    )


@pytest.mark.integration
def test_near_dup_f1(near_dup_config: PipelineConfig) -> None:
    """F1 >= 0.85 for near-duplicate pairs at Jaccard ~0.85.

    Matters because real corpora are dominated by *near*-duplicates (boilerplate,
    minor edits); detecting them is the system's main value.
    """
    base = _unique_records(1000, seed=10)
    records, ground_truth = inject_near_duplicates(
        base, num_pairs=100, target_jaccard=_NEAR_DUP_WORD_TARGET, seed=10
    )
    clusters, _ = DedupPipeline(near_dup_config).detect_clusters(records)
    _, _, f1 = precision_recall_f1(pairs_from_clusters(clusters), ground_truth)
    assert f1 >= 0.85


@pytest.mark.integration
def test_no_document_written_twice(near_dup_config: PipelineConfig, tmp_path: Path) -> None:
    """Every surviving document appears exactly once in the output.

    Matters because a representative-selection bug could keep two members of a
    cluster, re-introducing duplicates into the cleaned corpus.
    """
    base = _unique_records(1000, seed=11)
    records, _ = inject_near_duplicates(
        base, num_pairs=100, target_jaccard=_NEAR_DUP_WORD_TARGET, seed=11
    )
    dest = tmp_path / "out.jsonl"
    DedupPipeline(near_dup_config).run(records, dest)
    ids = [json.loads(line)["id"] for line in dest.read_text().strip().split("\n")]
    assert len(ids) == len(set(ids))  # no id repeated


@pytest.mark.integration
def test_subquadratic_scaling(near_dup_config: PipelineConfig) -> None:
    """Doubling the corpus does not quadruple the runtime.

    Matters because LSH exists precisely to avoid the O(n^2) all-pairs blow-up; a
    quadratic regression would make large corpora intractable.
    """
    small = _unique_records(1000, seed=12)
    large = _unique_records(2000, seed=13)

    pipe = DedupPipeline(near_dup_config)
    # Warm up to absorb one-time costs (imports, allocator) so neither timing is
    # biased by first-call overhead — important on noisy shared CI runners.
    pipe.detect_clusters(_unique_records(200, seed=99))

    start = time.perf_counter()
    pipe.detect_clusters(small)
    t_small = time.perf_counter() - start

    start = time.perf_counter()
    pipe.detect_clusters(large)
    t_large = time.perf_counter() - start

    # Quadratic would be ~4x for a 2x corpus; allow a generous 3.5x for
    # linear-ish behaviour plus CI noise (still strictly below the 4x quadratic
    # signature). t_small is floored to avoid divide-by-zero on very fast hosts.
    assert t_large < 3.5 * max(t_small, 1e-3)
