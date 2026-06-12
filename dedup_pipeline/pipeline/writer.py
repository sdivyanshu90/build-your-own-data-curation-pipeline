"""Deduplicated output writer and statistics export (Stage 10).

Streams the original source a second time and writes only the documents whose
index is in the keep set, in JSONL or Parquet. Surviving documents are written
**verbatim** (original text, not the normalized form used internally), so
deduplication only ever *removes* documents, never alters them. A statistics
JSON file is written alongside the output.

Responsibility:
    * Write the surviving corpus and the run statistics.

Inputs:
    * The keep-set of document indices, the original source, the destination,
      and precomputed run statistics.

Outputs:
    * A JSONL/Parquet corpus file and a ``*_stats.json`` sidecar; returns the
      statistics dict.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dedup_pipeline.config import PipelineConfig
from dedup_pipeline.exceptions import IOError  # noqa: A004 - domain-specific name
from dedup_pipeline.pipeline.reader import DocumentReader

logger = logging.getLogger(__name__)


class DeduplicatedWriter:
    """Write the surviving corpus and a statistics sidecar.

    Thread-safety:
        Stateless aside from the immutable config/reader; create one per run.

    Args:
        config: The pipeline configuration (output format, field names).
        reader: The :class:`DocumentReader` used to re-stream the source.

    Example:
        >>> from dedup_pipeline.config import PipelineConfig
        >>> from dedup_pipeline.pipeline.reader import DocumentReader
        >>> cfg = PipelineConfig(output_format="jsonl")
        >>> w = DeduplicatedWriter(cfg, DocumentReader(cfg))
        >>> isinstance(w, DeduplicatedWriter)
        True
    """

    def __init__(self, config: PipelineConfig, reader: DocumentReader) -> None:
        self._config = config
        self._reader = reader

    def stats_path_for(self, dest: Path) -> Path:
        """Return the statistics sidecar path for an output destination.

        Args:
            dest: The corpus output path.

        Returns:
            A path like ``output_stats.json`` next to ``dest``.
        """
        return dest.with_name(f"{dest.stem}_stats.json")

    def write(
        self,
        doc_indices_to_keep: set[int],
        source: str | Path | list[dict[str, Any]],
        dest: Path,
        input_count: int,
        cluster_size_histogram: dict[int, int],
        runtime_per_stage: dict[str, float],
    ) -> dict[str, Any]:
        """Write surviving documents and the statistics JSON.

        Args:
            doc_indices_to_keep: Indices (into the stage-1 stream order) to keep.
            source: The original source, re-streamed in the same order.
            dest: Output corpus path.
            input_count: Total documents read in stage 1.
            cluster_size_histogram: Map of duplicate-cluster size -> count.
            runtime_per_stage: Map of stage name -> seconds elapsed.

        Returns:
            The statistics dict that was also written to disk.

        Raises:
            IOError: If the output or stats file cannot be written.

        Example:
            >>> import tempfile, pathlib
            >>> from dedup_pipeline.config import PipelineConfig
            >>> from dedup_pipeline.pipeline.reader import DocumentReader
            >>> cfg = PipelineConfig()
            >>> w = DeduplicatedWriter(cfg, DocumentReader(cfg))
            >>> dest = pathlib.Path(tempfile.mkdtemp()) / "out.jsonl"
            >>> src = [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]
            >>> stats = w.write({0}, src, dest, 2, {2: 1}, {"total": 0.0})
            >>> stats["output_count"]
            1
        """
        try:
            ensure_parent = dest.parent
            ensure_parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IOError(f"could not create output directory for {dest}: {exc}") from exc

        if self._config.output_format == "jsonl":
            output_count = self._write_jsonl(doc_indices_to_keep, source, dest)
        else:
            output_count = self._write_parquet(doc_indices_to_keep, source, dest)

        dedup_rate = (
            (input_count - output_count) / input_count if input_count > 0 else 0.0
        )
        stats: dict[str, Any] = {
            "input_count": input_count,
            "output_count": output_count,
            "dedup_rate": dedup_rate,
            # JSON keys must be strings: serialise the histogram's int sizes.
            "cluster_size_histogram": {
                str(size): count for size, count in sorted(cluster_size_histogram.items())
            },
            "runtime_per_stage": runtime_per_stage,
            "config": self._config.to_serializable_dict(),
        }
        self._write_stats(self.stats_path_for(dest), stats)
        logger.info(
            "Wrote %d/%d documents to %s (dedup_rate=%.4f)",
            output_count,
            input_count,
            dest,
            dedup_rate,
        )
        return stats

    def _iter_kept_records(
        self,
        doc_indices_to_keep: set[int],
        source: str | Path | list[dict[str, Any]],
    ) -> Any:
        """Yield reconstructed records for kept documents, in source order.

        Args:
            doc_indices_to_keep: Indices to keep.
            source: The original source.

        Yields:
            ``dict`` records faithful to the original input (original text plus
            preserved metadata).
        """
        for index, doc in enumerate(self._reader.stream(source)):
            if index in doc_indices_to_keep:
                # Reconstruct the original record: metadata + the text field.
                record = dict(doc.metadata)
                record[self._config.text_field] = doc.text
                yield record

    def _write_jsonl(
        self,
        doc_indices_to_keep: set[int],
        source: str | Path | list[dict[str, Any]],
        dest: Path,
    ) -> int:
        """Stream kept documents to a JSONL file; return the count written."""
        count = 0
        try:
            with dest.open("w", encoding="utf-8") as handle:
                for record in self._iter_kept_records(doc_indices_to_keep, source):
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
        except OSError as exc:
            raise IOError(f"failed writing JSONL output {dest}: {exc}") from exc
        return count

    def _write_parquet(
        self,
        doc_indices_to_keep: set[int],
        source: str | Path | list[dict[str, Any]],
        dest: Path,
    ) -> int:
        """Stream kept documents to Parquet in batches; return the count."""
        count = 0
        batch: list[dict[str, Any]] = []
        writer: pq.ParquetWriter | None = None
        try:
            for record in self._iter_kept_records(doc_indices_to_keep, source):
                batch.append(record)
                count += 1
                if len(batch) >= self._config.batch_size:
                    writer = self._flush_parquet_batch(batch, writer, dest)
                    batch = []
            if batch:
                writer = self._flush_parquet_batch(batch, writer, dest)
            if writer is None:
                # No surviving records: write an empty file so dest always exists.
                empty = pa.table({self._config.text_field: pa.array([], pa.string())})
                pq.write_table(empty, dest)
            else:
                writer.close()
        except (OSError, pa.ArrowInvalid) as exc:
            if writer is not None:
                writer.close()
            raise IOError(f"failed writing Parquet output {dest}: {exc}") from exc
        return count

    @staticmethod
    def _flush_parquet_batch(
        batch: list[dict[str, Any]],
        writer: pq.ParquetWriter | None,
        dest: Path,
    ) -> pq.ParquetWriter:
        """Write one batch to Parquet, creating the writer on first call.

        Args:
            batch: Records to write.
            writer: The existing writer or ``None``.
            dest: Output path.

        Returns:
            The (possibly newly created) writer.
        """
        table = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(dest, table.schema)
        writer.write_table(table)
        return writer

    @staticmethod
    def _write_stats(stats_path: Path, stats: dict[str, Any]) -> None:
        """Write the statistics dict as pretty JSON.

        Args:
            stats_path: Destination for the statistics file.
            stats: The statistics mapping.

        Raises:
            IOError: If the file cannot be written.
        """
        try:
            with stats_path.open("w", encoding="utf-8") as handle:
                json.dump(stats, handle, indent=2, sort_keys=False)
        except OSError as exc:
            raise IOError(f"failed writing stats JSON {stats_path}: {exc}") from exc
