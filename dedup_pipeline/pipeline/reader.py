"""Streaming document reader (Stage 1 of the pipeline).

Reads documents from heterogeneous sources **without loading the whole corpus
into memory**, yielding :class:`Document` objects one at a time. Supported
sources:

    * an in-memory ``list[dict]``;
    * a JSONL file (optionally gzip-compressed) or a glob of them;
    * a Parquet file (streamed by row-group batches);
    * a HuggingFace Hub dataset id (streamed).

Malformed records are logged at ``WARNING`` and skipped rather than aborting the
run, so one bad line never poisons a multi-million-line shard.

Responsibility:
    * Normalise diverse inputs into a uniform :class:`Document` stream.

Inputs:
    * A source descriptor (list, path, glob, or dataset id) plus the config.

Outputs:
    * An iterator of :class:`Document`.
"""

from __future__ import annotations

import glob
import gzip
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import pyarrow.parquet as pq

from dedup_pipeline.config import PipelineConfig
from dedup_pipeline.exceptions import IOError  # noqa: A004 - domain-specific name

logger = logging.getLogger(__name__)

# File suffixes recognised as JSONL (line-delimited JSON), with optional gzip.
_JSONL_SUFFIXES: frozenset[str] = frozenset({".jsonl", ".json", ".ndjson"})
_PARQUET_SUFFIXES: frozenset[str] = frozenset({".parquet", ".pq"})
# Characters that mark a string source as a filesystem glob pattern.
_GLOB_CHARS: tuple[str, ...] = ("*", "?", "[")


@dataclass(slots=True)
class Document:
    """A single corpus document flowing through the pipeline.

    Attributes:
        id: A stable identifier (from the source or synthesised by index).
        text: The raw document text.
        metadata: Arbitrary per-document metadata carried alongside the text.
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentReader:
    """Stream :class:`Document` objects from a source.

    Thread-safety:
        Stateless aside from the immutable config; a single reader may be used
        from multiple threads, but each :meth:`stream` call returns an
        independent iterator.

    Args:
        config: The pipeline configuration (supplies ``text_field``,
            ``id_field``, and ``batch_size``).

    Example:
        >>> from dedup_pipeline.config import PipelineConfig
        >>> reader = DocumentReader(PipelineConfig())
        >>> docs = list(reader.stream([{"id": "a", "text": "hello"}]))
        >>> docs[0].text
        'hello'
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def stream(self, source: str | Path | list[dict[str, Any]]) -> Iterator[Document]:
        """Yield documents from any supported source.

        Args:
            source: An in-memory ``list[dict]``, a JSONL/Parquet path, a glob
                pattern string, or a HuggingFace dataset id.

        Yields:
            :class:`Document` objects in source order.

        Raises:
            IOError: If a path source matches no files or has an unsupported
                format.

        Example:
            >>> reader = DocumentReader(PipelineConfig())
            >>> [d.id for d in reader.stream([{"text": "x"}, {"text": "y"}])]
            ['0', '1']
        """
        if isinstance(source, list):
            yield from self._stream_records(source)
            return

        source_str = str(source)
        if any(ch in source_str for ch in _GLOB_CHARS):
            yield from self._stream_glob(source_str)
            return

        path = Path(source)
        if path.exists():
            yield from self._stream_path(path)
            return

        # Not a list, glob, or existing path -> treat as a HuggingFace dataset id.
        yield from self._stream_huggingface(source_str)

    def _make_document(self, record: dict[str, Any], index: int) -> Document | None:
        """Convert a raw record dict into a :class:`Document`.

        Args:
            record: The raw mapping read from the source.
            index: The running record index (used to synthesise a missing id).

        Returns:
            A :class:`Document`, or ``None`` if the record lacks usable text (in
            which case a WARNING is logged and the caller skips it).
        """
        text = record.get(self._config.text_field)
        if not isinstance(text, str):
            logger.warning(
                "Skipping record %d: missing/non-string field %r",
                index,
                self._config.text_field,
            )
            return None
        raw_id = record.get(self._config.id_field)
        doc_id = str(raw_id) if raw_id is not None else str(index)
        # Everything except the text field is preserved as metadata.
        metadata = {
            k: v for k, v in record.items() if k != self._config.text_field
        }
        return Document(id=doc_id, text=text, metadata=metadata)

    def _stream_records(self, records: list[dict[str, Any]]) -> Iterator[Document]:
        """Stream an in-memory list of record dicts."""
        for index, record in enumerate(records):
            doc = self._make_document(record, index)
            if doc is not None:
                yield doc

    def _stream_glob(self, pattern: str) -> Iterator[Document]:
        """Stream every file matching a glob, in sorted (deterministic) order."""
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            raise IOError(f"glob pattern matched no files: {pattern!r}")
        logger.info("Glob %r matched %d file(s)", pattern, len(matches))
        index = 0
        for file_path in matches:
            for doc in self._stream_path(Path(file_path), start_index=index):
                index += 1
                yield doc

    def _stream_path(self, path: Path, start_index: int = 0) -> Iterator[Document]:
        """Dispatch a single file to the JSONL or Parquet reader by suffix."""
        suffixes = path.suffixes
        # Handle '.jsonl.gz' by inspecting the suffix beneath '.gz'.
        is_gzip = path.suffix == ".gz"
        effective_suffix = (
            suffixes[-2] if is_gzip and len(suffixes) >= 2 else path.suffix
        )
        if effective_suffix in _JSONL_SUFFIXES:
            yield from self._stream_jsonl(path, is_gzip, start_index)
        elif effective_suffix in _PARQUET_SUFFIXES:
            yield from self._stream_parquet(path, start_index)
        else:
            raise IOError(
                f"unsupported file format {effective_suffix!r} for {path}; "
                "expected JSONL or Parquet"
            )

    def _stream_jsonl(
        self, path: Path, is_gzip: bool, start_index: int
    ) -> Iterator[Document]:
        """Stream a JSONL file line by line (never loads the whole file)."""
        index = start_index
        try:
            # Conditional gzip/plain open requires binding before `with`; the
            # handle is still closed by the `with` block below.
            handle: IO[str] = (
                gzip.open(path, "rt", encoding="utf-8")  # noqa: SIM115
                if is_gzip
                else path.open("r", encoding="utf-8")
            )
            with handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "Skipping malformed JSON at %s:%d: %s", path, index, exc
                        )
                        index += 1
                        continue
                    if not isinstance(record, dict):
                        logger.warning(
                            "Skipping non-object JSON at %s:%d", path, index
                        )
                        index += 1
                        continue
                    doc = self._make_document(record, index)
                    index += 1
                    if doc is not None:
                        yield doc
        except OSError as exc:
            raise IOError(f"failed reading JSONL file {path}: {exc}") from exc

    def _stream_parquet(self, path: Path, start_index: int) -> Iterator[Document]:
        """Stream a Parquet file by row-group batches (bounded memory)."""
        index = start_index
        try:
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=self._config.batch_size):
                for record in batch.to_pylist():
                    doc = self._make_document(record, index)
                    index += 1
                    if doc is not None:
                        yield doc
        except (OSError, ValueError) as exc:
            raise IOError(f"failed reading Parquet file {path}: {exc}") from exc

    def _stream_huggingface(self, dataset_id: str) -> Iterator[Document]:
        """Stream a HuggingFace Hub dataset id (lazy import of ``datasets``)."""
        try:
            from datasets import load_dataset
        except ImportError as exc:  # optional heavy dependency
            raise IOError(
                f"source {dataset_id!r} is not a path; the 'datasets' package is "
                "required to load it as a HuggingFace dataset"
            ) from exc
        logger.info("Streaming HuggingFace dataset %r", dataset_id)
        try:
            dataset = load_dataset(dataset_id, split="train", streaming=True)
        except (ValueError, FileNotFoundError, OSError) as exc:
            raise IOError(
                f"could not load HuggingFace dataset {dataset_id!r}: {exc}"
            ) from exc
        for index, record in enumerate(dataset):
            doc = self._make_document(dict(record), index)
            if doc is not None:
                yield doc
