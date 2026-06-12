"""Unit tests for :class:`DocumentReader` (Stage 1, streaming ingestion).

The reader normalises heterogeneous inputs (in-memory lists, JSONL, gzip JSONL,
Parquet, globs) into a single uniform :class:`Document` stream. Two properties
are load-bearing for the whole pipeline: (1) it must be *resilient* — a single
malformed line or a record missing its text field is skipped with a warning, not
raised, so one bad row never aborts a multi-million-line shard; and (2) it must
be *faithful and deterministic* — every good record yields a Document with the
correct id (explicit or index-synthesised), text, and preserved metadata, in a
stable order. If ingestion silently drops good documents, reorders them, or
mis-assigns ids, every downstream stage (and the final keep-set indexing) is
corrupted. These tests pin resilience, fidelity, and the per-format dispatch.

HuggingFace streaming is intentionally untested here because it requires network
access.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dedup_pipeline.config import PipelineConfig
from dedup_pipeline.exceptions import IOError as DedupIOError
from dedup_pipeline.pipeline.reader import Document, DocumentReader


def _write_jsonl(path: Path, lines: list[str]) -> None:
    """Write raw (already-serialised) JSONL lines to a file via tmp_path."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----- In-memory list of dicts -------------------------------------------


def test_stream_inmemory_list_yields_documents() -> None:
    """An in-memory list yields one Document per record with the right text.

    Matters because the in-memory path is the canonical test/debug entry point
    and the contract every other source must match: text is pulled from the
    configured text_field, and the count equals the number of records.
    """
    reader = DocumentReader(PipelineConfig())
    docs = list(reader.stream([{"text": "hello"}, {"text": "world"}]))
    assert [d.text for d in docs] == ["hello", "world"]
    assert all(isinstance(d, Document) for d in docs)


def test_missing_id_is_synthesised_from_index() -> None:
    """A record without an id field gets str(index) as its id.

    Matters because every Document needs a stable identifier for cluster
    reporting and metadata, and the keep-set is computed over stream order; the
    synthesised id must be the string of the running index so it aligns with that
    order.
    """
    reader = DocumentReader(PipelineConfig())
    docs = list(reader.stream([{"text": "a"}, {"text": "b"}, {"text": "c"}]))
    assert [d.id for d in docs] == ["0", "1", "2"]


def test_explicit_id_is_preserved_and_metadata_excludes_text() -> None:
    """An explicit id is kept verbatim; non-text fields become metadata.

    Matters because user-supplied ids are how callers trace a surviving document
    back to its source row, and metadata must carry every non-text field so the
    writer can reconstruct the original record faithfully. The text field itself
    must NOT leak into metadata or it would be written twice.
    """
    reader = DocumentReader(PipelineConfig())
    [doc] = list(reader.stream([{"id": "doc-7", "text": "body", "lang": "en"}]))
    assert doc.id == "doc-7"
    assert doc.text == "body"
    assert doc.metadata == {"id": "doc-7", "lang": "en"}
    assert "text" not in doc.metadata


def test_record_missing_text_field_is_skipped() -> None:
    """A record lacking the text field is skipped, not raised.

    Matters because real corpora contain rows with no usable text; the pipeline
    must drop them silently (with a warning) rather than aborting, while still
    emitting every well-formed neighbour.
    """
    reader = DocumentReader(PipelineConfig())
    docs = list(reader.stream([{"text": "good"}, {"no_text": "oops"}, {"text": "fine"}]))
    assert [d.text for d in docs] == ["good", "fine"]


def test_custom_text_field_is_honoured() -> None:
    """A configured non-default text_field controls which key is read as text.

    Matters because datasets name their text column differently (e.g. "content");
    the reader must honour config.text_field so the same pipeline ingests any
    schema without remapping the data first.
    """
    reader = DocumentReader(PipelineConfig(text_field="content"))
    [doc] = list(reader.stream([{"content": "hi", "text": "ignored"}]))
    assert doc.text == "hi"
    # The default "text" key is now just metadata, not the document text.
    assert doc.metadata == {"text": "ignored"}


# ----- JSONL file ---------------------------------------------------------


def test_stream_jsonl_file_reads_each_line(tmp_path: Path) -> None:
    """Each non-blank JSONL line becomes one Document, in file order.

    Matters because JSONL is the primary on-disk format; the reader must stream
    line by line (never loading the whole file) and preserve order so stream
    indices stay aligned with the keep-set.
    """
    path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        path,
        [json.dumps({"id": "a", "text": "one"}), json.dumps({"id": "b", "text": "two"})],
    )
    reader = DocumentReader(PipelineConfig())
    docs = list(reader.stream(path))
    assert [(d.id, d.text) for d in docs] == [("a", "one"), ("b", "two")]


def test_malformed_json_line_is_skipped_good_lines_survive(tmp_path: Path) -> None:
    """A malformed JSON line is skipped; surrounding good lines still stream.

    Matters because a single corrupt line in a million-line shard must not abort
    the run. The reader logs a warning and continues, so this is the core
    resilience guarantee that lets ingestion tolerate dirty data.
    """
    path = tmp_path / "dirty.jsonl"
    _write_jsonl(
        path,
        [
            json.dumps({"id": "a", "text": "good1"}),
            "{ this is not valid json",
            json.dumps({"id": "b", "text": "good2"}),
        ],
    )
    reader = DocumentReader(PipelineConfig())
    docs = list(reader.stream(path))
    assert [d.text for d in docs] == ["good1", "good2"]


def test_jsonl_record_missing_text_field_is_skipped(tmp_path: Path) -> None:
    """A JSONL record without the text field is skipped, not raised.

    Matters because the file path must apply the same resilience rule as the
    in-memory path: rows with no usable text are dropped so the pipeline keeps
    flowing on heterogeneous shards.
    """
    path = tmp_path / "mixed.jsonl"
    _write_jsonl(
        path,
        [
            json.dumps({"text": "keep"}),
            json.dumps({"meta": "no text here"}),
            json.dumps({"text": "also keep"}),
        ],
    )
    reader = DocumentReader(PipelineConfig())
    docs = list(reader.stream(path))
    assert [d.text for d in docs] == ["keep", "also keep"]


def test_jsonl_explicit_id_and_metadata_preserved(tmp_path: Path) -> None:
    """A JSONL record keeps its explicit id and preserves non-text fields.

    Matters because the writer reconstructs the original record from metadata;
    losing the id or any sidecar field on the file path would silently strip
    columns from the deduplicated output.
    """
    path = tmp_path / "meta.jsonl"
    _write_jsonl(path, [json.dumps({"id": "x1", "text": "t", "score": 0.5})])
    reader = DocumentReader(PipelineConfig())
    [doc] = list(reader.stream(path))
    assert doc.id == "x1"
    assert doc.metadata == {"id": "x1", "score": 0.5}


# ----- Glob patterns ------------------------------------------------------


def test_glob_combines_multiple_files(tmp_path: Path) -> None:
    """A glob pattern streams every matching file, combining their records.

    Matters because corpora are routinely sharded across many files; the glob
    path must read all shards (in deterministic sorted order) so no shard is
    silently dropped from the corpus.
    """
    _write_jsonl(tmp_path / "shard_a.jsonl", [json.dumps({"text": "a1"}), json.dumps({"text": "a2"})])
    _write_jsonl(tmp_path / "shard_b.jsonl", [json.dumps({"text": "b1"})])
    reader = DocumentReader(PipelineConfig())
    pattern = str(tmp_path / "*.jsonl")
    docs = list(reader.stream(pattern))
    assert len(docs) == 3
    assert {d.text for d in docs} == {"a1", "a2", "b1"}


def test_glob_no_match_raises_io_error(tmp_path: Path) -> None:
    """A glob pattern matching zero files raises the package IOError.

    Matters because a typo'd or empty input glob must fail loudly at the start of
    the run, not silently produce an empty corpus that the user mistakes for a
    successful (but total) deduplication.
    """
    reader = DocumentReader(PipelineConfig())
    pattern = str(tmp_path / "no_such_*.jsonl")
    with pytest.raises(DedupIOError):
        list(reader.stream(pattern))


# ----- gzip JSONL ---------------------------------------------------------


def test_stream_gzip_jsonl(tmp_path: Path) -> None:
    """A .jsonl.gz file is transparently decompressed and streamed.

    Matters because corpora are almost always stored gzipped to save space; the
    reader must detect the .gz suffix and decode through gzip so compressed
    shards are first-class inputs with no manual decompression step.
    """
    path = tmp_path / "corpus.jsonl.gz"
    payload = "\n".join(
        [json.dumps({"id": "g1", "text": "zipped one"}), json.dumps({"id": "g2", "text": "zipped two"})]
    ) + "\n"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(payload)
    reader = DocumentReader(PipelineConfig())
    docs = list(reader.stream(path))
    assert [(d.id, d.text) for d in docs] == [("g1", "zipped one"), ("g2", "zipped two")]


# ----- Parquet ------------------------------------------------------------


def test_stream_parquet_file(tmp_path: Path) -> None:
    """A Parquet file is streamed by row batches into Documents.

    Matters because Parquet is the columnar production format; the reader must
    materialise rows as record dicts and apply the same text/id/metadata rules so
    Parquet and JSONL inputs are interchangeable.
    """
    path = tmp_path / "corpus.parquet"
    table = pa.table({"id": ["p1", "p2", "p3"], "text": ["alpha", "beta", "gamma"]})
    pq.write_table(table, path)
    reader = DocumentReader(PipelineConfig())
    docs = list(reader.stream(path))
    assert [(d.id, d.text) for d in docs] == [("p1", "alpha"), ("p2", "beta"), ("p3", "gamma")]


# ----- Unsupported format -------------------------------------------------


def test_unsupported_extension_raises_io_error(tmp_path: Path) -> None:
    """An existing file with an unsupported extension raises the package IOError.

    Matters because feeding an unrecognised format (e.g. a .txt or .csv shard)
    must fail with a clear, typed error at ingestion rather than being parsed
    incorrectly and producing a corrupted document stream.
    """
    path = tmp_path / "corpus.txt"
    path.write_text("just some plain text\n", encoding="utf-8")
    reader = DocumentReader(PipelineConfig())
    with pytest.raises(DedupIOError):
        list(reader.stream(path))


# ----- Pathological / edge cases -----------------------------------------


def test_empty_jsonl_file_yields_nothing(tmp_path: Path) -> None:
    """A completely empty JSONL file yields zero Documents without error.

    Pathological case: an empty (zero-byte) shard is a legitimate corner of a
    sharded corpus. Matters because the reader must treat it as "no documents
    here", not as an error, so an empty shard among many does not abort the run.
    """
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    reader = DocumentReader(PipelineConfig())
    assert list(reader.stream(path)) == []


def test_jsonl_with_only_blank_lines_yields_nothing(tmp_path: Path) -> None:
    """A JSONL file of only blank/whitespace lines yields zero Documents.

    Pathological case: trailing newlines and blank separator lines are common in
    hand-edited or concatenated shards. Matters because blank lines must be
    skipped (not parsed as malformed JSON and not counted as documents), so
    whitespace formatting never injects phantom records or warnings that distort
    the input count.
    """
    path = tmp_path / "blanks.jsonl"
    path.write_text("\n   \n\t\n\n", encoding="utf-8")
    reader = DocumentReader(PipelineConfig())
    assert list(reader.stream(path)) == []
