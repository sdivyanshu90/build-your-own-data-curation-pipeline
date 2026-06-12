"""Unit tests for :class:`DeduplicatedWriter` (Stage 10, output + statistics).

The writer is the pipeline's final act: it re-streams the original source and
emits only the documents whose stream index is in the keep-set, plus a
statistics sidecar that is the run's permanent provenance record. Three
properties are load-bearing: (1) *selectivity and fidelity* — exactly the kept
documents are written, verbatim (original text, not the internal normalized
form), so deduplication only ever removes documents and never alters survivors;
(2) *honest accounting* — the returned stats report output_count, the passed-in
input_count, and dedup_rate = (input - output) / input, and the sidecar JSON is
well-formed and complete; and (3) *format correctness* — both JSONL and Parquet
backends produce a readable file with the kept rows. If any of these slip,
either good data is dropped/altered or the run's reported dedup statistics lie.
These tests pin all three across both output formats and the empty/keep-all
edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dedup_pipeline.config import PipelineConfig
from dedup_pipeline.pipeline.reader import DocumentReader
from dedup_pipeline.pipeline.writer import DeduplicatedWriter

# Three-record source reused across the JSONL tests.
_SOURCE = [
    {"id": "d0", "text": "first document"},
    {"id": "d1", "text": "second document"},
    {"id": "d2", "text": "third document"},
]
_HISTOGRAM = {2: 3, 5: 1}
_RUNTIME = {"read": 0.1, "minhash": 0.2, "total": 0.5}


def _make_writer(config: PipelineConfig) -> DeduplicatedWriter:
    """Build a writer wired to a reader on the same config."""
    return DeduplicatedWriter(config, DocumentReader(config))


def _read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL output file into a list of record dicts via tmp_path."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ----- JSONL output: selectivity, fidelity, and returned stats -----------


def test_jsonl_writes_only_kept_records_verbatim(tmp_path: Path) -> None:
    """keep={0,2} writes exactly 2 valid-JSON lines preserving original text.

    Matters because this is the deduplication contract: only the kept stream
    indices survive, each survivor is the original record (verbatim text, not the
    normalized form used internally), and the file is valid JSONL. A drift here
    means dropping kept docs or mutating their content.
    """
    writer = _make_writer(PipelineConfig())
    dest = tmp_path / "out.jsonl"
    writer.write({0, 2}, _SOURCE, dest, input_count=3, cluster_size_histogram=_HISTOGRAM, runtime_per_stage=_RUNTIME)
    records = _read_jsonl(dest)
    assert len(records) == 2
    assert [r["text"] for r in records] == ["first document", "third document"]
    # The kept ids are preserved from the source metadata.
    assert [r["id"] for r in records] == ["d0", "d2"]


def test_jsonl_returned_stats_have_correct_counts_and_rate(tmp_path: Path) -> None:
    """write() returns stats with output_count, the passed input_count, and dedup_rate.

    Matters because these returned numbers are what the caller logs and trusts as
    the run summary; dedup_rate must equal (input - output) / input exactly, or
    the reported "fraction removed" misrepresents the result.
    """
    writer = _make_writer(PipelineConfig())
    dest = tmp_path / "out.jsonl"
    stats = writer.write({0, 2}, _SOURCE, dest, input_count=3, cluster_size_histogram=_HISTOGRAM, runtime_per_stage=_RUNTIME)
    assert stats["output_count"] == 2
    assert stats["input_count"] == 3
    assert stats["dedup_rate"] == pytest.approx((3 - 2) / 3)


def test_stats_sidecar_file_is_valid_json_with_required_keys(tmp_path: Path) -> None:
    """The stats sidecar exists, is valid JSON, and holds all required keys.

    Matters because the sidecar is the run's permanent provenance record (used to
    reproduce and audit a deduplication); a missing key or malformed JSON would
    silently lose the input/output counts, dedup rate, cluster histogram, timing,
    or config that justify the output corpus.
    """
    writer = _make_writer(PipelineConfig())
    dest = tmp_path / "out.jsonl"
    writer.write({0, 2}, _SOURCE, dest, input_count=3, cluster_size_histogram=_HISTOGRAM, runtime_per_stage=_RUNTIME)
    stats_path = writer.stats_path_for(dest)
    assert stats_path.exists()
    loaded = json.loads(stats_path.read_text(encoding="utf-8"))
    for key in ("input_count", "output_count", "dedup_rate", "cluster_size_histogram", "runtime_per_stage", "config"):
        assert key in loaded, f"stats sidecar missing key {key!r}"
    assert loaded["input_count"] == 3
    assert loaded["output_count"] == 2


def test_stats_path_for_sits_next_to_dest(tmp_path: Path) -> None:
    """stats_path_for(dest) returns '<stem>_stats.json' beside the destination.

    Matters because tooling discovers the stats file by this naming convention;
    if the sidecar landed elsewhere or under a different name, downstream
    reporting could not pair a corpus with its provenance.
    """
    writer = _make_writer(PipelineConfig())
    dest = tmp_path / "subdir" / "result.jsonl"
    expected = tmp_path / "subdir" / "result_stats.json"
    assert writer.stats_path_for(dest) == expected


def test_cluster_histogram_keys_serialized_as_strings(tmp_path: Path) -> None:
    """The cluster_size_histogram is serialized with string keys in the JSON.

    Matters because JSON object keys must be strings; the writer must coerce the
    integer cluster sizes to text or json.dump would either crash or (in Python)
    silently stringify them inconsistently, corrupting the histogram that
    summarises duplicate-group sizes.
    """
    writer = _make_writer(PipelineConfig())
    dest = tmp_path / "out.jsonl"
    stats = writer.write({0}, _SOURCE, dest, input_count=3, cluster_size_histogram={2: 3, 5: 1}, runtime_per_stage=_RUNTIME)
    # The in-memory returned histogram already uses string keys.
    assert set(stats["cluster_size_histogram"].keys()) == {"2", "5"}
    # And the on-disk JSON parses those keys back as strings (JSON keys are always strings).
    on_disk = json.loads(writer.stats_path_for(dest).read_text(encoding="utf-8"))
    assert on_disk["cluster_size_histogram"] == {"2": 3, "5": 1}


# ----- Parquet output -----------------------------------------------------


def test_parquet_output_is_readable_with_kept_rows(tmp_path: Path) -> None:
    """output_format='parquet' writes a readable Parquet file with the kept rows.

    Matters because Parquet is the columnar production output; the writer must
    emit a valid Parquet file containing exactly the kept documents so the
    deduplicated corpus is consumable by columnar tooling, and the reported
    output_count must match the row count actually on disk.
    """
    config = PipelineConfig(output_format="parquet")
    writer = _make_writer(config)
    dest = tmp_path / "out.parquet"
    stats = writer.write({0, 2}, _SOURCE, dest, input_count=3, cluster_size_histogram=_HISTOGRAM, runtime_per_stage=_RUNTIME)
    assert stats["output_count"] == 2
    table = pq.read_table(dest)
    assert table.num_rows == 2
    assert table.column("text").to_pylist() == ["first document", "third document"]


# ----- Pathological / edge cases -----------------------------------------


def test_keep_empty_set_writes_zero_documents(tmp_path: Path) -> None:
    """keep=set() writes an empty corpus; dedup_rate uses input_count > 0.

    Pathological case: a (degenerate) run that keeps nothing must still produce a
    valid output file and an honest dedup_rate of 1.0 (everything removed) rather
    than dividing by zero or crashing. Matters because the writer must never
    leave the destination missing and must report the removal fraction even at the
    extreme.
    """
    writer = _make_writer(PipelineConfig())
    dest = tmp_path / "empty_out.jsonl"
    stats = writer.write(set(), _SOURCE, dest, input_count=3, cluster_size_histogram={}, runtime_per_stage=_RUNTIME)
    assert stats["output_count"] == 0
    assert stats["dedup_rate"] == pytest.approx(1.0)
    assert dest.exists()
    assert _read_jsonl(dest) == []


def test_keep_all_indices_writes_every_document(tmp_path: Path) -> None:
    """keep=all indices writes every document; output_count == input and rate 0.

    Pathological case: a corpus with no duplicates keeps everything. Matters
    because the writer must pass through all documents untouched (dedup_rate 0.0)
    so a clean corpus is reported as such and no document is accidentally dropped
    when nothing should be removed.
    """
    writer = _make_writer(PipelineConfig())
    dest = tmp_path / "all_out.jsonl"
    stats = writer.write({0, 1, 2}, _SOURCE, dest, input_count=3, cluster_size_histogram={}, runtime_per_stage=_RUNTIME)
    assert stats["output_count"] == 3
    assert stats["input_count"] == 3
    assert stats["dedup_rate"] == pytest.approx(0.0)
    assert len(_read_jsonl(dest)) == 3
