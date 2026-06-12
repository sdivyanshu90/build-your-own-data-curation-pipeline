"""End-to-end integration test: exact-duplicate removal.

Injects 50 exact duplicates into a 500-document corpus (550 total) and runs the
full pipeline, asserting recall, precision, output count, output validity, and
the statistics sidecar — the headline correctness contract of the system.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from dedup_pipeline.config import PipelineConfig
from dedup_pipeline.evaluation.metrics import pairs_from_clusters, precision_recall_f1
from dedup_pipeline.evaluation.synthetic_injector import inject_exact_duplicates
from dedup_pipeline.pipeline.pipeline import DedupPipeline


def _unique_records(n: int, seed: int, words: int = 70, vocab: int = 4000) -> list[dict[str, str]]:
    """Build n mutually-distinct documents from a large random vocabulary."""
    rng = random.Random(seed)
    return [
        {"id": f"u{i}", "text": " ".join(f"w{rng.randint(0, vocab)}" for _ in range(words))}
        for i in range(n)
    ]


@pytest.fixture
def exact_config() -> PipelineConfig:
    """A config tuned for exact-duplicate detection on word shingles."""
    return PipelineConfig(
        num_hash_functions=128,
        lsh_bands=16,
        lsh_rows=8,
        shingle_mode="word",
        shingle_size=3,
        jaccard_threshold=0.7,
        random_seed=42,
    )


@pytest.mark.integration
def test_exact_dedup_recall_precision(exact_config: PipelineConfig) -> None:
    """Recall >= 0.98 and precision >= 0.95 on 50 injected exact-duplicate pairs.

    Matters because exact duplicates are the easiest case; failing here means the
    core MinHash/LSH path is broken.
    """
    base = _unique_records(500, seed=1)
    records, ground_truth = inject_exact_duplicates(base, num_pairs=50, seed=1)
    assert len(records) == 550 and len(ground_truth) == 50

    clusters, n_docs = DedupPipeline(exact_config).detect_clusters(records)
    precision, recall, _ = precision_recall_f1(
        pairs_from_clusters(clusters), ground_truth
    )
    assert n_docs == 550
    assert recall >= 0.98  # at most one missed cluster
    assert precision >= 0.95  # at most a 5% false-positive rate


@pytest.mark.integration
def test_exact_dedup_output_and_stats(
    exact_config: PipelineConfig, tmp_path: Path
) -> None:
    """Output count is input-50, the file is valid JSONL, and stats are written.

    Matters because the user-facing artifact (the cleaned corpus + stats) must be
    correct and machine-readable, not just the in-memory clustering.
    """
    base = _unique_records(500, seed=2)
    records, _ = inject_exact_duplicates(base, num_pairs=50, seed=2)
    dest = tmp_path / "out.jsonl"

    stats = DedupPipeline(exact_config).run(records, dest)

    assert stats["input_count"] == 550
    assert stats["output_count"] == 550 - 50  # one removed from each pair
    assert stats["dedup_rate"] == pytest.approx(50 / 550, abs=0.005)

    lines = dest.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 500
    for line in lines:  # every output line must be parseable JSON
        assert isinstance(json.loads(line), dict)

    stats_file = dest.with_name("out_stats.json")
    assert stats_file.exists()
    loaded = json.loads(stats_file.read_text(encoding="utf-8"))
    assert loaded["dedup_rate"] == pytest.approx(0.09, abs=0.01)


@pytest.mark.integration
def test_all_unique_corpus_removes_nothing(exact_config: PipelineConfig, tmp_path: Path) -> None:
    """An all-unique corpus is passed through unchanged (pathological input).

    Matters because the deduplicator must never remove non-duplicates; a corpus
    with no duplicates must come out the same size.
    """
    records = _unique_records(200, seed=3)
    dest = tmp_path / "out.jsonl"
    stats = DedupPipeline(exact_config).run(records, dest)
    assert stats["output_count"] == 200
    assert stats["dedup_rate"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.integration
def test_all_identical_corpus_collapses_to_one(exact_config: PipelineConfig, tmp_path: Path) -> None:
    """An all-identical corpus collapses to a single document (pathological input).

    Matters because a giant boilerplate cluster (e.g. a repeated cookie banner)
    must reduce to one representative, not survive en masse.
    """
    records = [{"id": str(i), "text": "the same boilerplate text repeated"} for i in range(50)]
    dest = tmp_path / "out.jsonl"
    stats = DedupPipeline(exact_config).run(records, dest)
    assert stats["output_count"] == 1
