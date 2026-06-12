"""Shared pytest fixtures for the deduplication test suite.

Provides reproducible synthetic corpora with known ground-truth duplicate pairs,
a deterministic configuration, and a temporary output directory. All randomness
is seeded so every fixture is byte-stable across runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pytest

from dedup_pipeline.config import PipelineConfig
from dedup_pipeline.evaluation.synthetic_injector import (
    inject_exact_duplicates,
    inject_near_duplicates,
)

# Word-set target whose realized word-3-gram *shingle* Jaccard lands near 0.85
# (calibrated): each token edit breaks ~k shingles, so the shingle Jaccard is
# lower than the word-set Jaccard.
_NEAR_DUP_WORD_TARGET = 0.96


@dataclass(frozen=True)
class Corpus:
    """A synthetic corpus with its known duplicate ground truth.

    Attributes:
        records: The list of ``{"id", "text"}`` record dicts.
        ground_truth: Canonical ``(i, j)`` index pairs that are true duplicates.
    """

    records: list[dict[str, str]]
    ground_truth: set[tuple[int, int]]


def make_unique_records(
    n: int, seed: int = 0, words: int = 60, vocab: int = 4000
) -> list[dict[str, str]]:
    """Build ``n`` mutually-distinct documents from a large random vocabulary.

    Args:
        n: Number of documents.
        seed: RNG seed.
        words: Words per document.
        vocab: Vocabulary size (large enough that documents rarely collide).

    Returns:
        A list of ``{"id", "text"}`` records.
    """
    rng = random.Random(seed)
    return [
        {
            "id": f"u{i}",
            "text": " ".join(f"w{rng.randint(0, vocab)}" for _ in range(words)),
        }
        for i in range(n)
    ]


@pytest.fixture
def tiny_corpus() -> Corpus:
    """20 documents containing 5 injected exact-duplicate pairs.

    Used by fast unit/integration tests that need a known, small, exact-duplicate
    structure.
    """
    base = make_unique_records(15, seed=101)
    records, ground_truth = inject_exact_duplicates(base, num_pairs=5, seed=101)
    return Corpus(records=records, ground_truth=ground_truth)


@pytest.fixture
def medium_corpus() -> Corpus:
    """1,000 documents containing 50 injected near-duplicate pairs (J ~ 0.85).

    Used by integration tests that exercise near-duplicate detection and timing.
    """
    base = make_unique_records(950, seed=202, words=80)
    records, ground_truth = inject_near_duplicates(
        base, num_pairs=50, target_jaccard=_NEAR_DUP_WORD_TARGET, seed=202
    )
    return Corpus(records=records, ground_truth=ground_truth)


@pytest.fixture
def default_config() -> PipelineConfig:
    """A deterministic configuration tuned for fast, reproducible tests."""
    return PipelineConfig(
        num_hash_functions=128,
        lsh_bands=16,
        lsh_rows=8,
        shingle_mode="word",
        shingle_size=3,
        jaccard_threshold=0.7,
        random_seed=42,
    )


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """A fresh, ``tmp_path``-backed output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out
