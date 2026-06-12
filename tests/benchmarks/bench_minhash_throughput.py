"""Throughput benchmark for MinHash signature computation.

Measures documents/second of ``MinHasher.batch_transform`` at 128 hash functions
for corpus sizes of 1k, 10k, and 100k documents, and asserts a hard floor of
5,000 docs/sec on a standard laptop CPU.

Regression gate:
    Run the suite with ``pytest-benchmark``'s comparison to fail CI on >20%
    regressions, e.g. store a baseline with ``--benchmark-save=baseline`` and
    enforce ``--benchmark-compare=baseline --benchmark-compare-fail=mean:20%``.
    The absolute 5,000 docs/sec floor below is the always-on safety net.
"""

from __future__ import annotations

import random

import pytest

from dedup_pipeline.minhash.minhash import MinHasher

_NUM_HASHES = 128
_SHINGLES_PER_DOC = 20
_MIN_THROUGHPUT = 5_000  # docs/sec floor


def _make_shingle_sets(n_docs: int, seed: int = 0) -> list[set[int]]:
    """Build ``n_docs`` small random shingle sets."""
    rng = random.Random(seed)
    return [
        {rng.randrange(10_000_000) for _ in range(_SHINGLES_PER_DOC)}
        for _ in range(n_docs)
    ]


@pytest.mark.benchmark
@pytest.mark.parametrize("n_docs", [1_000, 10_000, 100_000])
def test_minhash_throughput(benchmark: object, n_docs: int) -> None:
    """batch_transform sustains >= 5,000 docs/sec at 128 hash functions.

    Matters because throughput is the difference between a pipeline that finishes
    a 100M-document corpus in hours versus days; a regression here is a
    production-blocking issue.
    """
    shingle_sets = _make_shingle_sets(n_docs)
    minhasher = MinHasher(_NUM_HASHES, seed=42).fit()

    result = benchmark(minhasher.batch_transform, shingle_sets)  # type: ignore[operator]

    # Correctness holds regardless of whether timing stats were collected.
    assert result.shape == (n_docs, _NUM_HASHES)

    # Under coverage, pytest-benchmark disables timing; stats are then absent.
    stats = getattr(benchmark, "stats", None)
    inner = getattr(stats, "stats", None) if stats is not None else None
    if inner is None:
        pytest.skip("benchmark timing disabled (e.g. running under coverage)")

    mean_seconds = inner.mean
    throughput = n_docs / mean_seconds if mean_seconds > 0 else float("inf")
    assert throughput >= _MIN_THROUGHPUT, f"{throughput:.0f} docs/sec < {_MIN_THROUGHPUT}"
