"""Unit tests for the hash primitives.

The MinHash estimator is only unbiased if its hash family behaves like random
permutations and the shingle content hash is well-distributed. These tests
verify uniformity, reproducibility, avalanche, distinctness, and speed.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy import stats

from dedup_pipeline.exceptions import HashingError
from dedup_pipeline.minhash.hash_functions import (
    HASH_MODULUS,
    UniversalHashFamily,
    stable_hash64,
)


def test_chi_squared_uniformity() -> None:
    """Hash outputs are ~uniform over [0, P) (chi-squared not rejected).

    Matters because non-uniform hashing biases MinHash collision probabilities
    and corrupts every similarity estimate.
    """
    family = UniversalHashFamily(num_functions=1, seed=7)
    rng = np.random.default_rng(0)
    inputs = rng.integers(0, 2**63, size=200_000, dtype=np.uint64)
    outputs = family.hash_values(0, inputs)
    num_bins = 256
    counts, _ = np.histogram(outputs, bins=num_bins, range=(0, HASH_MODULUS))
    # Uniform null: equal expected count per bin.
    _, p_value = stats.chisquare(counts)
    assert p_value > 0.01


def test_seed_reproducibility_same_seed() -> None:
    """Same seed yields byte-identical coefficient arrays.

    Matters because reproducibility of an entire run hinges on the hash family
    being a pure function of the seed.
    """
    fam_a = UniversalHashFamily(64, seed=42)
    fam_b = UniversalHashFamily(64, seed=42)
    assert np.array_equal(fam_a.a, fam_b.a)
    assert np.array_equal(fam_a.b, fam_b.b)


def test_seed_reproducibility_different_seed() -> None:
    """Different seeds yield different coefficient arrays.

    Matters because independent runs/shards must be able to decorrelate their
    hashing when desired.
    """
    fam_a = UniversalHashFamily(64, seed=1)
    fam_b = UniversalHashFamily(64, seed=2)
    assert not np.array_equal(fam_a.a, fam_b.a)


def test_avalanche_effect() -> None:
    """A one-character input change flips ~half of the 64 output bits.

    Matters because a weak content hash would map near-duplicate shingles to
    correlated ids, leaking structure into the signatures.
    """
    rng = np.random.default_rng(3)
    flipped_fractions = []
    for _ in range(500):
        base = "".join(chr(rng.integers(97, 123)) for _ in range(12))
        variant = base[:-1] + chr((ord(base[-1]) - 97 + 1) % 26 + 97)
        diff_bits = bin(stable_hash64(base) ^ stable_hash64(variant)).count("1")
        flipped_fractions.append(diff_bits / 64.0)
    assert 0.4 < float(np.mean(flipped_fractions)) < 0.6


def test_hash_family_all_distinct() -> None:
    """No two functions in a family share the same (a, b) coefficients.

    Matters because duplicate hash functions waste signature slots and shrink the
    effective number of independent estimates.
    """
    family = UniversalHashFamily(256, seed=11)
    pairs = set(zip(family.a.tolist(), family.b.tolist(), strict=True))
    assert len(pairs) == 256


def test_performance_10k_shingles() -> None:
    """Hashing 10,000 shingles completes well under 100 ms.

    Matters because shingle hashing runs once per shingle across the whole
    corpus; it must not be a bottleneck.
    """
    shingles = [f"shingle-{i}-token" for i in range(10_000)]
    start = time.perf_counter()
    for s in shingles:
        stable_hash64(s, seed=42)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.1


def test_stable_hash_is_deterministic() -> None:
    """The content hash is stable across calls for the same input.

    Matters because signatures must be reproducible across processes; a salted
    hash would break that.
    """
    assert stable_hash64("hello", seed=5) == stable_hash64("hello", seed=5)


def test_stable_hash_empty_string() -> None:
    """Hashing the empty string is valid (pathological input).

    Matters because empty shingles can arise from degenerate documents.
    """
    assert isinstance(stable_hash64("", seed=0), int)


def test_stable_hash_non_str_raises() -> None:
    """A non-string input raises HashingError (pathological input).

    Matters because a type error must surface as a domain error, not a raw crash.
    """
    with pytest.raises(HashingError):
        stable_hash64(12345)  # type: ignore[arg-type]


def test_family_zero_functions_raises() -> None:
    """Requesting zero hash functions raises HashingError (pathological input).

    Matters because an empty family would silently produce zero-length signatures.
    """
    with pytest.raises(HashingError):
        UniversalHashFamily(0, seed=0)


def test_hash_values_out_of_range_raises() -> None:
    """An out-of-range function index raises HashingError.

    Matters because silently wrapping the index would mix up hash functions.
    """
    family = UniversalHashFamily(4, seed=0)
    with pytest.raises(HashingError):
        family.hash_values(4, np.array([1], dtype=np.uint64))


def test_hash_values_within_modulus() -> None:
    """Every hash output lies in [0, P).

    Matters because signatures are stored as uint32 and must not exceed the
    modulus.
    """
    family = UniversalHashFamily(8, seed=9)
    out = family.hash_values(0, np.arange(1000, dtype=np.uint64))
    assert int(out.max()) < HASH_MODULUS
