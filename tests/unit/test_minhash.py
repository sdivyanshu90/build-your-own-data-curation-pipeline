"""Unit tests for :class:`MinHasher`.

These verify the statistical contract of MinHash — unbiasedness and the
``sqrt(J(1-J)/n)`` standard error — plus operational guarantees: batch/transform
parity, reproducibility, and correct handling of degenerate sets.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from dedup_pipeline.exceptions import HashingError
from dedup_pipeline.minhash.minhash import EMPTY_SIGNATURE_VALUE, MinHasher


def test_unbiasedness_over_many_pairs() -> None:
    """Mean estimated Jaccard tracks mean true Jaccard within 0.02.

    Matters because MinHash's whole value rests on being an *unbiased* estimator;
    a systematic offset would bias every dedup decision.
    """
    rng = random.Random(0)
    minhasher = MinHasher(num_hash_functions=256, seed=42).fit()
    estimates: list[float] = []
    truths: list[float] = []
    for _ in range(1000):
        size = rng.randint(20, 200)
        a = set(rng.sample(range(100_000), size))
        keep = rng.random()
        b = {x for x in a if rng.random() < keep}
        b |= set(rng.sample(range(100_000, 200_000), rng.randint(0, size)))
        if not a or not b:
            continue
        truths.append(len(a & b) / len(a | b))
        estimates.append(
            MinHasher.estimate_jaccard(minhasher.transform(a), minhasher.transform(b))
        )
    assert abs(float(np.mean(estimates)) - float(np.mean(truths))) < 0.02


def test_variance_matches_theory() -> None:
    """Empirical estimator std matches sqrt(J(1-J)/n) within 20%.

    Matters because the error bound is what justifies the choice of
    num_hash_functions; if the realized variance were larger, accuracy claims
    would be wrong.
    """
    a = set(range(100))
    b = set(range(50, 150))
    true_j = len(a & b) / len(a | b)
    n = 128
    estimates = [
        MinHasher.estimate_jaccard(
            MinHasher(n, seed=s).fit().transform(a),
            MinHasher(n, seed=s).fit().transform(b),
        )
        for s in range(500)
    ]
    empirical_std = float(np.std(estimates))
    theoretical_std = math.sqrt(true_j * (1 - true_j) / n)
    assert abs(empirical_std - theoretical_std) / theoretical_std < 0.2


def test_batch_parity() -> None:
    """batch_transform equals stacking individual transform calls.

    Matters because batching is a performance optimisation that must never change
    results.
    """
    minhasher = MinHasher(128, seed=42).fit()
    a = {1, 2, 3, 4, 5}
    b = {3, 4, 5, 6, 7}
    batch = minhasher.batch_transform([a, b])
    assert np.array_equal(batch[0], minhasher.transform(a))
    assert np.array_equal(batch[1], minhasher.transform(b))


def test_identical_sets_estimate_one() -> None:
    """Identical sets estimate Jaccard exactly 1.0.

    Matters because exact duplicates must be detected with certainty.
    """
    minhasher = MinHasher(128, seed=42).fit()
    s = {10, 20, 30, 40}
    assert MinHasher.estimate_jaccard(minhasher.transform(s), minhasher.transform(s)) == 1.0


def test_disjoint_sets_estimate_near_zero() -> None:
    """Disjoint sets estimate Jaccard ~0.

    Matters because unrelated documents must not be flagged as duplicates.
    """
    minhasher = MinHasher(256, seed=42).fit()
    a = set(range(0, 100))
    b = set(range(1000, 1100))
    estimate = MinHasher.estimate_jaccard(minhasher.transform(a), minhasher.transform(b))
    assert estimate < 0.05


def test_reproducibility_same_seed() -> None:
    """Same config and input produce identical signatures.

    Matters because reproducible signatures are required for resumable runs and
    debuggability.
    """
    s = {1, 2, 3, 4, 5, 6}
    sig_a = MinHasher(128, seed=42).fit().transform(s)
    sig_b = MinHasher(128, seed=42).fit().transform(s)
    assert np.array_equal(sig_a, sig_b)


def test_empty_set_sentinel() -> None:
    """An empty shingle set yields the all-sentinel signature (pathological).

    Matters because empty documents must produce a well-defined signature instead
    of crashing the vectorized math.
    """
    minhasher = MinHasher(16, seed=42).fit()
    sig = minhasher.transform(set())
    assert sig.tolist() == [EMPTY_SIGNATURE_VALUE] * 16


def test_two_empty_sets_are_duplicates() -> None:
    """Two empty sets estimate Jaccard 1.0 (pathological input).

    Matters because empty/junk documents should cluster together rather than
    scatter into spurious singletons.
    """
    minhasher = MinHasher(16, seed=42).fit()
    assert MinHasher.estimate_jaccard(minhasher.transform(set()), minhasher.transform(set())) == 1.0


def test_single_element_set() -> None:
    """A one-element set produces a valid, repeatable signature (pathological).

    Matters because very short documents are common and must be handled.
    """
    minhasher = MinHasher(32, seed=42).fit()
    sig = minhasher.transform({99})
    assert sig.shape == (32,)
    assert np.array_equal(sig, minhasher.batch_transform([{99}])[0])


def test_signature_dtype_and_shape() -> None:
    """The batch signature matrix is (n_docs, n_hashes) uint32.

    Matters because downstream banding/storage assume this exact layout and dtype.
    """
    minhasher = MinHasher(64, seed=42).fit()
    sig = minhasher.batch_transform([{1, 2}, {3, 4}, set()])
    assert sig.shape == (3, 64)
    assert sig.dtype == np.uint32


def test_numba_path_matches_numpy_path() -> None:
    """The Numba JIT batch path produces identical signatures to NumPy.

    Matters because the optional fast path must be a pure drop-in optimisation;
    any divergence would silently change dedup results when the flag is enabled.
    """
    pytest.importorskip("numba")
    docs = [{1, 2, 3}, {3, 4, 5, 6}, set(), {7, 8}]
    plain = MinHasher(64, seed=42, use_numba=False).fit().batch_transform(docs)
    jitted = MinHasher(64, seed=42, use_numba=True).fit().batch_transform(docs)
    assert np.array_equal(plain, jitted)


def test_transform_before_fit_raises() -> None:
    """Calling transform before fit raises HashingError (pathological usage).

    Matters because using an unfitted hasher would otherwise dereference a None
    hash family and crash opaquely.
    """
    with pytest.raises(HashingError):
        MinHasher(8, seed=42).transform({1, 2, 3})


def test_estimate_jaccard_shape_mismatch_raises() -> None:
    """Comparing signatures of different lengths raises HashingError.

    Matters because a length mismatch indicates inconsistent configuration; a
    silent miscomparison would corrupt similarity estimates.
    """
    a = np.array([1, 2, 3, 4], dtype=np.uint32)
    b = np.array([1, 2, 3], dtype=np.uint32)
    with pytest.raises(HashingError):
        MinHasher.estimate_jaccard(a, b)


def test_is_fitted_and_num_hash_functions() -> None:
    """is_fitted flips on fit() and num_hash_functions reports the length.

    Matters because callers (and the pipeline) rely on these accessors to wire
    up banding and storage with the correct signature width.
    """
    minhasher = MinHasher(32, seed=1)
    assert not minhasher.is_fitted
    assert minhasher.num_hash_functions == 32
    minhasher.fit()
    assert minhasher.is_fitted
