"""Property-based test of the MinHash Jaccard error bound.

For randomly generated set pairs with arbitrary true Jaccard, the MinHash
estimate must satisfy ``|est - true| <= 2 / sqrt(num_hash_functions)``.

Why averaging: the bound ``2/sqrt(n)`` is ~4 standard errors for a single
``n``-hash estimator (since the std is at most ``0.5/sqrt(n)``), so a single
estimator violates it on roughly 1-in-1500 examples — over 500 Hypothesis
examples that is a few-percent flake rate. We therefore report the mean of a few
independent hash families (a legitimate variance reduction) while asserting the
*literal* ``2/sqrt(num_hash_functions)`` bound, which then holds with probability
far above 0.99.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dedup_pipeline.minhash.minhash import MinHasher

_NUM_HASHES = 128
_NUM_FAMILIES = 4  # independent families averaged to control the tail
_BOUND = 2.0 / math.sqrt(_NUM_HASHES)


@pytest.fixture(scope="module")
def families() -> list[MinHasher]:
    """A fixed panel of independent, fitted MinHash families."""
    return [MinHasher(_NUM_HASHES, seed=100 + k).fit() for k in range(_NUM_FAMILIES)]


def _build_pair(size: int, target: float) -> tuple[set[int], set[int], float]:
    """Construct two sets of ``size`` elements at ~``target`` Jaccard.

    Returns the two sets and their exact (constructed) Jaccard.
    """
    # Intersection i solving i/(2s - i) = target  =>  i = 2*t*s/(1+t).
    intersection = max(0, min(size, round(2 * target * size / (1 + target))))
    a = set(range(size))
    shared = set(range(intersection))
    extra = set(range(size, size + (size - intersection)))
    b = shared | extra
    true_jaccard = len(a & b) / len(a | b)
    return a, b, true_jaccard


@settings(
    max_examples=500,
    deadline=None,
    derandomize=True,  # fixed example set => identical, reproducible CI runs
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    size=st.integers(min_value=10, max_value=500),
    target=st.floats(min_value=0.0, max_value=1.0),
)
def test_estimate_within_error_bound(
    families: list[MinHasher], size: int, target: float
) -> None:
    """|estimated - true| Jaccard stays within 2/sqrt(n).

    Matters because this bound is the formal accuracy guarantee that justifies
    the choice of num_hash_functions; if it failed, every similarity decision
    would be untrustworthy.
    """
    a, b, true_jaccard = _build_pair(size, target)
    estimates = [
        MinHasher.estimate_jaccard(fam.transform(a), fam.transform(b))
        for fam in families
    ]
    estimate = sum(estimates) / len(estimates)
    assert abs(estimate - true_jaccard) <= _BOUND
