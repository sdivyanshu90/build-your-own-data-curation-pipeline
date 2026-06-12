"""Unit tests for candidate-pair enumeration and pair-key encoding.

Stage 6 turns the inverted bucket index into a stream of unique, canonically
ordered candidate pairs ``(i, j)`` with ``i < j``. The same pair can surface in
several bands' buckets, so a Bloom filter (or an exact set) must suppress
duplicates. These tests pin the canonical ordering, cross-bucket deduplication,
the exact-vs-Bloom equivalence at small scale, and the order-sensitive pair
encoding that the dedup set relies on for uniqueness.
"""

from __future__ import annotations

import numpy as np  # noqa: F401 - imported per suite convention for array helpers

from dedup_pipeline.lsh.bucket_index import BucketIndex
from dedup_pipeline.lsh.candidate_pairs import (
    encode_pair,
    enumerate_candidate_pairs,
)


def test_pairs_emitted_with_canonical_ordering() -> None:
    """Every emitted pair has i < j even when bucket docs are unsorted.

    Matters because the union-find/merge stage assumes canonical ordering; an
    out-of-order or reversed pair would key the dedup set inconsistently and
    could merge the wrong components.
    """
    idx = BucketIndex()
    idx.add_bucket(1, [2, 0, 1])  # deliberately unsorted
    pairs = list(enumerate_candidate_pairs(idx, False, 10, 0.01))
    assert all(i < j for i, j in pairs)
    assert sorted(pairs) == [(0, 1), (0, 2), (1, 2)]


def test_pair_in_multiple_buckets_yielded_once() -> None:
    """A pair appearing in two buckets is emitted only once.

    Matters because identical documents collide in many bands; without
    cross-bucket dedup the candidate stream would contain heavy duplicates,
    inflating downstream verification cost and double-counting.
    """
    idx = BucketIndex()
    idx.add_bucket(1, [0, 1])
    idx.add_bucket(2, [0, 1])  # same pair (0, 1) again
    pairs = list(enumerate_candidate_pairs(idx, False, 10, 0.01))
    assert pairs == [(0, 1)]
    assert len(pairs) == len(set(pairs))


def test_exact_dedup_produces_full_pair_set() -> None:
    """Exact mode yields the complete C(k,2) pair set for a bucket.

    Matters because precision/recall of dedup depends on enumerating *all* pairs
    within a candidate bucket; missing one would skip a real duplicate
    comparison. Bucket [0,1,2] must yield exactly {(0,1),(0,2),(1,2)}.
    """
    idx = BucketIndex()
    idx.add_bucket(1, [0, 1, 2])
    pairs = set(enumerate_candidate_pairs(idx, False, 10, 0.01))
    assert pairs == {(0, 1), (0, 2), (1, 2)}


def test_bloom_dedup_matches_exact_at_small_scale() -> None:
    """Bloom mode yields the same full pair set as exact mode at small scale.

    Matters because the Bloom path is the production memory-bounded dedup; at low
    occupancy its false-positive rate is negligible, so it must not drop any of
    the few pairs in a tiny bucket.
    """
    idx = BucketIndex()
    idx.add_bucket(1, [0, 1, 2])
    pairs = set(enumerate_candidate_pairs(idx, True, 1000, 0.01))
    assert pairs == {(0, 1), (0, 2), (1, 2)}


def test_encode_pair_is_order_sensitive_and_unique() -> None:
    """encode_pair packs (i, j) as (i<<32)|j and is order-sensitive.

    Matters because the dedup set keys on this encoding; if encode_pair(1,5) and
    encode_pair(5,1) collided, a canonical pair and its reverse would be treated
    as identical, and the exact value must equal (1<<32)|5 for cross-run
    stability.
    """
    assert encode_pair(1, 5) != encode_pair(5, 1)
    assert encode_pair(1, 5) == (1 << 32) | 5


def test_singleton_buckets_yield_no_pairs() -> None:
    """An index of only singleton buckets yields no candidate pairs.

    Matters because a single-document bucket can form no pair; emitting anything
    here would fabricate a duplicate relationship that does not exist
    (pathological all-unique shard).
    """
    idx = BucketIndex()
    idx.add(1, 0)  # singleton bucket
    idx.add(2, 5)  # singleton bucket
    pairs = list(enumerate_candidate_pairs(idx, False, 10, 0.01))
    assert pairs == []


def test_empty_index_yields_no_pairs() -> None:
    """An entirely empty index yields no candidate pairs (pathological).

    Matters because empty shards occur routinely in a sharded pipeline; the
    generator must terminate cleanly with zero output rather than raising or
    hanging.
    """
    idx = BucketIndex()
    pairs = list(enumerate_candidate_pairs(idx, False, 10, 0.01))
    assert pairs == []
