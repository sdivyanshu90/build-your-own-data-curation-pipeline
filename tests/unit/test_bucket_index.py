"""Unit tests for bucket-key hashing, the inverted index, and the Bloom filter.

Stage 5 turns a banded signature matrix into an inverted index
``bucket_key -> [doc_idx, ...]``; documents sharing a bucket in any band are
candidate near-duplicates. These tests pin the band-hashing contract (shape,
dtype, per-band collision behaviour), the index's merge/candidate semantics, its
picklability (required for multiprocessing fan-out), and the Bloom filter's
no-false-negative guarantee that bounds dedup recall loss.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from dedup_pipeline.exceptions import ConfigError
from dedup_pipeline.lsh.banding import BandingScheme
from dedup_pipeline.lsh.bucket_index import (
    BloomFilter,
    BucketIndex,
    build_bucket_index,
    compute_bucket_keys,
)


def test_compute_bucket_keys_shape_and_dtype() -> None:
    """compute_bucket_keys returns a (n_docs, num_bands) uint64 array.

    Matters because downstream grouping reshapes/sorts these keys per band; a
    wrong shape or a narrower dtype would silently truncate keys and fabricate
    or lose bucket collisions, corrupting candidate generation.
    """
    sig = np.array([[1, 2, 3, 4], [1, 2, 9, 9]], dtype=np.uint32)
    keys = compute_bucket_keys(sig, num_bands=2, num_rows=2)
    assert keys.shape == (2, 2)
    assert keys.dtype == np.uint64


def test_compute_bucket_keys_per_band_collisions() -> None:
    """Identical band columns collide; differing band columns do not.

    Matters because LSH correctness rests on this exact rule: two docs must
    share a bucket key iff they agree on every row of that band. Band 0 (cols
    0,1) is identical across both docs so keys must match; band 1 (cols 2,3)
    differs so keys must not match.
    """
    sig = np.array([[1, 2, 3, 4], [1, 2, 9, 9]], dtype=np.uint32)
    keys = compute_bucket_keys(sig, num_bands=2, num_rows=2)
    assert keys[0, 0] == keys[1, 0]
    assert keys[0, 1] != keys[1, 1]


def test_compute_bucket_keys_width_mismatch_raises() -> None:
    """compute_bucket_keys raises ConfigError when width != bands*rows.

    Matters because a mismatched decomposition would reshape garbage across band
    boundaries and produce meaningless keys; failing loudly prevents silent
    corruption of the whole dedup run.
    """
    sig = np.array([[1, 2, 3, 4], [1, 2, 9, 9]], dtype=np.uint32)
    with pytest.raises(ConfigError):
        compute_bucket_keys(sig, num_bands=3, num_rows=2)  # 3*2=6 != width 4


def test_bucket_index_add_bucket_and_as_dict() -> None:
    """add_bucket stores the doc list and as_dict() reflects it exactly.

    Matters because as_dict() is the persisted/inspected form of the index; if a
    bucket's membership were not recorded faithfully, candidate pairs derived
    from it would be wrong.
    """
    idx = BucketIndex()
    idx.add_bucket(100, [0, 3, 7])
    assert idx.as_dict() == {100: [0, 3, 7]}


def test_bucket_index_candidate_buckets_skips_singletons() -> None:
    """candidate_buckets() yields only buckets with at least two documents.

    Matters because a singleton bucket can form no pair; emitting it would waste
    work and, worse, signal a non-existent candidate group downstream.
    """
    idx = BucketIndex()
    idx.add_bucket(1, [0, 1, 2])
    idx.add_bucket(2, [5])  # singleton: must be skipped
    buckets = list(idx.candidate_buckets())
    assert buckets == [[0, 1, 2]]
    assert all(len(docs) >= 2 for docs in buckets)


def test_bucket_index_add_appends_single_docs() -> None:
    """add() appends individual doc indices, accumulating a bucket.

    Matters because per-band builders may stream docs one at a time; appends
    must accumulate (not overwrite) so the full bucket membership is preserved.
    """
    idx = BucketIndex()
    idx.add(42, 0)
    idx.add(42, 1)
    idx.add(42, 2)
    assert idx.as_dict() == {42: [0, 1, 2]}


def test_bucket_index_add_bucket_merges_existing_key() -> None:
    """Adding to an existing key extends (merges) rather than overwrites it.

    Matters because a (rare) cross-band hash collision must not drop documents:
    merging can only over-generate candidate pairs (verification filters those),
    whereas overwriting would silently lose real candidates.
    """
    idx = BucketIndex()
    idx.add_bucket(7, [0, 1])
    idx.add_bucket(7, [2, 3])
    assert idx.as_dict() == {7: [0, 1, 2, 3]}


def test_bucket_index_is_picklable_and_preserves_state() -> None:
    """A BucketIndex round-trips through pickle, preserving as_dict().

    Matters because the index is shipped across process boundaries during
    parallel band building; __getstate__ must drop the unpicklable threading
    lock while keeping every bucket intact, or distributed dedup would fail.
    """
    idx = BucketIndex()
    idx.add_bucket(1, [0, 1])
    idx.add_bucket(2, [2, 3, 4])
    restored = pickle.loads(pickle.dumps(idx))
    assert restored.as_dict() == idx.as_dict()
    # The lock is recreated on unpickle, so the restored index stays usable.
    restored.add(2, 5)
    assert restored.as_dict()[2] == [2, 3, 4, 5]


def test_build_bucket_index_groups_identical_signatures() -> None:
    """build_bucket_index puts two identical signatures in a shared bucket.

    Matters because this is the end-to-end purpose of Stage 5: documents with
    matching bands must co-occur in a candidate bucket while a distinct document
    must not be grouped with them. Docs 0 and 1 are identical; doc 2 differs.
    """
    sig = np.array(
        [[1, 2, 3, 4], [1, 2, 3, 4], [9, 9, 9, 9]], dtype=np.uint32
    )
    # b*r == 4 == signature width.
    idx = build_bucket_index(sig, BandingScheme(4, 2, 2))
    buckets = list(idx.candidate_buckets())
    # The identical pair {0, 1} must appear together in some candidate bucket.
    assert any({0, 1} <= set(docs) for docs in buckets)
    # Doc 2 (distinct in every band) must not be grouped with 0 or 1.
    assert all(2 not in docs for docs in buckets)


def test_bloom_filter_add_if_absent_first_true_then_false() -> None:
    """add_if_absent returns True the first time and False for a repeat.

    Matters because the pair generator uses this exact signal to suppress
    re-emitting a pair seen in another band; if a repeat returned True the same
    pair would be yielded twice.
    """
    bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
    assert bf.add_if_absent(42) is True
    assert bf.add_if_absent(42) is False


def test_bloom_filter_no_false_negative_and_membership() -> None:
    """An added key is always reported present (no false negatives).

    Matters because the Bloom filter's whole correctness guarantee is one-sided:
    it may drop a pair (false positive) but must never claim an inserted key is
    absent, which would let a duplicate slip through as new.
    """
    bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
    bf.add_if_absent(123456789)
    assert 123456789 in bf
    # A never-added key probes correctly via __contains__ as well.
    assert bf.add_if_absent(987654321) is True
    assert 987654321 in bf


def test_bloom_filter_sizing_is_positive() -> None:
    """num_bits and num_hashes are positive for a valid configuration.

    Matters because a zero-bit array or zero hash probes would make every query
    degenerate (always-absent or always-present), breaking dedup entirely.
    """
    bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
    assert bf.num_bits > 0
    assert bf.num_hashes > 0


def test_bloom_filter_invalid_expected_items_raises() -> None:
    """BloomFilter rejects expected_items < 1 with ConfigError.

    Matters because sizing math divides by expected_items; a non-positive count
    is nonsensical and must be rejected loudly rather than producing a degenerate
    filter.
    """
    with pytest.raises(ConfigError):
        BloomFilter(expected_items=0, false_positive_rate=0.01)


def test_bloom_filter_invalid_fp_rate_raises() -> None:
    """BloomFilter rejects a false-positive rate outside (0, 1).

    Matters because the bit-array size formula takes log(fp_rate); a rate of 0
    or 1 (or out of range) yields undefined/degenerate sizing and must be caught.
    """
    with pytest.raises(ConfigError):
        BloomFilter(expected_items=1000, false_positive_rate=0.0)
    with pytest.raises(ConfigError):
        BloomFilter(expected_items=1000, false_positive_rate=1.0)


def test_build_bucket_index_empty_matrix_is_empty(
) -> None:
    """A 0-document signature matrix builds an empty index (pathological).

    Matters because empty shards are routine in a sharded pipeline; building must
    short-circuit to an empty index instead of crashing on the reshape/sort path
    or yielding spurious buckets.
    """
    sig = np.empty((0, 4), dtype=np.uint32)
    idx = build_bucket_index(sig, BandingScheme(4, 2, 2))
    assert len(idx) == 0
    assert list(idx.candidate_buckets()) == []


def test_build_bucket_index_all_identical_share_buckets() -> None:
    """When every document is identical they all land in shared buckets.

    Matters because a fully-duplicated shard is a stress case: all docs must
    co-occur so every true pair is generated, with no doc orphaned into a
    singleton.
    """
    sig = np.array(
        [[5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]],
        dtype=np.uint32,
    )
    idx = build_bucket_index(sig, BandingScheme(4, 2, 2))
    buckets = list(idx.candidate_buckets())
    assert buckets  # at least one candidate bucket exists
    # Every document index appears in some candidate bucket (none orphaned).
    covered = {doc for docs in buckets for doc in docs}
    assert covered == {0, 1, 2, 3}
