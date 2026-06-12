"""Candidate-pair enumeration from the inverted bucket index.

Stage 6 turns buckets into a stream of unique candidate pairs ``(i, j)`` with
``i < j``. The same pair can appear in several bands' buckets, so a Bloom filter
(or an exact set for small runs) suppresses duplicates. The result is a
generator, so pairs are never all materialised in memory at once.

Responsibility:
    * Yield deduplicated, canonically-ordered candidate pairs.

Inputs:
    * A :class:`~dedup_pipeline.lsh.bucket_index.BucketIndex`.

Outputs:
    * An iterator of ``(int, int)`` pairs with ``i < j``.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

from dedup_pipeline.lsh.bucket_index import BloomFilter, BucketIndex

# Bit shift used to pack an ordered pair (i, j) into a single integer key. Valid
# for document indices below 2**32 (4.29e9), far above any realistic corpus.
_PAIR_SHIFT: int = 32


def encode_pair(i: int, j: int) -> int:
    """Pack an ordered index pair into a single integer key.

    Args:
        i: The smaller index.
        j: The larger index.

    Returns:
        ``(i << 32) | j`` — a unique key for ``i, j < 2**32``.

    Example:
        >>> encode_pair(1, 5)
        4294967301
    """
    return (i << _PAIR_SHIFT) | j


def enumerate_candidate_pairs(
    index: BucketIndex,
    use_bloom_filter: bool,
    bloom_expected_pairs: int,
    bloom_false_positive_rate: float,
) -> Iterator[tuple[int, int]]:
    """Yield unique candidate pairs ``(i, j)`` with ``i < j``.

    For each candidate bucket, all ``C(k, 2)`` index pairs are emitted in
    canonical order, deduplicated across buckets.

    Args:
        index: The inverted bucket index (only multi-doc buckets matter).
        use_bloom_filter: If ``True``, use a :class:`BloomFilter` for dedup
            (constant memory, may drop a pair at the configured FP rate); if
            ``False``, use an exact ``set`` (exact, memory grows with pair count).
        bloom_expected_pairs: Bloom sizing hint (ignored when exact).
        bloom_false_positive_rate: Bloom target FP rate (ignored when exact).

    Yields:
        Candidate pairs ``(i, j)`` with ``i < j``, each at most once.

    Example:
        >>> idx = BucketIndex()
        >>> idx.add_bucket(1, [2, 0, 1])
        >>> idx.add_bucket(2, [0, 1])  # (0, 1) also here -> must not repeat
        >>> sorted(enumerate_candidate_pairs(idx, False, 10, 0.01))
        [(0, 1), (0, 2), (1, 2)]
    """
    seen_exact: set[int] = set()
    seen_bloom: BloomFilter | None = (
        BloomFilter(bloom_expected_pairs, bloom_false_positive_rate)
        if use_bloom_filter
        else None
    )

    for docs in index.candidate_buckets():
        # Sort so combinations come out as (i, j) with i < j.
        ordered = sorted(docs)
        for i, j in itertools.combinations(ordered, 2):
            key = encode_pair(i, j)
            if seen_bloom is not None:
                if not seen_bloom.add_if_absent(key):
                    continue  # already (probably) emitted
            else:
                if key in seen_exact:
                    continue
                seen_exact.add(key)
            yield (i, j)
