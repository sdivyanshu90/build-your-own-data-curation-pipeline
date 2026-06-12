"""Concurrency tests for the mutable-shared-state classes.

Hard constraint: every class that maintains mutable shared state
(:class:`BucketIndex`, :class:`UnionFind`, :class:`SignatureStore`) documents a
thread-safety guarantee and must have an accompanying concurrency test. These
stress those guarantees from many threads and assert the final state is correct.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from dedup_pipeline.clustering.union_find import UnionFind
from dedup_pipeline.lsh.bucket_index import BucketIndex
from dedup_pipeline.minhash.signature_store import SignatureStore

_NUM_THREADS = 8
_PER_THREAD = 200


@pytest.mark.integration
def test_bucket_index_concurrent_add() -> None:
    """Concurrent add_bucket calls all land without loss or corruption.

    Matters because the bucket index is documented as thread-safe so multiple
    band workers can populate it in parallel; a race would drop candidate pairs.
    """
    index = BucketIndex()

    def worker(thread_id: int) -> None:
        for k in range(_PER_THREAD):
            key = thread_id * _PER_THREAD + k  # globally unique key per add
            index.add_bucket(key, [thread_id, k])

    with ThreadPoolExecutor(max_workers=_NUM_THREADS) as pool:
        list(pool.map(worker, range(_NUM_THREADS)))

    buckets = index.as_dict()
    assert len(buckets) == _NUM_THREADS * _PER_THREAD
    # Spot-check that every list survived intact.
    assert all(len(v) == 2 for v in buckets.values())


@pytest.mark.integration
def test_union_find_concurrent_union() -> None:
    """Concurrent unions produce a single correct component for a chain.

    Matters because clustering may union from multiple workers; a race could
    split or mislabel a duplicate cluster.
    """
    n = _NUM_THREADS * _PER_THREAD
    uf = UnionFind(n)

    def worker(thread_id: int) -> None:
        # Each thread links a contiguous segment; segments overlap at endpoints
        # so the whole range collapses into one component.
        start = thread_id * _PER_THREAD
        end = min(start + _PER_THREAD, n - 1)
        for i in range(start, end):
            uf.union(i, i + 1)

    with ThreadPoolExecutor(max_workers=_NUM_THREADS) as pool:
        list(pool.map(worker, range(_NUM_THREADS)))

    assert uf.connected(0, n - 1)
    assert uf.num_components() == 1


@pytest.mark.integration
def test_signature_store_concurrent_append() -> None:
    """Concurrent appends preserve every row with no overwrites.

    Matters because the store is documented as supporting thread-safe append; a
    race on the cursor would clobber signatures and corrupt the matrix.
    """
    width = 4
    capacity = _NUM_THREADS * _PER_THREAD
    store = SignatureStore(capacity, width, backend="memory")

    def worker(thread_id: int) -> None:
        # Each row is filled with the thread id so we can verify counts later.
        block = np.full((_PER_THREAD, width), thread_id, dtype=np.uint32)
        store.append(block)

    with ThreadPoolExecutor(max_workers=_NUM_THREADS) as pool:
        list(pool.map(worker, range(_NUM_THREADS)))

    matrix = store.matrix
    # Every thread's id must appear exactly _PER_THREAD times across the rows.
    row_values = matrix[:, 0]
    for thread_id in range(_NUM_THREADS):
        assert int(np.count_nonzero(row_values == thread_id)) == _PER_THREAD
