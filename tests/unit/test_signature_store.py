"""Unit tests for :class:`dedup_pipeline.minhash.signature_store.SignatureStore`.

The signature store is the load-bearing container for the entire MinHash stage:
every downstream LSH band, every candidate pair, and every final dedup decision
reads rows back out of it. If the store silently corrupts a row, returns a live
view instead of a copy, mis-tracks its append cursor, or accepts out-of-bounds
writes, the pipeline produces *wrong duplicate sets* without crashing -- the most
dangerous failure mode for a data-curation system. These tests pin the store's
write/read contract and its capacity/bounds guarantees so such silent corruption
is impossible.
"""

from __future__ import annotations

import numpy as np
import pytest

from dedup_pipeline.exceptions import ConfigError
from dedup_pipeline.minhash.signature_store import SignatureStore


def _sig(row_values: list[list[int]]) -> np.ndarray:
    """Build a ``(k, n)`` uint32 signature block from nested Python ints.

    Centralises dtype construction so every test feeds the store the exact
    ``uint32`` arrays the real MinHasher produces; a stray int64 block would mask
    dtype-handling bugs in the store.
    """
    return np.array(row_values, dtype=np.uint32)


# --------------------------------------------------------------------------- #
# memory backend: core write/read contract                                    #
# --------------------------------------------------------------------------- #


def test_append_two_blocks_then_get_returns_them() -> None:
    """Appending two ``(1, n)`` blocks makes ``get(0)``/``get(1)`` return them
    verbatim.

    This is the store's primary happy path: the MinHasher streams one batch of
    signatures at a time via ``append``, and the rest of the pipeline later reads
    each document's row back by index. If append-then-get did not round-trip
    exactly, every duplicate decision built on those rows would be computed from
    corrupted signatures.
    """
    store = SignatureStore(2, 4, backend="memory")

    block0 = _sig([[1, 2, 3, 4]])
    block1 = _sig([[5, 6, 7, 8]])
    store.append(block0)
    store.append(block1)

    assert store.get(0).tolist() == [1, 2, 3, 4]
    assert store.get(1).tolist() == [5, 6, 7, 8]


def test_len_reports_row_capacity() -> None:
    """``len(store)`` reports the fixed row *capacity*, not the fill cursor.

    Downstream stages size their LSH tables and progress bars from ``len(store)``
    (the document count). It must reflect the capacity declared at construction
    regardless of how many rows have actually been appended, or the pipeline would
    under- or over-allocate band buckets.
    """
    store = SignatureStore(7, 4, backend="memory")
    assert len(store) == 7
    # Capacity is independent of fill state.
    store.append(_sig([[1, 1, 1, 1]]))
    assert len(store) == 7


def test_num_hash_functions_property() -> None:
    """The ``num_hash_functions`` property echoes the configured signature width.

    LSH banding splits each signature into ``bands * rows == num_hash_functions``
    columns. Consumers read this property to validate band geometry; a wrong value
    would desynchronise the band slicing from the stored row width.
    """
    store = SignatureStore(3, 16, backend="memory")
    assert store.num_hash_functions == 16


def test_matrix_property_exposes_live_view() -> None:
    """The ``matrix`` property returns the underlying array (a live view).

    Bulk consumers (e.g. vectorised band hashing) operate directly on the whole
    matrix for speed and must see writes made through ``append``/``set_range``.
    The documented contract is a *view*, not a copy, so this confirms the property
    reflects mutations rather than handing back a stale snapshot.
    """
    store = SignatureStore(2, 3, backend="memory")
    store.set_range(0, _sig([[9, 9, 9]]))
    mat = store.matrix
    assert mat.shape == (2, 3)
    assert mat[0].tolist() == [9, 9, 9]
    # Mutating the view is visible through get -> it is genuinely the same buffer.
    mat[1] = np.array([4, 5, 6], dtype=np.uint32)
    assert store.get(1).tolist() == [4, 5, 6]


def test_set_range_writes_at_explicit_offset() -> None:
    """``set_range(start, block)`` writes exactly at ``start`` and nowhere else.

    When batch offsets are known ahead of time (e.g. resuming from a checkpoint),
    the pipeline places each batch at its precomputed row. An off-by-one in the
    offset would shift every document's signature, silently mislabelling which doc
    owns which row.
    """
    store = SignatureStore(5, 2, backend="memory")
    store.set_range(2, _sig([[7, 8], [9, 10]]))

    assert store.get(2).tolist() == [7, 8]
    assert store.get(3).tolist() == [9, 10]
    # Untouched rows remain zero -> set_range did not bleed outside its range.
    assert store.get(0).tolist() == [0, 0]
    assert store.get(4).tolist() == [0, 0]


def test_append_returns_post_append_cursor() -> None:
    """``append`` returns the cursor *after* the write (the running row count).

    Producers use the returned cursor to know how many rows are now committed and
    to detect completion. If it returned the pre-append position or the block size
    instead of the new total, batch bookkeeping and checkpoint offsets would drift.
    """
    store = SignatureStore(5, 2, backend="memory")
    assert store.append(_sig([[1, 1], [2, 2]])) == 2  # two rows -> cursor 2
    assert store.append(_sig([[3, 3]])) == 3  # one more -> cursor 3


def test_append_past_capacity_raises_config_error() -> None:
    """Appending beyond the declared capacity raises :class:`ConfigError`.

    Capacity is fixed at the corpus's document count. Writing past it would either
    corrupt memory (memory backend) or extend the file (mmap backend) and admit a
    "document" that does not exist. Failing loud here protects the 1:1 mapping
    between rows and real documents.
    """
    store = SignatureStore(2, 2, backend="memory")
    store.append(_sig([[1, 1]]))
    with pytest.raises(ConfigError):
        store.append(_sig([[2, 2], [3, 3]]))  # would reach cursor 3 > capacity 2


def test_set_range_wrong_column_width_raises_config_error() -> None:
    """A block whose column count differs from the store width raises
    :class:`ConfigError`.

    The signature width is fixed by ``num_hash_functions``. A mismatched-width
    block means the producer used a different hash count than the store was sized
    for; admitting it would ragged-edge the matrix and break every band slice.
    """
    store = SignatureStore(3, 4, backend="memory")
    with pytest.raises(ConfigError):
        store.set_range(0, _sig([[1, 2, 3]]))  # width 3 != store width 4


def test_set_range_out_of_bounds_raises_config_error() -> None:
    """A ``set_range`` whose row span exceeds capacity raises
    :class:`ConfigError`.

    This guards the explicit-offset write path the same way the cursor guards the
    append path: a batch placed past the end would write to non-existent
    documents. Bounds must be enforced even when the caller supplies the offset.
    """
    store = SignatureStore(3, 2, backend="memory")
    with pytest.raises(ConfigError):
        store.set_range(2, _sig([[1, 1], [2, 2]]))  # [2, 4) exceeds capacity 3


def test_get_returns_a_copy_not_a_view() -> None:
    """``get`` returns an independent copy; mutating it never touches the store.

    Downstream code routinely keeps and mutates per-document signature arrays
    (sorting, masking, band extraction). If ``get`` leaked the backing buffer,
    such mutations would silently rewrite the stored signature and poison every
    later read of that document -- a corruption that surfaces only as wrong dedup
    results far from the cause.
    """
    store = SignatureStore(2, 3, backend="memory")
    store.set_range(0, _sig([[10, 20, 30], [40, 50, 60]]))

    first = store.get(0)
    first[:] = 0  # aggressively clobber the returned array
    # The store must be unchanged despite the mutation above.
    assert store.get(0).tolist() == [10, 20, 30]
    assert store.get(1).tolist() == [40, 50, 60]


def test_get_many_returns_requested_rows_in_order() -> None:
    """``get_many`` returns exactly the requested rows, in the requested order.

    LSH yields candidate pairs as index lists; the verifier fetches their
    signatures in bulk via ``get_many`` and compares them positionally. If the
    returned order did not match the requested order, Jaccard estimates would be
    computed between the wrong documents.
    """
    store = SignatureStore(3, 2, backend="memory")
    store.set_range(0, _sig([[1, 1], [2, 2], [3, 3]]))

    out = store.get_many([2, 0])
    assert out.tolist() == [[3, 3], [1, 1]]


def test_get_out_of_bounds_index_raises_config_error() -> None:
    """``get`` with an index outside ``[0, num_docs)`` raises
    :class:`ConfigError`.

    A request for a non-existent row signals an upstream id/offset bug. Returning
    garbage (or a negative-index wraparound) would feed a bogus signature into the
    dedup comparison; raising surfaces the bug at its source instead.
    """
    store = SignatureStore(2, 2, backend="memory")
    with pytest.raises(ConfigError):
        store.get(2)  # valid indices are 0 and 1
    with pytest.raises(ConfigError):
        store.get(-1)  # negative indices must not wrap around


# --------------------------------------------------------------------------- #
# mmap backend                                                                #
# --------------------------------------------------------------------------- #


def test_mmap_backend_round_trips_through_disk(tmp_path) -> None:
    """The mmap backend round-trips appended rows and materialises a real file.

    For corpora too large for RAM the pipeline uses the disk-backed backend. It
    must honour the identical write/read contract as the memory backend, persist
    on ``flush``, and produce an actual on-disk file (so a later run / process can
    re-open it). A backend that diverged here would make large-corpus runs return
    different results than small in-RAM ones.
    """
    path = tmp_path / "sig.dat"
    store = SignatureStore(3, 2, backend="mmap", path=path)

    store.append(_sig([[11, 12]]))
    store.append(_sig([[13, 14], [15, 16]]))
    store.flush()

    assert path.exists()
    assert store.get(0).tolist() == [11, 12]
    assert store.get(1).tolist() == [13, 14]
    assert store.get(2).tolist() == [15, 16]


def test_mmap_backend_without_path_raises_config_error() -> None:
    """Requesting the mmap backend without a ``path`` raises
    :class:`ConfigError`.

    The disk backend cannot exist without a backing file. Defaulting to a temp or
    silently falling back to memory would either lose data on exit or blow past RAM
    on the very corpus the mmap backend was chosen to handle. The misconfiguration
    must fail at construction.
    """
    with pytest.raises(ConfigError):
        SignatureStore(3, 2, backend="mmap", path=None)


# --------------------------------------------------------------------------- #
# configuration / pathological edge cases                                     #
# --------------------------------------------------------------------------- #


def test_unknown_backend_name_raises_config_error() -> None:
    """An unrecognised backend name raises :class:`ConfigError`.

    Backend selection often comes from a config file or CLI flag. A typo'd backend
    must abort loudly at construction rather than degrade into an undefined storage
    mode, so misconfiguration can never silently change dedup behaviour.
    """
    with pytest.raises(ConfigError):
        SignatureStore(3, 2, backend="redis")


def test_zero_capacity_store_has_len_zero(tmp_path) -> None:
    """PATHOLOGICAL: a ``num_docs=0`` store has length 0 and accepts no rows.

    An empty corpus (everything filtered upstream, or an empty shard) must not
    crash the store at construction. ``len`` must be 0 and any append must overflow
    immediately -- the degenerate input the pipeline will hit at the tail of a
    sharded run and must survive without a special-case branch upstream.
    """
    store = SignatureStore(0, 4, backend="memory")
    assert len(store) == 0
    with pytest.raises(ConfigError):
        store.append(_sig([[1, 2, 3, 4]]))  # any row overflows capacity 0
    # get on the empty store is also out of bounds.
    with pytest.raises(ConfigError):
        store.get(0)


def test_single_row_store_append_and_get(tmp_path) -> None:
    """PATHOLOGICAL: a single-row store fills to exactly capacity, then overflows.

    The 1-document corpus is the smallest non-empty boundary case: the first
    append must succeed and return cursor 1 (store full), and the *next* append --
    even of a single row -- must overflow. This pins the off-by-one boundary
    between "exactly full" and "one too many", the spot where capacity checks most
    often break.
    """
    store = SignatureStore(1, 3, backend="memory")
    assert len(store) == 1
    assert store.append(_sig([[7, 8, 9]])) == 1  # cursor now at capacity
    assert store.get(0).tolist() == [7, 8, 9]
    with pytest.raises(ConfigError):
        store.append(_sig([[1, 2, 3]]))  # store is already full
