"""Storage for the MinHash signature matrix.

The signature matrix has shape ``(n_docs, n_hashes)`` and dtype ``uint32``. For
a 100M-document corpus at 128 hashes that is ``100e6 * 128 * 4 bytes ≈ 51 GB``,
which does not fit in RAM on a typical machine. This module therefore offers two
backends behind one interface:

    * ``"memory"`` — a plain in-RAM ``numpy.ndarray`` (fast, bounded by RAM).
    * ``"mmap"`` — a disk-backed ``numpy.memmap`` (RAM-independent, OS-paged).

Responsibility:
    * Allocate, fill (row-range or thread-safe append), and read back signatures.

Inputs:
    * Signature rows produced by
      :class:`~dedup_pipeline.minhash.minhash.MinHasher`.

Outputs:
    * Signature rows / sub-matrices as ``uint32`` arrays.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from dedup_pipeline.exceptions import ConfigError, IOError  # noqa: A004 - domain name

# The signature matrix dtype. uint32 is sufficient because every hash output is
# < 2**32 (see dedup_pipeline.minhash.hash_functions.HASH_MODULUS) and halves
# the footprint relative to uint64.
SIGNATURE_DTYPE: type[np.unsignedinteger[Any]] = np.uint32


class SignatureStore:
    """A fixed-capacity ``(n_docs, n_hashes)`` signature matrix.

    Capacity is set at construction. Fill it either by writing explicit row
    ranges with :meth:`set_range` (when you know each batch's offset) or by
    :meth:`append`-ing batches in arrival order (a lock-guarded cursor tracks
    the next free row).

    Thread-safety:
        * :meth:`append` is **thread-safe**: an internal :class:`threading.Lock`
          serialises cursor advancement, so concurrent producers never overwrite
          each other's rows. The actual array write happens after the cursor is
          reserved, on a disjoint row range, so writes proceed in parallel.
        * :meth:`set_range` is thread-safe **only for disjoint ranges**; the
          caller owns range allocation. Overlapping concurrent ranges are a
          caller bug and are not guarded.
        * Reads (:meth:`get`, :meth:`get_many`, :attr:`matrix`) are safe once the
          relevant rows have been written and the writing thread has joined.

    Args:
        num_docs: Row capacity (number of documents).
        num_hash_functions: Column count (signature length).
        backend: ``"memory"`` or ``"mmap"``.
        path: Required when ``backend="mmap"``; the backing file path.

    Raises:
        ConfigError: If ``backend`` is unknown, or ``mmap`` is requested without
            a ``path``.
        IOError: If the memory-mapped file cannot be created.

    Example:
        >>> store = SignatureStore(3, 4, backend="memory")
        >>> import numpy as np
        >>> store.append(np.array([[1, 2, 3, 4]], dtype=np.uint32))
        1
        >>> store.get(0).tolist()
        [1, 2, 3, 4]
    """

    def __init__(
        self,
        num_docs: int,
        num_hash_functions: int,
        backend: str = "memory",
        path: Path | None = None,
    ) -> None:
        self._num_docs = num_docs
        self._num_hashes = num_hash_functions
        self._backend = backend
        self._cursor = 0
        self._lock = threading.Lock()
        shape = (num_docs, num_hash_functions)

        if backend == "memory":
            self._matrix: npt.NDArray[Any] = np.zeros(shape, dtype=SIGNATURE_DTYPE)
        elif backend == "mmap":
            if path is None:
                raise ConfigError("backend='mmap' requires a 'path' argument")
            try:
                self._matrix = np.memmap(
                    path, dtype=SIGNATURE_DTYPE, mode="w+", shape=shape
                )
            except OSError as exc:  # disk full, bad path, permissions
                raise IOError(
                    f"could not create mmap signature file at {path}: {exc}"
                ) from exc
        else:
            raise ConfigError(
                f"unknown signature store backend {backend!r}; "
                "expected 'memory' or 'mmap'"
            )

    def __len__(self) -> int:
        """Return the row capacity (number of documents)."""
        return self._num_docs

    @property
    def num_hash_functions(self) -> int:
        """Return the signature length (column count)."""
        return self._num_hashes

    @property
    def matrix(self) -> npt.NDArray[Any]:
        """The underlying ``(n_docs, n_hashes)`` array (live view, not a copy)."""
        return self._matrix

    def set_range(self, start: int, signatures: npt.NDArray[Any]) -> None:
        """Write a block of signatures at an explicit starting row.

        Args:
            start: The first row index to write.
            signatures: A ``(k, n_hashes)`` ``uint32`` array.

        Raises:
            ConfigError: If the column count mismatches or the block would
                overflow capacity.

        Example:
            >>> import numpy as np
            >>> s = SignatureStore(4, 2)
            >>> s.set_range(2, np.array([[7, 8]], dtype=np.uint32))
            >>> s.get(2).tolist()
            [7, 8]
        """
        rows = signatures.shape[0]
        if signatures.shape[1] != self._num_hashes:
            raise ConfigError(
                f"signature width {signatures.shape[1]} != store width "
                f"{self._num_hashes}"
            )
        if start < 0 or start + rows > self._num_docs:
            raise ConfigError(
                f"row range [{start}, {start + rows}) out of bounds "
                f"[0, {self._num_docs})"
            )
        self._matrix[start : start + rows] = signatures

    def append(self, signatures: npt.NDArray[Any]) -> int:
        """Append a block of signatures at the current cursor (thread-safe).

        Args:
            signatures: A ``(k, n_hashes)`` ``uint32`` array.

        Returns:
            The cursor position **after** the append (i.e. the new row count).

        Raises:
            ConfigError: If the append would overflow capacity or the width
                mismatches.

        Example:
            >>> import numpy as np
            >>> s = SignatureStore(2, 2)
            >>> s.append(np.array([[1, 1]], dtype=np.uint32))
            1
            >>> s.append(np.array([[2, 2]], dtype=np.uint32))
            2
        """
        rows = signatures.shape[0]
        with self._lock:
            start = self._cursor
            if start + rows > self._num_docs:
                raise ConfigError(
                    f"append of {rows} rows overflows capacity {self._num_docs} "
                    f"(cursor at {start})"
                )
            self._cursor += rows  # reserve the range before releasing the lock
        # Disjoint range write outside the lock -> concurrent appends parallelise.
        self.set_range(start, signatures)
        return start + rows

    def get(self, index: int) -> npt.NDArray[Any]:
        """Return a single signature row.

        Args:
            index: The document row index.

        Returns:
            A ``(n_hashes,)`` ``uint32`` array (a copy, safe to keep).

        Raises:
            ConfigError: If ``index`` is out of bounds.
        """
        if not 0 <= index < self._num_docs:
            raise ConfigError(
                f"row index {index} out of bounds [0, {self._num_docs})"
            )
        return np.asarray(self._matrix[index]).copy()

    def get_many(self, indices: list[int]) -> npt.NDArray[Any]:
        """Return multiple signature rows.

        Args:
            indices: Document row indices.

        Returns:
            A ``(len(indices), n_hashes)`` ``uint32`` array.

        Example:
            >>> import numpy as np
            >>> s = SignatureStore(3, 2)
            >>> s.set_range(0, np.array([[1, 1], [2, 2], [3, 3]], dtype=np.uint32))
            >>> s.get_many([0, 2]).tolist()
            [[1, 1], [3, 3]]
        """
        return np.asarray(self._matrix[np.asarray(indices, dtype=np.int64)]).copy()

    def flush(self) -> None:
        """Flush a memory-mapped backend to disk; a no-op for the memory backend.

        Raises:
            IOError: If the underlying flush fails.
        """
        if self._backend == "mmap":
            try:
                # np.memmap exposes flush(); guarded for the rare flush failure.
                self._matrix.flush()  # type: ignore[attr-defined]
            except (OSError, ValueError) as exc:
                raise IOError(f"failed to flush mmap signature store: {exc}") from exc
