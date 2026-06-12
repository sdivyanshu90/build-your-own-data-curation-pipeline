"""Inverted bucket index and a Bloom filter for candidate-pair generation.

Stage 5 of the pipeline turns a banded signature matrix into an inverted index
mapping ``bucket_key -> [doc_idx, ...]``. Documents that land in the same bucket
(for some band) are candidate near-duplicates. Only buckets with at least two
documents are retained, since a singleton can form no pair.

The :class:`BloomFilter` here is used by the candidate-pair generator to suppress
re-emitting a pair that appears in several bands, bounding memory for very large
candidate sets [Bloom 1970].

Responsibility:
    * Compute per-band bucket keys, build/hold the inverted index, and provide a
      space-efficient set-membership filter.

Inputs:
    * A ``(n_docs, n_hashes)`` ``uint32`` signature matrix and a
      :class:`~dedup_pipeline.lsh.banding.BandingScheme`.

Outputs:
    * A :class:`BucketIndex` whose candidate buckets drive pair enumeration.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterator
from typing import Any

import mmh3
import numpy as np
import numpy.typing as npt

from dedup_pipeline.exceptions import ConfigError
from dedup_pipeline.lsh.banding import BandingScheme

# 64-bit FNV-1a constants (a fixed, well-known hashing recurrence, not a tunable
# knob). uint64 arithmetic wraps mod 2**64, which is exactly what FNV requires.
_FNV_OFFSET: np.uint64 = np.uint64(14695981039346656037)
_FNV_PRIME: np.uint64 = np.uint64(1099511628211)

# Bytes used to encode an integer pair key for the Bloom filter. 16 bytes (128
# bits) comfortably holds any pair key for corpora up to 2**64 documents.
_PAIR_KEY_BYTES: int = 16


def compute_bucket_keys(
    signatures: npt.NDArray[Any], num_bands: int, num_rows: int
) -> npt.NDArray[Any]:
    """Hash each band of every signature to a 64-bit bucket key.

    Uses a vectorized FNV-1a recurrence over the ``r`` columns of each band. The
    band index is mixed in first, so identical row-tuples in different bands map
    to different keys (preventing spurious cross-band collisions).

    Args:
        signatures: ``(n_docs, n_hashes)`` ``uint32`` matrix.
        num_bands: Number of bands ``b``.
        num_rows: Rows per band ``r`` (``b * r`` must equal ``n_hashes``).

    Returns:
        A ``(n_docs, num_bands)`` ``uint64`` array of bucket keys.

    Raises:
        ConfigError: If the matrix width does not equal ``num_bands * num_rows``.

    Example:
        >>> import numpy as np
        >>> sig = np.array([[1, 2, 3, 4], [1, 2, 9, 9]], dtype=np.uint32)
        >>> keys = compute_bucket_keys(sig, num_bands=2, num_rows=2)
        >>> bool(keys[0, 0] == keys[1, 0])  # band 0 identical -> same bucket
        True
        >>> bool(keys[0, 1] == keys[1, 1])  # band 1 differs -> different bucket
        False
    """
    n_docs, width = signatures.shape
    if width != num_bands * num_rows:
        raise ConfigError(
            f"signature width {width} != num_bands ({num_bands}) * "
            f"num_rows ({num_rows})"
        )
    reshaped = signatures.reshape(n_docs, num_bands, num_rows).astype(np.uint64)
    keys = np.empty((n_docs, num_bands), dtype=np.uint64)
    mask64 = (1 << 64) - 1
    # FNV deliberately relies on wraparound mod 2**64; silence the (expected)
    # uint64 overflow warning rather than let it surface or fail -W error runs.
    with np.errstate(over="ignore"):
        for band in range(num_bands):
            # Seed with the band index (computed in Python to avoid a scalar
            # overflow warning) so identical row-values in different bands differ.
            seed = ((int(_FNV_OFFSET) ^ (band + 1)) * int(_FNV_PRIME)) & mask64
            column = np.full(n_docs, np.uint64(seed), dtype=np.uint64)
            for row in range(num_rows):
                column = (column ^ reshaped[:, band, row]) * _FNV_PRIME
            keys[:, band] = column
    return keys


class BloomFilter:
    """A space-efficient probabilistic set for integer keys [Bloom 1970].

    Membership queries never yield false negatives but may yield false positives
    at the configured rate. In candidate-pair de-duplication a false positive
    merely *drops* a pair (a bounded recall cost), never creates a false merge.

    Thread-safety:
        **Not** thread-safe. It is used by a single-threaded pair generator. To
        share one across threads, guard every call externally.

    Args:
        expected_items: Anticipated number of distinct keys, used to size the
            bit array.
        false_positive_rate: Target false-positive probability in ``(0, 1)``.

    Raises:
        ConfigError: If arguments are out of range.

    Example:
        >>> bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
        >>> bf.add_if_absent(42)
        True
        >>> bf.add_if_absent(42)
        False
    """

    def __init__(self, expected_items: int, false_positive_rate: float) -> None:
        if expected_items < 1:
            raise ConfigError(f"expected_items must be >= 1, got {expected_items}")
        if not 0.0 < false_positive_rate < 1.0:
            raise ConfigError(
                f"false_positive_rate must be in (0, 1), got {false_positive_rate}"
            )
        # m = -n ln p / (ln 2)^2 bits; k = (m/n) ln 2 hash functions (optimal).
        ln2 = math.log(2.0)
        raw_bits = math.ceil(
            -expected_items * math.log(false_positive_rate) / (ln2 * ln2)
        )
        self._num_bits = max(8, ((raw_bits + 7) // 8) * 8)  # round up to a byte
        self._num_hashes = max(
            1, round((self._num_bits / expected_items) * ln2)
        )
        self._bits = bytearray(self._num_bits // 8)

    @property
    def num_bits(self) -> int:
        """The size of the bit array in bits."""
        return self._num_bits

    @property
    def num_hashes(self) -> int:
        """The number of hash probes per key."""
        return self._num_hashes

    def _indices(self, item: int) -> list[int]:
        """Compute the bit indices for a key via double hashing.

        Args:
            item: The integer key.

        Returns:
            ``num_hashes`` bit positions in ``[0, num_bits)``.
        """
        payload = item.to_bytes(_PAIR_KEY_BYTES, "little", signed=False)
        h1 = mmh3.hash(payload, 0, signed=False)
        h2 = mmh3.hash(payload, 1, signed=False)
        # Kirsch-Mitzenmacher double hashing: g_i = h1 + i*h2.
        return [(h1 + i * h2) % self._num_bits for i in range(self._num_hashes)]

    def add_if_absent(self, item: int) -> bool:
        """Add a key and report whether it was (probably) new.

        Args:
            item: The integer key.

        Returns:
            ``True`` if at least one bit was previously unset (definitely a new
            key); ``False`` if all bits were already set (probably seen before,
            subject to the false-positive rate).
        """
        was_new = False
        for idx in self._indices(item):
            byte, bit = divmod(idx, 8)
            mask = 1 << bit
            if not self._bits[byte] & mask:
                was_new = True
                self._bits[byte] |= mask
        return was_new

    def __contains__(self, item: int) -> bool:
        """Return whether a key is probably present.

        Args:
            item: The integer key.

        Returns:
            ``True`` if all probed bits are set (probably present); ``False`` if
            any is unset (definitely absent).
        """
        return all(
            self._bits[byte] & (1 << bit)
            for byte, bit in (divmod(idx, 8) for idx in self._indices(item))
        )


class BucketIndex:
    """Inverted index ``bucket_key -> [doc_idx, ...]`` for candidate generation.

    Only buckets containing two or more documents are stored, because a
    singleton bucket can produce no candidate pair.

    Thread-safety:
        **Thread-safe.** All mutations go through :meth:`add` / :meth:`add_bucket`
        which hold an internal :class:`threading.Lock`, so concurrent builders
        (e.g. one thread per band) can populate one index safely. Read methods
        should be called after all writers have joined.

    Example:
        >>> idx = BucketIndex()
        >>> idx.add_bucket(100, [0, 3, 7])
        >>> idx.as_dict()
        {100: [0, 3, 7]}
        >>> list(idx.candidate_buckets())
        [[0, 3, 7]]
    """

    def __init__(self) -> None:
        self._buckets: dict[int, list[int]] = {}
        self._lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state, dropping the unpicklable lock.

        Returns:
            The bucket mapping (the lock is recreated on unpickle).
        """
        return {"_buckets": self._buckets}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state and recreate a fresh lock after unpickling.

        Args:
            state: The mapping produced by :meth:`__getstate__`.
        """
        self._buckets = state["_buckets"]
        self._lock = threading.Lock()

    def add(self, bucket_key: int, doc_index: int) -> None:
        """Add a single document to a bucket.

        Args:
            bucket_key: The 64-bit bucket key.
            doc_index: The document index to append.
        """
        with self._lock:
            self._buckets.setdefault(bucket_key, []).append(doc_index)

    def add_bucket(self, bucket_key: int, doc_indices: list[int]) -> None:
        """Add (or extend) a whole bucket.

        If the key already exists (an astronomically unlikely cross-band hash
        collision), the document lists are merged rather than overwritten, which
        is safe — it can only create extra candidate pairs, which verification
        filters out.

        Args:
            bucket_key: The 64-bit bucket key.
            doc_indices: Document indices sharing this bucket.
        """
        with self._lock:
            existing = self._buckets.get(bucket_key)
            if existing is None:
                self._buckets[bucket_key] = list(doc_indices)
            else:
                existing.extend(doc_indices)

    def candidate_buckets(self) -> Iterator[list[int]]:
        """Iterate buckets that contain at least two documents.

        Yields:
            Document-index lists of length >= 2.
        """
        for docs in self._buckets.values():
            if len(docs) >= 2:
                yield docs

    def as_dict(self) -> dict[int, list[int]]:
        """Return the underlying ``bucket_key -> doc_indices`` mapping (live)."""
        return self._buckets

    def __len__(self) -> int:
        """Return the number of stored buckets."""
        return len(self._buckets)


def build_bucket_index(
    signatures: npt.NDArray[Any], banding: BandingScheme
) -> BucketIndex:
    """Build the inverted bucket index from a signature matrix.

    For each band the document rows are grouped by bucket key using a sort +
    boundary scan (``O(n log n)`` per band rather than a Python dict loop), and
    only multi-document groups are stored.

    Args:
        signatures: ``(n_docs, n_hashes)`` ``uint32`` matrix.
        banding: The band/row decomposition.

    Returns:
        A :class:`BucketIndex` containing only candidate buckets.

    Example:
        >>> import numpy as np
        >>> from dedup_pipeline.lsh.banding import BandingScheme
        >>> sig = np.array([[1, 1], [1, 1], [9, 9]], dtype=np.uint32)
        >>> idx = build_bucket_index(sig, BandingScheme(2, 1, 2))
        >>> list(idx.candidate_buckets())
        [[0, 1]]
    """
    n_docs = signatures.shape[0]
    keys = compute_bucket_keys(signatures, banding.num_bands, banding.num_rows)
    index = BucketIndex()
    if n_docs == 0:
        return index
    for band in range(banding.num_bands):
        column = keys[:, band]
        order = np.argsort(column, kind="stable")
        sorted_keys = column[order]
        # Group boundaries: positions where the sorted key changes value.
        change = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
        zero = np.zeros(1, dtype=np.int64)
        last = np.full(1, n_docs, dtype=np.int64)
        starts: npt.NDArray[Any] = np.concatenate((zero, change))
        ends: npt.NDArray[Any] = np.concatenate((change, last))
        for start, end in zip(starts, ends, strict=False):
            if end - start >= 2:  # only multi-doc buckets can form pairs
                index.add_bucket(int(sorted_keys[start]), order[start:end].tolist())
    return index
