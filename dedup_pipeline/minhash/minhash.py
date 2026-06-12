"""MinHash signature computation.

MinHash compresses a shingle *set* into a fixed-length integer *signature* such
that the fraction of equal signature positions between two documents is an
unbiased estimate of their Jaccard similarity [Broder 1997]. With ``n`` hash
functions the estimator has standard error ``~1/sqrt(n)``.

This module provides :class:`MinHasher` with a scikit-learn-style
``fit``/``transform``/``batch_transform`` API. The batch path is vectorized over
the *hash functions* and uses :func:`numpy.minimum.reduceat` for a per-document
segmented minimum, so there is **no Python loop over documents**. An optional
Numba kernel (enabled by config) replaces the inner NumPy work for large
corpora.

Responsibility:
    * Turn shingle sets into a ``(n_docs, n_hashes)`` ``uint32`` signature matrix
      and estimate Jaccard from signatures.

Inputs:
    * Shingle sets (``set[int]``) from
      :class:`~dedup_pipeline.text_processing.shingler.Shingler`.

Outputs:
    * ``uint32`` NumPy signatures.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from dedup_pipeline.exceptions import HashingError
from dedup_pipeline.minhash.hash_functions import (
    HASH_MODULUS,
    UniversalHashFamily,
)

# uint64 modulus reused so every operand stays uint64 (a Python-int operand
# would upcast uint64 arrays to float64).
_MOD_U64: np.uint64 = np.uint64(HASH_MODULUS)

# Signature value for an empty shingle set. It is larger than any real hash
# output (which is < HASH_MODULUS < 2**32 - 1), so an empty document never
# collides with a non-empty one; two empty documents share this signature and
# are (correctly) treated as duplicates of each other.
EMPTY_SIGNATURE_VALUE: int = (1 << 32) - 1  # 2**32 - 1, fits uint32

# Lazily-built, process-cached Numba kernel (see _get_numba_kernel).
_NUMBA_KERNEL: Callable[..., npt.NDArray[Any]] | None = None


def _get_numba_kernel() -> Callable[..., npt.NDArray[Any]]:
    """Build (once) and return the Numba-compiled signature kernel.

    The kernel is compiled on first use and cached for the process. Numba is an
    optional dependency, so the import happens here rather than at module load.

    Returns:
        The compiled kernel callable.

    Raises:
        HashingError: If ``numba`` is not installed.
    """
    global _NUMBA_KERNEL
    if _NUMBA_KERNEL is not None:
        return _NUMBA_KERNEL
    try:
        from numba import njit
    except ImportError as exc:  # optional dependency missing  # pragma: no cover
        raise HashingError(
            "use_numba=True but the optional 'numba' package is not installed"
        ) from exc

    @njit(cache=True)  # type: ignore[misc]
    def _kernel(  # pragma: no cover - JIT machine code is untraceable by coverage.py
        all_x: npt.NDArray[Any],
        seg_starts: npt.NDArray[Any],
        seg_lengths: npt.NDArray[Any],
        coeff_a: npt.NDArray[Any],
        coeff_b: npt.NDArray[Any],
        modulus: np.uint64,
        num_hashes: int,
    ) -> npt.NDArray[Any]:
        # This body executes as compiled machine code under Numba, so coverage.py
        # (a CPython bytecode tracer) cannot observe its lines even though
        # test_numba_path_matches_numpy_path exercises it end to end.
        n_docs = seg_starts.shape[0]
        out = np.empty((n_docs, num_hashes), dtype=np.uint32)
        for d in range(n_docs):
            start = seg_starts[d]
            length = seg_lengths[d]
            for j in range(num_hashes):
                aj = coeff_a[j]
                bj = coeff_b[j]
                best = modulus  # every hash output is < modulus, so this is +inf
                for t in range(length):
                    x = all_x[start + t] % modulus
                    h = (aj * x + bj) % modulus
                    if h < best:
                        best = h
                out[d, j] = np.uint32(best)
        return out

    _NUMBA_KERNEL = cast("Callable[..., npt.NDArray[Any]]", _kernel)
    return _NUMBA_KERNEL


class MinHasher:
    """Compute and compare MinHash signatures.

    Call :meth:`fit` once to construct the hash family, then :meth:`transform`
    (one document) or :meth:`batch_transform` (many). Both produce identical
    results for the same document, so batching is a pure performance choice.

    Thread-safety:
        After :meth:`fit`, the instance is effectively immutable (it only reads
        the frozen hash family), so a fitted :class:`MinHasher` may be shared
        across threads and processes. Calling :meth:`fit` concurrently with
        :meth:`transform` is not supported; fit first, then share.

    Args:
        num_hash_functions: Signature length ``n``.
        seed: Seed for the hash family (reproducibility).
        use_numba: If ``True``, :meth:`batch_transform` uses the Numba kernel.

    Example:
        >>> mh = MinHasher(num_hash_functions=128, seed=42).fit()
        >>> sig_a = mh.transform({1, 2, 3, 4})
        >>> sig_b = mh.transform({1, 2, 3, 4})
        >>> MinHasher.estimate_jaccard(sig_a, sig_b)
        1.0
    """

    def __init__(
        self,
        num_hash_functions: int,
        seed: int,
        use_numba: bool = False,
    ) -> None:
        self._n = num_hash_functions
        self._seed = seed
        self._use_numba = use_numba
        self._family: UniversalHashFamily | None = None

    @property
    def num_hash_functions(self) -> int:
        """The signature length ``n``."""
        return self._n

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._family is not None

    def fit(self) -> MinHasher:
        """Construct the universal hash family.

        MinHash does not learn from data; ``fit`` simply builds the (seeded)
        hash family, mirroring the scikit-learn estimator API so the object can
        be configured once and reused.

        Returns:
            ``self``, to allow ``MinHasher(...).fit()`` chaining.

        Example:
            >>> MinHasher(8, 42).fit().is_fitted
            True
        """
        self._family = UniversalHashFamily(self._n, self._seed)
        return self

    def _require_fitted(self) -> UniversalHashFamily:
        """Return the fitted family or raise.

        Returns:
            The constructed :class:`UniversalHashFamily`.

        Raises:
            HashingError: If :meth:`fit` has not been called.
        """
        if self._family is None:
            raise HashingError("MinHasher.fit() must be called before transform")
        return self._family

    def transform(self, shingles: set[int]) -> npt.NDArray[Any]:
        """Compute the signature of a single shingle set.

        Complexity:
            ``O(n * m)`` for ``n`` hash functions and ``m`` shingles, via a
            single ``(n, m)`` broadcast minimum.

        Args:
            shingles: A document's integer shingle set.

        Returns:
            A ``(n,)`` ``uint32`` signature. An empty set yields a signature of
            all :data:`EMPTY_SIGNATURE_VALUE`.

        Example:
            >>> mh = MinHasher(8, 42).fit()
            >>> mh.transform(set()).tolist() == [EMPTY_SIGNATURE_VALUE] * 8
            True
        """
        family = self._require_fitted()
        if not shingles:
            return np.full(self._n, EMPTY_SIGNATURE_VALUE, dtype=np.uint32)
        x = np.fromiter(shingles, dtype=np.uint64, count=len(shingles))
        xr = x % _MOD_U64  # reduce ids into [0, P)
        a = family.a[:, None]  # (n, 1)
        b = family.b[:, None]  # (n, 1)
        # (a * x + b) mod P stays < (P-1)*P < 2**64, so uint64 never overflows.
        hashed = (a * xr[None, :] + b) % _MOD_U64  # (n, m)
        return cast("npt.NDArray[Any]", hashed.min(axis=1).astype(np.uint32))

    def batch_transform(self, shingle_sets: list[set[int]]) -> npt.NDArray[Any]:
        """Compute signatures for many documents at once.

        Vectorized over hash functions; documents are handled by a segmented
        minimum (no Python loop over documents). Produces results identical to
        calling :meth:`transform` on each set.

        Complexity:
            ``O(n * T)`` for ``n`` hash functions and ``T`` total shingles across
            the batch (the NumPy path loops ``n`` times over a length-``T``
            array; the Numba path fuses the loops).

        Args:
            shingle_sets: One shingle set per document.

        Returns:
            A ``(len(shingle_sets), n)`` ``uint32`` signature matrix, row-aligned
            with the input. Rows for empty sets are all :data:`EMPTY_SIGNATURE_VALUE`.

        Example:
            >>> mh = MinHasher(16, 42).fit()
            >>> batch = mh.batch_transform([{1, 2, 3}, {1, 2, 3}])
            >>> bool((batch[0] == batch[1]).all())
            True
        """
        family = self._require_fitted()
        n_docs = len(shingle_sets)
        signatures = np.empty((n_docs, self._n), dtype=np.uint32)

        nonempty_indices = [i for i, s in enumerate(shingle_sets) if s]
        for i, s in enumerate(shingle_sets):
            if not s:
                signatures[i, :] = EMPTY_SIGNATURE_VALUE
        if not nonempty_indices:
            return signatures

        lengths = [len(shingle_sets[i]) for i in nonempty_indices]
        total = sum(lengths)
        all_x = np.fromiter(
            (v for i in nonempty_indices for v in shingle_sets[i]),
            dtype=np.uint64,
            count=total,
        )
        all_x %= _MOD_U64
        seg_lengths = np.asarray(lengths, dtype=np.int64)
        # Segment start offsets for reduceat / the Numba kernel.
        seg_starts = np.zeros(len(lengths), dtype=np.int64)
        seg_starts[1:] = np.cumsum(seg_lengths)[:-1]
        idx = np.asarray(nonempty_indices, dtype=np.int64)

        if self._use_numba:
            computed = _get_numba_kernel()(
                all_x, seg_starts, seg_lengths, family.a, family.b, _MOD_U64, self._n
            )
            signatures[idx, :] = computed
            return signatures

        a = family.a
        b = family.b
        for j in range(self._n):
            # (a_j * x + b_j) mod P over all shingles, then per-document min.
            hashed = (a[j] * all_x + b[j]) % _MOD_U64  # (T,)
            seg_min = np.minimum.reduceat(hashed, seg_starts)  # (n_nonempty,)
            signatures[idx, j] = seg_min.astype(np.uint32)
        return signatures

    @staticmethod
    def estimate_jaccard(sig_a: npt.NDArray[Any], sig_b: npt.NDArray[Any]) -> float:
        """Estimate Jaccard similarity from two signatures.

        The estimator is the fraction of equal signature positions, an unbiased
        estimate of ``J(A, B)`` [Broder 1997].

        Args:
            sig_a: First signature, shape ``(n,)``.
            sig_b: Second signature, shape ``(n,)``.

        Returns:
            The estimated Jaccard similarity in ``[0.0, 1.0]``.

        Raises:
            HashingError: If the signatures have different lengths.

        Example:
            >>> import numpy as np
            >>> a = np.array([1, 2, 3, 4], dtype=np.uint32)
            >>> b = np.array([1, 2, 9, 9], dtype=np.uint32)
            >>> MinHasher.estimate_jaccard(a, b)
            0.5
        """
        if sig_a.shape != sig_b.shape:
            raise HashingError(
                f"signature shapes differ: {sig_a.shape} vs {sig_b.shape}"
            )
        matches: Any = np.count_nonzero(sig_a == sig_b)
        return float(matches) / float(sig_a.shape[0])
