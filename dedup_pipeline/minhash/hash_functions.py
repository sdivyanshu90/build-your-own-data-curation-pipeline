"""Hash primitives: a universal hash family plus a stable shingle hash.

Two distinct hashing concerns live here:

    1. **Shingle hashing** (:func:`stable_hash64`) — turn a shingle *string*
       into a 64-bit integer with ``xxhash``. This is a content hash: it must
       be fast and stable across processes/runs so the same shingle always maps
       to the same integer.

    2. **MinHash permutation hashing** (:class:`UniversalHashFamily`) — a family
       of ``n`` functions ``h_i(x) = (a_i * x + b_i) mod P`` drawn from a
       Carter-Wegman universal family [Carter & Wegman 1979]. Each function acts
       as a random permutation of the shingle-id universe, which is exactly what
       the MinHash estimator requires [Broder 1997].

The modulus ``P = 2**32 - 5`` is the largest prime below ``2**32``. Choosing a
sub-2**32 prime and reducing inputs mod ``P`` guarantees that ``a * x + b`` with
``a, x, b < P`` stays below ``2**64`` (since ``(P-1) * P < 2**64``), so the whole
computation runs in native ``uint64`` NumPy with **no overflow and no Python big
ints** — the key to vectorizing MinHash.

Responsibility:
    * Construct reproducible hash families and provide a stable content hash.

Inputs:
    * ``num_functions`` and a ``seed`` (family); shingle strings (content hash).

Outputs:
    * ``uint64`` coefficient arrays and ``uint64`` hash values.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import xxhash

from dedup_pipeline.exceptions import HashingError

# Largest prime below 2**32. Reducing shingle ids and coefficients modulo this
# value keeps every intermediate product within uint64 (see module docstring).
HASH_MODULUS: int = 4_294_967_291  # == 2**32 - 5, prime

# NumPy scalar form, reused to keep arithmetic in uint64 (a Python int operand
# would silently upcast a uint64 array to float64 and lose precision).
_MOD_U64: np.uint64 = np.uint64(HASH_MODULUS)


def stable_hash64(data: str, seed: int = 0) -> int:
    """Hash a shingle string to a 64-bit integer with ``xxhash``.

    The hash is stable across processes and Python runs (unlike the builtin
    ``hash``, which is salted per process), so a shingle always maps to the same
    integer id. That stability is required for reproducible signatures.

    Args:
        data: The shingle string to hash.
        seed: A seed mixed into the hash so independent runs can be decorrelated
            while staying reproducible for a fixed seed.

    Returns:
        An integer in ``[0, 2**64)``.

    Raises:
        HashingError: If the ``xxhash`` backend rejects the input (for example,
            a non-string slips through type checking).

    Example:
        >>> stable_hash64("abc", seed=42) == stable_hash64("abc", seed=42)
        True
    """
    try:
        return xxhash.xxh64_intdigest(data.encode("utf-8"), seed=seed)
    except (TypeError, AttributeError) as exc:  # non-str / unencodable input
        raise HashingError(f"Failed to hash shingle {data!r}: {exc}") from exc


class UniversalHashFamily:
    """A reproducible family of ``n`` universal hash functions over shingle ids.

    Each function is ``h_i(x) = (a_i * x + b_i) mod P`` with ``a_i`` drawn from
    ``[1, P)`` and ``b_i`` from ``[0, P)`` using a seeded NumPy generator. Same
    seed and ``num_functions`` ⇒ byte-identical coefficient arrays, which makes
    every downstream signature reproducible.

    Thread-safety:
        Immutable after construction. The coefficient arrays are never modified,
        so a single family may be shared across threads and processes without
        locking. (The returned arrays are the internal buffers; callers must not
        mutate them — treat them as read-only.)

    Args:
        num_functions: Number ``n`` of hash functions to construct.
        seed: Seed for the coefficient generator.

    Raises:
        HashingError: If ``num_functions < 1``.

    Example:
        >>> fam = UniversalHashFamily(num_functions=4, seed=42)
        >>> len(fam)
        4
        >>> UniversalHashFamily(4, 42).a.tolist() == UniversalHashFamily(4, 42).a.tolist()
        True
    """

    def __init__(self, num_functions: int, seed: int) -> None:
        if num_functions < 1:
            raise HashingError(
                f"num_functions must be >= 1, got {num_functions}"
            )
        self._num_functions = num_functions
        self._seed = seed
        rng = np.random.default_rng(seed)
        # a in [1, P): exclude 0 so no function collapses to the constant b.
        self._a: npt.NDArray[Any] = rng.integers(
            1, HASH_MODULUS, size=num_functions, dtype=np.uint64
        )
        # b in [0, P): the additive offset (Carter-Wegman 2-universal family).
        self._b: npt.NDArray[Any] = rng.integers(
            0, HASH_MODULUS, size=num_functions, dtype=np.uint64
        )

    def __len__(self) -> int:
        """Return the number of hash functions in the family."""
        return self._num_functions

    @property
    def a(self) -> npt.NDArray[Any]:
        """The multiplicative coefficients ``a_i`` as a ``uint64`` array."""
        return self._a

    @property
    def b(self) -> npt.NDArray[Any]:
        """The additive coefficients ``b_i`` as a ``uint64`` array."""
        return self._b

    @property
    def seed(self) -> int:
        """The seed used to generate this family."""
        return self._seed

    @property
    def modulus(self) -> int:
        """The prime modulus ``P`` used by every function in the family."""
        return HASH_MODULUS

    def hash_values(self, func_index: int, values: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Apply one hash function to an array of shingle ids.

        Args:
            func_index: Index ``i`` of the hash function to apply.
            values: A ``uint64`` array of shingle ids (any magnitude; reduced
                mod ``P`` internally).

        Returns:
            A ``uint64`` array (same shape as ``values``) of hash outputs in
            ``[0, P)``.

        Raises:
            HashingError: If ``func_index`` is out of range.

        Example:
            >>> fam = UniversalHashFamily(2, 42)
            >>> out = fam.hash_values(0, np.array([10, 20, 30], dtype=np.uint64))
            >>> bool((out < fam.modulus).all())
            True
        """
        if not 0 <= func_index < self._num_functions:
            raise HashingError(
                f"func_index {func_index} out of range [0, {self._num_functions})"
            )
        a = self._a[func_index]
        b = self._b[func_index]
        reduced = values % _MOD_U64  # bring ids into [0, P)
        # (a * reduced + b) stays < (P-1)*P < 2**64, so no uint64 overflow.
        return cast("npt.NDArray[Any]", (a * reduced + b) % _MOD_U64)
