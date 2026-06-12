"""LSH banding: split a signature into bands and analyse the S-curve.

Locality-Sensitive Hashing via banding makes candidate generation sub-quadratic
[Indyk & Motwani 1998]. The length-``n`` signature is split into ``b`` bands of
``r`` rows (``n = b * r``). Two documents become a *candidate pair* if they agree
on **all** ``r`` rows of **at least one** band. For documents of Jaccard
similarity ``s`` the candidate probability is

    P(s) = 1 - (1 - s**r)**b

which traces an S-curve whose knee sits near ``(1/b)**(1/r)`` [Leskovec et al.
2014].

Responsibility:
    * Provide :class:`BandingScheme` (validated b/r decomposition + S-curve
      utilities) and a ``(b, r)`` recommender for a target threshold.

Inputs:
    * ``num_hash_functions``, ``num_bands``, ``num_rows`` (or a config).

Outputs:
    * Band slices, candidate probabilities, S-curve samples, recommendations.
"""

from __future__ import annotations

from dedup_pipeline.config import PipelineConfig
from dedup_pipeline.exceptions import ConfigError

# Default recall floor used by recommend_bands: the recommended (b, r) must put
# at least this candidate probability at the target Jaccard threshold.
DEFAULT_RECALL_FLOOR: float = 0.95


def candidate_probability(s: float, num_bands: int, num_rows: int) -> float:
    """Probability that two documents of similarity ``s`` share a bucket.

    Implements ``P(s) = 1 - (1 - s**r)**b`` [Leskovec et al. 2014].

    Args:
        s: Jaccard similarity in ``[0, 1]``.
        num_bands: Number of bands ``b``.
        num_rows: Rows per band ``r``.

    Returns:
        The candidate probability in ``[0, 1]``.

    Example:
        >>> round(candidate_probability(0.8, 16, 8), 4)
        0.9471
    """
    return 1.0 - (1.0 - s**num_rows) ** num_bands


def approx_threshold(num_bands: int, num_rows: int) -> float:
    """Approximate S-curve knee location ``(1/b)**(1/r)``.

    Args:
        num_bands: Number of bands ``b``.
        num_rows: Rows per band ``r``.

    Returns:
        The similarity at which the S-curve rises most steeply.

    Example:
        >>> round(approx_threshold(16, 8), 4)
        0.7071
    """
    return float((1.0 / num_bands) ** (1.0 / num_rows))


class BandingScheme:
    """A validated decomposition of a signature into ``b`` bands of ``r`` rows.

    Thread-safety:
        Immutable after construction; safe to share across threads/processes.

    Args:
        num_hash_functions: Signature length ``n``.
        num_bands: Number of bands ``b``.
        num_rows: Rows per band ``r``.

    Raises:
        ConfigError: If ``b * r != n`` or any value is non-positive.

    Example:
        >>> bs = BandingScheme(128, 16, 8)
        >>> bs.band_slices()[0]
        (0, 8)
        >>> round(bs.probability_at(0.8), 4)
        0.9471
    """

    def __init__(
        self, num_hash_functions: int, num_bands: int, num_rows: int
    ) -> None:
        if num_bands < 1 or num_rows < 1:
            raise ConfigError(
                f"num_bands ({num_bands}) and num_rows ({num_rows}) must be >= 1"
            )
        if num_bands * num_rows != num_hash_functions:
            raise ConfigError(
                f"num_bands ({num_bands}) * num_rows ({num_rows}) = "
                f"{num_bands * num_rows} != num_hash_functions "
                f"({num_hash_functions})"
            )
        self._n = num_hash_functions
        self._b = num_bands
        self._r = num_rows

    @classmethod
    def from_config(cls, config: PipelineConfig) -> BandingScheme:
        """Build a banding scheme from a :class:`PipelineConfig`.

        Args:
            config: The pipeline configuration.

        Returns:
            A validated :class:`BandingScheme`.

        Raises:
            ConfigError: If the config's b/r do not factor n (already enforced
                by the config validator, re-checked here for defensive safety).
        """
        return cls(config.num_hash_functions, config.lsh_bands, config.lsh_rows)

    @property
    def num_bands(self) -> int:
        """Number of bands ``b``."""
        return self._b

    @property
    def num_rows(self) -> int:
        """Rows per band ``r``."""
        return self._r

    @property
    def num_hash_functions(self) -> int:
        """Signature length ``n``."""
        return self._n

    def band_slices(self) -> list[tuple[int, int]]:
        """Return the ``(start, end)`` column slice for each band.

        Returns:
            A list of ``b`` half-open ``(start, end)`` index pairs.

        Example:
            >>> BandingScheme(6, 3, 2).band_slices()
            [(0, 2), (2, 4), (4, 6)]
        """
        return [(i * self._r, (i + 1) * self._r) for i in range(self._b)]

    def probability_at(self, s: float) -> float:
        """Candidate probability at similarity ``s`` for this scheme.

        Args:
            s: Jaccard similarity.

        Returns:
            ``P(s) = 1 - (1 - s**r)**b``.
        """
        return candidate_probability(s, self._b, self._r)

    def approx_threshold(self) -> float:
        """The S-curve knee location for this scheme.

        Returns:
            ``(1/b)**(1/r)``.
        """
        return approx_threshold(self._b, self._r)

    def s_curve(self, points: list[float]) -> list[tuple[float, float]]:
        """Sample the S-curve at the given similarities.

        Args:
            points: Similarity values to evaluate.

        Returns:
            ``(s, P(s))`` pairs, one per input point.

        Example:
            >>> [round(p, 3) for _, p in BandingScheme(128, 16, 8).s_curve([0.5, 0.9])]
            [0.01, 1.0]
        """
        return [(s, self.probability_at(s)) for s in points]


def recommend_bands(
    num_hash_functions: int,
    target_threshold: float,
    recall_floor: float = DEFAULT_RECALL_FLOOR,
) -> tuple[int, int]:
    """Recommend ``(b, r)`` for a target Jaccard threshold.

    Among all factorizations ``b * r == num_hash_functions``, return the one
    whose candidate probability at ``target_threshold`` is at least
    ``recall_floor`` and whose S-curve knee is closest to (but not above) the
    threshold — i.e. the highest-precision scheme that still meets the recall
    bar. If none meets the bar, return the factorization with the highest
    probability at the threshold (maximising recall).

    Args:
        num_hash_functions: Signature length ``n`` to factor.
        target_threshold: The Jaccard threshold to tune for.
        recall_floor: Minimum acceptable ``P(target_threshold)``.

    Returns:
        A ``(num_bands, num_rows)`` tuple.

    Raises:
        ConfigError: If ``num_hash_functions < 1``.

    Example:
        >>> b, r = recommend_bands(128, 0.8)
        >>> candidate_probability(0.8, b, r) >= 0.95
        True
    """
    if num_hash_functions < 1:
        raise ConfigError(f"num_hash_functions must be >= 1, got {num_hash_functions}")

    factorizations = [
        (b, num_hash_functions // b)
        for b in range(1, num_hash_functions + 1)
        if num_hash_functions % b == 0
    ]

    qualifying = [
        (b, r)
        for (b, r) in factorizations
        if candidate_probability(target_threshold, b, r) >= recall_floor
    ]
    if qualifying:
        # Highest precision = knee closest to threshold from below = largest knee.
        return max(qualifying, key=lambda br: approx_threshold(br[0], br[1]))
    # No scheme meets the recall floor: fall back to maximum recall at threshold.
    return max(
        factorizations,
        key=lambda br: candidate_probability(target_threshold, br[0], br[1]),
    )
