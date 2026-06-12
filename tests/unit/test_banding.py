"""Unit tests for LSH banding and the S-curve.

The banding parameters (b, r) set the precision/recall operating point via the
candidate probability ``P(s) = 1 - (1 - s**r)**b``. These tests pin the exact
analytical values and the (b, r) recommender.

Note:
    For ``b=16, r=8, s=0.8`` the *exact* candidate probability is
    ``1 - (1 - 0.8**8)**16 = 0.9471`` — not 0.97 (0.97 would require ~44 bands).
    We assert the mathematically correct value.
"""

from __future__ import annotations

import pytest

from dedup_pipeline.exceptions import ConfigError
from dedup_pipeline.lsh.banding import (
    BandingScheme,
    approx_threshold,
    candidate_probability,
    recommend_bands,
)


def test_s_curve_value_at_threshold() -> None:
    """P(candidate) at b=16, r=8, s=0.8 equals the exact 0.9471.

    Matters because the dedup operating point is derived from this number; an
    off-by-a-few-percent error misstates recall.
    """
    assert candidate_probability(0.8, 16, 8) == pytest.approx(0.9471, abs=1e-3)
    assert BandingScheme(128, 16, 8).probability_at(0.8) == pytest.approx(0.9471, abs=1e-3)


def test_approx_threshold_value() -> None:
    """The S-curve knee for (16, 8) is (1/16)^(1/8) = 0.7071.

    Matters because the knee tells you roughly where the curve flips from
    rejecting to accepting; mislocating it mis-tunes the system.
    """
    assert approx_threshold(16, 8) == pytest.approx(0.7071, abs=1e-3)


def test_s_curve_monotone_increasing() -> None:
    """P(candidate) increases monotonically with similarity s.

    Matters because a non-monotone curve would mean more-similar documents are
    *less* likely to be detected — a correctness failure.
    """
    scheme = BandingScheme(128, 16, 8)
    samples = [s / 20 for s in range(21)]  # 0.0 .. 1.0
    probs = [scheme.probability_at(s) for s in samples]
    assert all(b >= a for a, b in zip(probs, probs[1:], strict=False))


def test_band_slices_partition_signature() -> None:
    """Band slices tile the signature exactly with no gaps or overlaps.

    Matters because a mis-tiled signature would drop or double-count hash values
    in bucketing.
    """
    slices = BandingScheme(12, 3, 4).band_slices()
    assert slices == [(0, 4), (4, 8), (8, 12)]


def test_validation_raises_on_bad_factorization() -> None:
    """Constructing a scheme with b*r != n raises ConfigError.

    Matters because a mismatch would silently leave hash values unused or
    over-index the signature.
    """
    with pytest.raises(ConfigError):
        BandingScheme(128, 16, 7)


def test_threshold_targeting_meets_recall_floor() -> None:
    """recommend_bands(128, 0.8) achieves P >= 0.95 at the target threshold.

    Matters because the recommender must guarantee the requested recall at the
    chosen operating point.
    """
    bands, rows = recommend_bands(128, 0.8)
    assert bands * rows == 128
    assert candidate_probability(0.8, bands, rows) >= 0.95


def test_probability_endpoints() -> None:
    """P(0) = 0 and P(1) = 1 for any (b, r) (pathological extremes).

    Matters because totally dissimilar pairs must never be candidates and
    identical pairs must always be.
    """
    assert candidate_probability(0.0, 16, 8) == pytest.approx(0.0)
    assert candidate_probability(1.0, 16, 8) == pytest.approx(1.0)


def test_recommend_bands_invalid_n_raises() -> None:
    """recommend_bands with n < 1 raises ConfigError (pathological input).

    Matters because a zero/negative signature length is nonsensical and must be
    rejected loudly.
    """
    with pytest.raises(ConfigError):
        recommend_bands(0, 0.8)


def test_single_band_single_row_edge() -> None:
    """A (b=1, r=1) scheme is the identity collision rule P(s) = s (edge case).

    Matters because degenerate banding must still obey the formula.
    """
    assert candidate_probability(0.37, 1, 1) == pytest.approx(0.37)
