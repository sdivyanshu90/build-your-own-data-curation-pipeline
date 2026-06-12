"""Unit tests for the pair-level evaluation metrics.

Deduplication quality is judged on *pairs*: the system's predicted duplicate
pairs are compared against a known ground truth to derive precision, recall, and
F1. These metrics are the contract by which the whole pipeline is declared
correct or not, so their pair canonicalisation, confusion counting, and the
boundary conventions for empty inputs must be exact and reproducible.
"""

from __future__ import annotations

import pytest

from dedup_pipeline.evaluation.metrics import (
    ConfusionMatrix,
    canonical_pairs,
    confusion_matrix,
    estimate_precision,
    pairs_from_clusters,
    precision_recall_f1,
)
from dedup_pipeline.exceptions import EvaluationError


def test_canonical_pairs_normalizes_order_and_dedups() -> None:
    """(3, 1) and (1, 3) collapse to a single canonical (1, 3) pair.

    Matters because predicted and ground-truth pairs arrive in arbitrary order;
    if ordering were significant the same duplicate relationship counted twice
    (or as a miss) would corrupt every confusion count downstream.
    """
    assert canonical_pairs([(3, 1), (1, 3)]) == {(1, 3)}


def test_canonical_pairs_drops_self_pairs() -> None:
    """A self-pair (2, 2) is silently dropped from the canonical set.

    Matters because a document is never a duplicate of itself; admitting
    self-pairs would inflate true positives and fabricate perfect-looking
    precision.
    """
    assert canonical_pairs([(2, 2), (0, 1)]) == {(0, 1)}


def test_confusion_matrix_partial_overlap() -> None:
    """predicted={(0,1),(2,3)} vs truth={(0,1),(4,5)} gives tp=1, fp=1, fn=1.

    Matters because this is the core counting step: one shared pair is a true
    positive, the unmatched prediction a false positive, and the unmatched truth
    a false negative. Mis-bucketing any of these mis-scores the pipeline.
    """
    cm = confusion_matrix({(0, 1), (2, 3)}, {(0, 1), (4, 5)})
    assert cm.true_positives == 1
    assert cm.false_positives == 1
    assert cm.false_negatives == 1
    assert cm.precision == 0.5
    assert cm.recall == 0.5
    assert cm.f1 == 0.5


def test_confusion_matrix_perfect_prediction() -> None:
    """When predicted == truth, precision == recall == f1 == 1.0.

    Matters because a perfect detector must score a perfect 1.0; any deviation
    would mean the metric penalises correct behaviour and cannot certify the
    pipeline.
    """
    truth = {(0, 1), (2, 3), (4, 5)}
    cm = confusion_matrix(set(truth), set(truth))
    assert cm.false_positives == 0
    assert cm.false_negatives == 0
    assert cm.precision == 1.0
    assert cm.recall == 1.0
    assert cm.f1 == 1.0


def test_pairs_from_clusters_expands_intra_cluster_pairs() -> None:
    """A 3-element cluster expands to all three of its internal pairs.

    Matters because the pipeline reports duplicates as clusters but is scored on
    pairs; the cluster -> pair expansion must be the complete combinatorial set
    or recall is undercounted.
    """
    assert pairs_from_clusters([[0, 1, 2]]) == {(0, 1), (0, 2), (1, 2)}


def test_precision_recall_f1_matches_confusion_matrix_properties() -> None:
    """precision_recall_f1 returns exactly the ConfusionMatrix property values.

    Matters because the convenience function is documented as a thin wrapper; if
    it diverged from the ConfusionMatrix properties, callers using the two paths
    would get inconsistent scores for the same data.
    """
    predicted = {(0, 1), (2, 3), (6, 7)}
    truth = {(0, 1), (4, 5)}
    cm = confusion_matrix(predicted, truth)
    p, r, f1 = precision_recall_f1(predicted, truth)
    assert p == cm.precision
    assert r == cm.recall
    assert f1 == cm.f1


def test_estimate_precision_recovers_true_fraction() -> None:
    """With a judge accepting only (0, 1) of two equally-likely pairs, est=0.5.

    Matters because precision is often estimated by sampling predicted pairs;
    with a sample large enough to cover the whole pool the estimate must equal
    the exact fraction of true positives, or sampled audits would mislead.
    """
    predicted = {(0, 1), (2, 3)}
    est = estimate_precision(predicted, lambda p: p == (0, 1), sample_size=100)
    assert est == 0.5


def test_estimate_precision_empty_predicted_raises() -> None:
    """estimate_precision on an empty prediction set raises EvaluationError.

    Matters because precision over zero predictions is undefined for a sampling
    estimator (nothing to draw); failing loudly prevents a silent divide-by-zero
    or a misleading 0.0.
    """
    with pytest.raises(EvaluationError):
        estimate_precision(set(), lambda p: True, sample_size=10)


def test_estimate_precision_nonpositive_sample_size_raises() -> None:
    """estimate_precision with sample_size < 1 raises EvaluationError.

    Matters because a zero or negative sample yields no data and a 0/0 estimate;
    rejecting it keeps the audit honest about insufficient sampling.
    """
    with pytest.raises(EvaluationError):
        estimate_precision({(0, 1)}, lambda p: True, sample_size=0)


def test_empty_predicted_and_empty_truth_is_vacuously_perfect() -> None:
    """No predictions and no true pairs yields precision=recall=f1=1.0.

    Matters because a corpus with genuinely no duplicates must score 1.0 (the
    detector correctly found nothing); the documented convention sets precision
    to 1.0 when tp+fp==0 and recall to 1.0 when tp+fn==0, so f1 is also 1.0.
    """
    cm = confusion_matrix(set(), set())
    assert cm.true_positives == 0
    assert cm.false_positives == 0
    assert cm.false_negatives == 0
    assert cm.precision == 1.0
    assert cm.recall == 1.0
    assert cm.f1 == 1.0


def test_nonempty_predicted_empty_truth_zero_precision() -> None:
    """Predicting pairs when truth is empty gives precision 0.0, recall 1.0.

    Matters because every prediction here is a false positive (tp=0, fp>0), so
    precision is 0/(0+fp)=0.0; recall stays 1.0 by convention since there is
    nothing to miss. f1 must therefore collapse to 0.0.
    """
    cm = confusion_matrix({(0, 1), (2, 3)}, set())
    assert cm.true_positives == 0
    assert cm.false_positives == 2
    assert cm.false_negatives == 0
    assert cm.precision == 0.0
    assert cm.recall == 1.0
    assert cm.f1 == 0.0


def test_canonical_pairs_malformed_pair_raises() -> None:
    """A pair whose length is not 2 raises EvaluationError (pathological).

    Matters because malformed pairs signal upstream corruption; canonicalising a
    3-tuple by ignoring an element would silently miscount duplicates, so the
    metric must reject it instead.
    """
    with pytest.raises(EvaluationError):
        canonical_pairs([(0, 1, 2)])


def test_confusion_matrix_only_self_pairs_is_empty(
) -> None:
    """A prediction of only self-pairs canonicalises to empty (pathological).

    Matters because self-pairs carry no duplicate information; once dropped, the
    confusion matrix must behave as the empty-empty vacuous case rather than
    crediting phantom true positives.
    """
    cm = confusion_matrix({(2, 2), (3, 3)}, set())
    assert cm.true_positives == 0
    assert cm.false_positives == 0
    assert cm.false_negatives == 0
    assert cm.precision == 1.0
    assert cm.recall == 1.0
    assert cm.f1 == 1.0


def test_confusion_matrix_dataclass_is_frozen() -> None:
    """ConfusionMatrix is an immutable (frozen) dataclass.

    Matters because confusion counts are an evaluation record; allowing in-place
    mutation of tp/fp/fn after computation would let a derived metric silently
    contradict the counts it was computed from.
    """
    cm = ConfusionMatrix(true_positives=1, false_positives=0, false_negatives=0)
    with pytest.raises(Exception):
        cm.true_positives = 5  # type: ignore[misc]
