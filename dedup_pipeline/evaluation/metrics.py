"""Pair-level evaluation metrics for deduplication quality.

Deduplication quality is measured on *pairs*: a pair ``(i, j)`` is a true
duplicate if the ground truth says so. The system predicts a set of duplicate
pairs (the union of its clusters' internal pairs), and we compare the two sets:

    * **TP** — predicted and truly duplicate;
    * **FP** — predicted but not truly duplicate;
    * **FN** — truly duplicate but missed.

Precision ``= TP / (TP + FP)``, recall ``= TP / (TP + FN)``, ``F1`` is their
harmonic mean. Because annotating all ``O(n^2)`` pairs is infeasible, this module
also offers a sampling estimator for precision.

Responsibility:
    * Compute confusion counts, P/R/F1, and a sampled precision estimate.

Inputs:
    * Predicted and ground-truth pair sets (any pair ordering).

Outputs:
    * A :class:`ConfusionMatrix` and float metrics.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from dedup_pipeline.exceptions import EvaluationError


def canonical_pairs(pairs: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    """Normalise pairs to ``(min, max)`` form and drop self-pairs.

    Args:
        pairs: An iterable of index pairs in any order.

    Returns:
        A set of canonical ``(i, j)`` pairs with ``i < j``.

    Raises:
        EvaluationError: If a pair is malformed (not two distinct-ordering
            integers) — a self-pair ``(i, i)`` is silently dropped.

    Example:
        >>> sorted(canonical_pairs([(3, 1), (1, 3), (2, 2)]))
        [(1, 3)]
    """
    result: set[tuple[int, int]] = set()
    for pair in pairs:
        if len(pair) != 2:
            raise EvaluationError(f"expected a 2-tuple pair, got {pair!r}")
        i, j = pair
        if i == j:
            continue  # a document is not a duplicate of itself
        result.add((i, j) if i < j else (j, i))
    return result


@dataclass(frozen=True)
class ConfusionMatrix:
    """Pair-level confusion counts with derived metrics.

    Attributes:
        true_positives: Correctly predicted duplicate pairs.
        false_positives: Predicted pairs that are not true duplicates.
        false_negatives: True duplicate pairs that were missed.
    """

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """Precision ``TP / (TP + FP)``; ``1.0`` when no pair was predicted."""
        denom = self.true_positives + self.false_positives
        if denom == 0:
            return 1.0  # vacuously precise: no predictions => no false positives
        return self.true_positives / denom

    @property
    def recall(self) -> float:
        """Recall ``TP / (TP + FN)``; ``1.0`` when there are no true pairs."""
        denom = self.true_positives + self.false_negatives
        if denom == 0:
            return 1.0  # nothing to find => perfect recall by convention
        return self.true_positives / denom

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall; ``0.0`` if both are zero."""
        p, r = self.precision, self.recall
        if p + r == 0.0:
            return 0.0
        return 2.0 * p * r / (p + r)


def confusion_matrix(
    predicted: Iterable[tuple[int, int]],
    ground_truth: Iterable[tuple[int, int]],
) -> ConfusionMatrix:
    """Compute the pair-level confusion matrix.

    Args:
        predicted: Predicted duplicate pairs (any ordering).
        ground_truth: True duplicate pairs (any ordering).

    Returns:
        A :class:`ConfusionMatrix`.

    Example:
        >>> cm = confusion_matrix([(0, 1), (2, 3)], [(0, 1), (4, 5)])
        >>> (cm.true_positives, cm.false_positives, cm.false_negatives)
        (1, 1, 1)
    """
    pred = canonical_pairs(predicted)
    truth = canonical_pairs(ground_truth)
    true_positives = len(pred & truth)
    return ConfusionMatrix(
        true_positives=true_positives,
        false_positives=len(pred) - true_positives,
        false_negatives=len(truth) - true_positives,
    )


def precision_recall_f1(
    predicted: Iterable[tuple[int, int]],
    ground_truth: Iterable[tuple[int, int]],
) -> tuple[float, float, float]:
    """Return ``(precision, recall, f1)`` for predicted vs. ground-truth pairs.

    Args:
        predicted: Predicted duplicate pairs.
        ground_truth: True duplicate pairs.

    Returns:
        A ``(precision, recall, f1)`` tuple.

    Example:
        >>> p, r, f = precision_recall_f1([(0, 1)], [(0, 1), (2, 3)])
        >>> (round(p, 3), round(r, 3), round(f, 3))
        (1.0, 0.5, 0.667)
    """
    cm = confusion_matrix(predicted, ground_truth)
    return cm.precision, cm.recall, cm.f1


def pairs_from_clusters(clusters: Iterable[Iterable[int]]) -> set[tuple[int, int]]:
    """Expand clusters into the set of all intra-cluster pairs.

    Args:
        clusters: An iterable of clusters (each an iterable of indices).

    Returns:
        The set of canonical ``(i, j)`` pairs implied by the clusters.

    Example:
        >>> sorted(pairs_from_clusters([[0, 1, 2]]))
        [(0, 1), (0, 2), (1, 2)]
    """
    pairs: set[tuple[int, int]] = set()
    for cluster in clusters:
        for i, j in itertools.combinations(sorted(cluster), 2):
            pairs.add((i, j))
    return pairs


def estimate_precision(
    predicted: Iterable[tuple[int, int]],
    is_true_duplicate: Callable[[tuple[int, int]], bool],
    sample_size: int,
    seed: int = 0,
) -> float:
    """Estimate precision by annotating a random sample of predicted pairs.

    This avoids judging all ``O(n^2)`` pairs: precision only depends on the
    predicted-positive set, so a uniform sample of predicted pairs gives an
    unbiased precision estimate.

    Args:
        predicted: The predicted duplicate pairs.
        is_true_duplicate: A judge returning whether a pair is a true duplicate
            (e.g. a human annotation lookup or an exact-Jaccard check).
        sample_size: Number of predicted pairs to sample (capped at the total).
        seed: RNG seed for reproducible sampling.

    Returns:
        The fraction of sampled pairs judged true duplicates.

    Raises:
        EvaluationError: If there are no predicted pairs to sample, or
            ``sample_size < 1``.

    Example:
        >>> est = estimate_precision([(0, 1), (2, 3)], lambda p: p == (0, 1), 2)
        >>> est
        0.5
    """
    pool = list(canonical_pairs(predicted))
    if not pool:
        raise EvaluationError("cannot estimate precision: no predicted pairs")
    if sample_size < 1:
        raise EvaluationError(f"sample_size must be >= 1, got {sample_size}")
    rng = random.Random(seed)
    k = min(sample_size, len(pool))
    sample = rng.sample(pool, k)
    hits = sum(1 for pair in sample if is_true_duplicate(pair))
    return hits / k
