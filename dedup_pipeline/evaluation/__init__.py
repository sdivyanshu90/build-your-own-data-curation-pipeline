"""Evaluation: pair metrics and synthetic duplicate injection."""

from __future__ import annotations

from dedup_pipeline.evaluation.metrics import (
    ConfusionMatrix,
    confusion_matrix,
    pairs_from_clusters,
    precision_recall_f1,
)
from dedup_pipeline.evaluation.synthetic_injector import (
    inject_exact_duplicates,
    inject_near_duplicates,
)

__all__ = [
    "ConfusionMatrix",
    "confusion_matrix",
    "inject_exact_duplicates",
    "inject_near_duplicates",
    "pairs_from_clusters",
    "precision_recall_f1",
]
