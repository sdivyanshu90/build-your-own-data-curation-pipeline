"""Synthetic duplicate injection for evaluation.

To measure recall exactly you need a corpus whose duplicates you *know*. This
module injects controlled duplicates into a corpus and returns the ground-truth
pair set:

    * **exact** duplicates — byte-for-byte copies;
    * **near** duplicates — copies with a controlled fraction of tokens edited,
      so the word-level Jaccard lands near a target value.

For a document of unique tokens, deleting a fraction ``f`` yields word-set
Jaccard ``J = 1 - f``; substituting a fraction ``f`` yields ``J = (1 - f) /
(1 + f)`` [Lee et al. 2022]. These inverses let callers ask for a target Jaccard
directly.

Responsibility:
    * Append known duplicates to a corpus and report the ground-truth pairs.

Inputs:
    * A corpus (``list[dict]``) and injection parameters.

Outputs:
    * A new corpus list and a set of ground-truth ``(i, j)`` pairs.
"""

from __future__ import annotations

import random
from typing import Any

from dedup_pipeline.exceptions import EvaluationError


def edit_fraction_for_jaccard(target_jaccard: float, mode: str) -> float:
    """Invert the Jaccard/edit-fraction relationship.

    Args:
        target_jaccard: Desired word-set Jaccard in ``(0, 1]``.
        mode: ``"delete"`` (``J = 1 - f``) or ``"substitute"``
            (``J = (1 - f) / (1 + f)``).

    Returns:
        The token edit fraction ``f`` in ``[0, 1)`` that achieves the target.

    Raises:
        EvaluationError: If the target is out of range or the mode is unknown.

    Example:
        >>> edit_fraction_for_jaccard(0.85, "delete")
        0.15000000000000002
        >>> round(edit_fraction_for_jaccard(0.85, "substitute"), 4)
        0.0811
    """
    if not 0.0 < target_jaccard <= 1.0:
        raise EvaluationError(
            f"target_jaccard must be in (0, 1], got {target_jaccard}"
        )
    if mode == "delete":
        return 1.0 - target_jaccard
    if mode == "substitute":
        return (1.0 - target_jaccard) / (1.0 + target_jaccard)
    raise EvaluationError(f"unknown mode {mode!r}; expected 'delete' or 'substitute'")


def make_near_duplicate(
    text: str, edit_fraction: float, rng: random.Random, mode: str = "delete"
) -> str:
    """Produce a near-duplicate of ``text`` by editing a fraction of its tokens.

    Args:
        text: The source document text.
        edit_fraction: Fraction of whitespace tokens to edit, in ``[0, 1)``.
        rng: A seeded RNG controlling which tokens are edited.
        mode: ``"delete"`` removes tokens; ``"substitute"`` replaces them with
            fresh unique tokens.

    Returns:
        The perturbed text. Edits are spread across the document so shingles
        break in multiple places rather than at a single contiguous span.

    Raises:
        EvaluationError: If ``edit_fraction`` is out of range or the mode is
            unknown.

    Example:
        >>> import random
        >>> out = make_near_duplicate("a b c d e f g h i j", 0.2, random.Random(0))
        >>> out != "a b c d e f g h i j" and len(out) > 0
        True
    """
    if not 0.0 <= edit_fraction < 1.0:
        raise EvaluationError(
            f"edit_fraction must be in [0, 1), got {edit_fraction}"
        )
    tokens = text.split()
    if not tokens:
        return text
    num_edits = round(edit_fraction * len(tokens))
    if num_edits == 0:
        return text
    edit_positions = set(rng.sample(range(len(tokens)), min(num_edits, len(tokens))))
    if mode == "delete":
        kept = [tok for idx, tok in enumerate(tokens) if idx not in edit_positions]
        return " ".join(kept)
    if mode == "substitute":
        result = list(tokens)
        for idx in edit_positions:
            # A fresh token unlikely to collide with the vocabulary.
            result[idx] = f"__synthsub{rng.randrange(10**9)}__"
        return " ".join(result)
    raise EvaluationError(f"unknown mode {mode!r}; expected 'delete' or 'substitute'")


def inject_exact_duplicates(
    records: list[dict[str, Any]],
    num_pairs: int,
    text_field: str = "text",
    id_field: str = "id",
    seed: int = 0,
) -> tuple[list[dict[str, Any]], set[tuple[int, int]]]:
    """Append exact copies of randomly chosen records.

    Args:
        records: The base corpus (not mutated; a shallow copy is returned).
        num_pairs: Number of exact-duplicate pairs to inject.
        text_field: The text key (copied verbatim).
        id_field: The id key (the copy gets a fresh ``dup{k}`` id).
        seed: RNG seed for reproducible source selection.

    Returns:
        A tuple ``(new_records, ground_truth_pairs)`` where each ground-truth
        pair is ``(source_index, duplicate_index)`` in the returned list.

    Raises:
        EvaluationError: If ``num_pairs`` exceeds the corpus size or is negative.

    Example:
        >>> recs = [{"id": str(i), "text": f"doc {i}"} for i in range(5)]
        >>> new, gt = inject_exact_duplicates(recs, 2, seed=0)
        >>> len(new), len(gt)
        (7, 2)
    """
    if not 0 <= num_pairs <= len(records):
        raise EvaluationError(
            f"num_pairs ({num_pairs}) must be in [0, {len(records)}]"
        )
    rng = random.Random(seed)
    new_records = list(records)
    ground_truth: set[tuple[int, int]] = set()
    source_indices = rng.sample(range(len(records)), num_pairs)
    for k, src_idx in enumerate(source_indices):
        dup = dict(records[src_idx])
        dup[id_field] = f"dup{k}"
        dup_idx = len(new_records)
        new_records.append(dup)
        ground_truth.add((src_idx, dup_idx))
    return new_records, ground_truth


def inject_near_duplicates(
    records: list[dict[str, Any]],
    num_pairs: int,
    target_jaccard: float,
    mode: str = "delete",
    text_field: str = "text",
    id_field: str = "id",
    seed: int = 0,
) -> tuple[list[dict[str, Any]], set[tuple[int, int]]]:
    """Append near-duplicate copies at approximately a target word-set Jaccard.

    Args:
        records: The base corpus (not mutated; a shallow copy is returned).
        num_pairs: Number of near-duplicate pairs to inject.
        target_jaccard: Desired approximate word-set Jaccard of each pair.
        mode: ``"delete"`` or ``"substitute"`` (see :func:`make_near_duplicate`).
        text_field: The text key.
        id_field: The id key (the copy gets a fresh ``neardup{k}`` id).
        seed: RNG seed for reproducible selection and perturbation.

    Returns:
        A tuple ``(new_records, ground_truth_pairs)``.

    Raises:
        EvaluationError: If ``num_pairs`` exceeds the corpus size or parameters
            are invalid.

    Example:
        >>> recs = [{"id": str(i), "text": " ".join(f"w{j}" for j in range(50))}
        ...         for i in range(5)]
        >>> new, gt = inject_near_duplicates(recs, 2, 0.85, seed=1)
        >>> len(new), len(gt)
        (7, 2)
    """
    if not 0 <= num_pairs <= len(records):
        raise EvaluationError(
            f"num_pairs ({num_pairs}) must be in [0, {len(records)}]"
        )
    edit_fraction = edit_fraction_for_jaccard(target_jaccard, mode)
    rng = random.Random(seed)
    new_records = list(records)
    ground_truth: set[tuple[int, int]] = set()
    source_indices = rng.sample(range(len(records)), num_pairs)
    for k, src_idx in enumerate(source_indices):
        source_text = str(records[src_idx].get(text_field, ""))
        near_text = make_near_duplicate(source_text, edit_fraction, rng, mode)
        dup = dict(records[src_idx])
        dup[id_field] = f"neardup{k}"
        dup[text_field] = near_text
        dup_idx = len(new_records)
        new_records.append(dup)
        ground_truth.add((src_idx, dup_idx))
    return new_records, ground_truth
