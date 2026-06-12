"""Unit tests for synthetic duplicate injection.

Recall can only be measured exactly against duplicates whose existence is known,
so this module fabricates controlled exact and near duplicates and reports their
ground-truth pairs. The injected ground truth is the yardstick every downstream
recall number is measured against; if injection mislabels a pair, places a copy
at the wrong index, or drifts off its target Jaccard, every evaluation built on
it is silently wrong. Determinism is equally load-bearing: benchmarks must be
byte-stable across runs.
"""

from __future__ import annotations

import random

import pytest

from dedup_pipeline.evaluation.synthetic_injector import (
    edit_fraction_for_jaccard,
    inject_exact_duplicates,
    inject_near_duplicates,
    make_near_duplicate,
)
from dedup_pipeline.exceptions import EvaluationError


def _word_set_jaccard(a: str, b: str) -> float:
    """Word-set Jaccard of two texts (helper, not a test)."""
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _base_records(n: int, words: int = 50) -> list[dict[str, str]]:
    """``n`` records of ``words`` distinct tokens each (helper, not a test)."""
    return [
        {"id": str(i), "text": " ".join(f"w{i}_{j}" for j in range(words))}
        for i in range(n)
    ]


def test_edit_fraction_for_jaccard_delete() -> None:
    """Delete mode inverts J = 1 - f, so target 0.85 yields f = 0.15.

    Matters because the caller asks for a Jaccard but the perturbation needs an
    edit fraction; a wrong inversion would land near-duplicates at the wrong
    similarity and miscalibrate the whole near-dup benchmark.
    """
    assert edit_fraction_for_jaccard(0.85, "delete") == pytest.approx(0.15)


def test_edit_fraction_for_jaccard_substitute() -> None:
    """Substitute mode inverts J = (1-f)/(1+f) for the target.

    Matters because substitution preserves token count, so its Jaccard-to-edit
    relationship differs from deletion; using the delete formula here would skew
    the realised similarity.
    """
    expected = (1.0 - 0.85) / (1.0 + 0.85)
    assert edit_fraction_for_jaccard(0.85, "substitute") == pytest.approx(expected)


def test_edit_fraction_for_jaccard_unknown_mode_raises() -> None:
    """An unrecognised mode raises EvaluationError.

    Matters because a typo'd mode must fail loudly rather than silently defaulting
    to one perturbation strategy and producing duplicates at an unintended
    similarity.
    """
    with pytest.raises(EvaluationError):
        edit_fraction_for_jaccard(0.85, "shuffle")


def test_edit_fraction_for_jaccard_out_of_range_raises() -> None:
    """A target Jaccard outside (0, 1] raises EvaluationError.

    Matters because Jaccard is bounded in (0, 1]; a target of 0.0 or >1.0 has no
    valid edit fraction, and accepting it would yield nonsensical (or negative)
    perturbations.
    """
    with pytest.raises(EvaluationError):
        edit_fraction_for_jaccard(0.0, "delete")
    with pytest.raises(EvaluationError):
        edit_fraction_for_jaccard(1.5, "delete")


def test_inject_exact_duplicates_counts_and_copies() -> None:
    """Injecting 2 pairs into 5 records yields 7 records, 2 gt pairs, exact copies.

    Matters because exact-duplicate recall is scored against these pairs: each gt
    pair must point at an appended record (index >= 5) whose text is a verbatim
    copy of its source, or the ground truth lies about what the detector should
    find.
    """
    base = _base_records(5)
    new_records, ground_truth = inject_exact_duplicates(base, num_pairs=2, seed=0)
    assert len(new_records) == 7
    assert len(ground_truth) == 2
    for src_idx, dup_idx in ground_truth:
        assert dup_idx >= 5
        assert new_records[dup_idx]["text"] == new_records[src_idx]["text"]


def test_inject_exact_duplicates_does_not_mutate_input() -> None:
    """The original records list is not mutated by injection.

    Matters because a fixture corpus is reused across many tests; mutating it in
    place would make later tests see contaminated, larger corpora and produce
    order-dependent failures.
    """
    base = _base_records(5)
    before = len(base)
    inject_exact_duplicates(base, num_pairs=2, seed=0)
    assert len(base) == before


def test_inject_exact_duplicates_too_many_pairs_raises() -> None:
    """Requesting more pairs than records raises EvaluationError.

    Matters because each pair samples a distinct source without replacement;
    asking for more sources than exist is impossible, and silently capping it
    would understate the requested ground-truth size.
    """
    base = _base_records(5)
    with pytest.raises(EvaluationError):
        inject_exact_duplicates(base, num_pairs=6, seed=0)


def test_make_near_duplicate_zero_edit_fraction_unchanged() -> None:
    """edit_fraction 0 returns the text unchanged.

    Matters because zero edits means the copy is identical; if it perturbed
    anyway, a "0% edited" near-dup would not actually be a near-dup at the
    requested similarity.
    """
    text = "a b c d e f g h i j"
    assert make_near_duplicate(text, 0.0, random.Random(0)) == text


def test_make_near_duplicate_delete_reduces_token_count() -> None:
    """Delete mode produces fewer tokens than the source.

    Matters because deletion is defined as token removal; the realised Jaccard
    formula J = 1 - f assumes tokens are actually dropped, so a same-length result
    would break the similarity calibration.
    """
    text = " ".join(f"t{i}" for i in range(20))
    out = make_near_duplicate(text, 0.3, random.Random(0), mode="delete")
    assert len(out.split()) < len(text.split())


def test_make_near_duplicate_substitute_preserves_count_changes_content() -> None:
    """Substitute mode keeps the token count but changes the text.

    Matters because substitution replaces tokens in place; the count must stay
    constant (its Jaccard formula depends on the union growing) while the content
    must differ, or no near-duplicate was actually created.
    """
    text = " ".join(f"t{i}" for i in range(20))
    out = make_near_duplicate(text, 0.3, random.Random(0), mode="substitute")
    assert len(out.split()) == len(text.split())
    assert out != text


def test_make_near_duplicate_out_of_range_raises() -> None:
    """edit_fraction outside [0, 1) raises EvaluationError.

    Matters because an edit fraction of 1.0 (or above) would delete the entire
    document, leaving no shared tokens and no meaningful near-duplicate; the
    bound must be enforced rather than silently producing an empty string.
    """
    with pytest.raises(EvaluationError):
        make_near_duplicate("a b c", 1.0, random.Random(0))
    with pytest.raises(EvaluationError):
        make_near_duplicate("a b c", -0.1, random.Random(0))


def test_inject_near_duplicates_counts_and_high_overlap() -> None:
    """Injecting 2 near-dup pairs yields 7 records, 2 gt pairs, high token overlap.

    Matters because near-dup recall is scored against these pairs: the copy must
    differ from its source (it is a *near* dup, not exact) yet share most tokens
    (word-set Jaccard > 0.5), confirming the perturbation landed near the target
    rather than rewriting the document.
    """
    base = _base_records(5, words=60)
    new_records, ground_truth = inject_near_duplicates(
        base, num_pairs=2, target_jaccard=0.85, seed=0
    )
    assert len(new_records) == 7
    assert len(ground_truth) == 2
    for src_idx, dup_idx in ground_truth:
        assert dup_idx >= 5
        src_text = new_records[src_idx]["text"]
        dup_text = new_records[dup_idx]["text"]
        assert dup_text != src_text
        assert _word_set_jaccard(src_text, dup_text) > 0.5


def test_inject_near_duplicates_too_many_pairs_raises() -> None:
    """Requesting more near-dup pairs than records raises EvaluationError.

    Matters because, like exact injection, sources are sampled without
    replacement; exceeding the corpus size is impossible and must fail loudly.
    """
    base = _base_records(5)
    with pytest.raises(EvaluationError):
        inject_near_duplicates(base, num_pairs=6, target_jaccard=0.85, seed=0)


def test_injection_is_deterministic_for_fixed_seed() -> None:
    """The same seed produces an identical injected corpus and ground truth.

    Matters because benchmarks must be reproducible; if injection drifted between
    runs the same code could report different recall, making regressions
    indistinguishable from noise.
    """
    base = _base_records(8, words=40)
    rec_a, gt_a = inject_near_duplicates(base, num_pairs=3, target_jaccard=0.85, seed=7)
    rec_b, gt_b = inject_near_duplicates(base, num_pairs=3, target_jaccard=0.85, seed=7)
    assert rec_a == rec_b
    assert gt_a == gt_b


def test_inject_zero_pairs_returns_corpus_unchanged() -> None:
    """num_pairs=0 returns an equal-content corpus and empty ground truth.

    Matters because a benchmark may request no injection; it must yield the
    original documents and an empty ground-truth set so the corpus stays a valid
    no-duplicate baseline.
    """
    base = _base_records(4)
    new_records, ground_truth = inject_exact_duplicates(base, num_pairs=0, seed=0)
    assert new_records == base
    assert ground_truth == set()


def test_make_near_duplicate_empty_text_returns_empty() -> None:
    """An empty source text returns empty regardless of edit fraction (pathological).

    Matters because empty documents can appear in real corpora; the perturbation
    must degrade gracefully to an empty string instead of raising or sampling
    from an empty token range.
    """
    assert make_near_duplicate("", 0.5, random.Random(0)) == ""
