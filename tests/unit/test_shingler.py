"""Unit tests for :class:`Shingler`.

A shingler turns normalized text into the *set* of integer shingle ids that
MinHash and Jaccard similarity operate on. Set semantics, hash stability, and
CJK fallback are load-bearing: if any drift, identical documents stop producing
identical signatures and deduplication silently breaks. These tests pin that
contract by reading behaviour through the public methods only.
"""

from __future__ import annotations

import pytest

from dedup_pipeline.text_processing.normalizer import TextNormalizer
from dedup_pipeline.text_processing.shingler import Shingler
from dedup_pipeline.text_processing.tokenizer import Tokenizer


@pytest.fixture
def tokenizer() -> Tokenizer:
    """The shared stateless tokenizer used by every shingler."""
    return Tokenizer()


@pytest.fixture
def normalizer() -> TextNormalizer:
    """A normalizer with the default CJK threshold (used only for is_cjk)."""
    return TextNormalizer(0.2)


def test_returns_set_of_ints_with_expected_count(
    tokenizer: Tokenizer, normalizer: TextNormalizer
) -> None:
    """Word mode k=3 over 5 tokens yields a set of 3 integer shingle ids.

    Matters because the shingle count is the Jaccard overlap denominator and the
    type must be a hashable int set for MinHash; a wrong count or type would
    corrupt every similarity estimate.
    """
    sh = Shingler(3, "word", 42, tokenizer, normalizer)
    shingles = sh.shingle_text("a b c d e")
    assert isinstance(shingles, set)
    assert len(shingles) == 3  # 5 tokens, trigrams -> 5 - 3 + 1
    assert all(isinstance(s, int) for s in shingles)


def test_repeated_shingle_is_deduplicated(
    tokenizer: Tokenizer, normalizer: TextNormalizer
) -> None:
    """Char mode k=2 over 'aaa' yields a single shingle (set de-dup).

    Matters because Jaccard is a *set* similarity; a document repeating the same
    k-gram must contribute it once, or repetition would inflate self-overlap and
    distort near-duplicate detection.
    """
    sh = Shingler(2, "char", 42, tokenizer, normalizer)
    # "aaa" -> bigrams {"aa", "aa"} -> one unique shingle.
    assert len(sh.shingle_text("aaa")) == 1


def test_identical_texts_match_different_texts_differ(
    tokenizer: Tokenizer, normalizer: TextNormalizer
) -> None:
    """Equal texts give equal shingle sets; unequal texts give unequal sets.

    Matters because this is the core premise of the whole pipeline: identical
    content must hash to identical shingles (so duplicates collide) while
    distinct content must not (so unrelated docs are not falsely merged).
    """
    sh = Shingler(3, "word", 42, tokenizer, normalizer)
    text = "the quick brown fox jumps"
    assert sh.shingle_text(text) == sh.shingle_text(text)
    assert sh.shingle_text(text) != sh.shingle_text("a completely different sentence")


def test_determinism_across_instances_with_same_seed(
    tokenizer: Tokenizer, normalizer: TextNormalizer
) -> None:
    """Two shinglers with the same seed produce identical sets for one text.

    Matters because signatures must be reproducible across processes and runs;
    if the same text yielded different shingle ids per instance, distributed
    dedup workers could never agree and cached signatures would be invalid.
    """
    sh_a = Shingler(3, "word", 7, tokenizer, normalizer)
    sh_b = Shingler(3, "word", 7, tokenizer, normalizer)
    text = "deterministic hashing keeps signatures reproducible always"
    assert sh_a.shingle_text(text) == sh_b.shingle_text(text)


def test_different_seeds_decorrelate_shingles(
    tokenizer: Tokenizer, normalizer: TextNormalizer
) -> None:
    """Different seeds map the same text to different shingle ids.

    Matters because the seed must actually flow into the content hash; if it did
    not, independent runs could not be decorrelated and seed configuration would
    be a silent no-op.
    """
    text = "the quick brown fox jumps"
    sh_a = Shingler(3, "word", 1, tokenizer, normalizer)
    sh_b = Shingler(3, "word", 2, tokenizer, normalizer)
    assert sh_a.shingle_text(text) != sh_b.shingle_text(text)


def test_cjk_word_mode_falls_back_to_char(
    tokenizer: Tokenizer, normalizer: TextNormalizer
) -> None:
    """CJK text under word mode still produces a non-empty shingle set.

    Matters because whitespace word tokenization yields a single useless token
    for space-free CJK; the automatic char fallback is what keeps CJK documents
    visible to the deduplicator at all.
    """
    sh = Shingler(3, "word", 42, tokenizer, normalizer)
    cjk = "これは日本語のテキストです"
    shingles = sh.shingle_text(cjk)
    assert len(shingles) > 0
    # Sanity: a single word-token fallback would give exactly one shingle, but
    # char fallback slides over the characters, so we expect many.
    assert len(shingles) > 1


def test_shingle_batch_aligns_positionally(
    tokenizer: Tokenizer, normalizer: TextNormalizer
) -> None:
    """shingle_batch returns one set per input, in input order.

    Matters because downstream stages join shingle sets back to documents by
    position; any reordering or length mismatch would attach signatures to the
    wrong records.
    """
    sh = Shingler(3, "word", 42, tokenizer, normalizer)
    texts = ["a b c d e", "x y z w", "single short"]
    out = sh.shingle_batch(texts)
    assert isinstance(out, list)
    assert len(out) == len(texts)
    # Each element must equal the standalone shingle of the same text.
    for text, shingles in zip(texts, out, strict=False):
        assert shingles == sh.shingle_text(text)


def test_pathological_empty_string_yields_empty_set(
    tokenizer: Tokenizer, normalizer: TextNormalizer
) -> None:
    """The empty string shingles to the empty set.

    Matters because empty/whitespace-only records (post-normalization) must carry
    zero shingles, so they neither match each other nor crash the batch
    (pathological input).
    """
    sh = Shingler(3, "word", 42, tokenizer, normalizer)
    assert sh.shingle_text("") == set()
    # Char mode on empty input is likewise empty (no characters to window over).
    sh_char = Shingler(2, "char", 42, tokenizer, normalizer)
    assert sh_char.shingle_text("") == set()


def test_pathological_short_token_with_oversized_k_yields_one_shingle(
    tokenizer: Tokenizer, normalizer: TextNormalizer
) -> None:
    """A single word with k larger than the token count still yields >= 1 shingle.

    Matters because very short documents must remain matchable; the degenerate
    n-gram rule (whole token list as one shingle) guarantees they emit a shingle
    instead of vanishing from the index (pathological input).
    """
    sh = Shingler(5, "word", 42, tokenizer, normalizer)
    # One token, k=5 -> degenerate single-gram path -> exactly one shingle.
    assert len(sh.shingle_text("hello")) == 1
