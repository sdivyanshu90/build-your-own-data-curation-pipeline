"""Unit tests for :class:`Tokenizer`.

Tokenization produces the base unit stream (words or characters) that shingling
slides its k-gram window over. If tokenization is wrong, every downstream
shingle, MinHash signature, and similarity decision is wrong too; these tests
pin the exact splitting, dispatch, and n-gram behaviour the shingler relies on.
"""

from __future__ import annotations

import pytest

from dedup_pipeline.text_processing.tokenizer import Tokenizer


@pytest.fixture
def tokenizer() -> Tokenizer:
    """A shared stateless tokenizer instance."""
    return Tokenizer()


def test_word_tokens_splits_on_non_word_chars(tokenizer: Tokenizer) -> None:
    """word_tokens splits a sentence into its bare word runs.

    Matters because word shingling treats each word as one unit; the split must
    drop spaces/punctuation so reformatting noise cannot perturb the shingle set.
    """
    assert tokenizer.word_tokens("the quick brown") == ["the", "quick", "brown"]


def test_word_tokens_splits_punctuation_and_symbols(tokenizer: Tokenizer) -> None:
    """Punctuation, symbols, and URL noise fragment into word runs only.

    Matters because crawled text is full of punctuation; tokenization must yield
    a stable word sequence so near-duplicates that differ only in punctuation
    still shingle alike.
    """
    assert tokenizer.word_tokens("e-mail: a@b.com, ok?") == [
        "e",
        "mail",
        "a",
        "b",
        "com",
        "ok",
    ]


def test_char_tokens_includes_spaces(tokenizer: Tokenizer) -> None:
    """char_tokens returns every character, spaces included.

    Matters because character shingling relies on the single inter-word spaces
    (kept by normalization) as boundary signal; dropping them would merge
    adjacent words and blur character-shingle discrimination.
    """
    assert tokenizer.char_tokens("a b") == ["a", " ", "b"]


def test_char_tokens_is_full_character_decomposition(tokenizer: Tokenizer) -> None:
    """char_tokens of a string equals list(string).

    Matters because char-mode k-grams must see exactly the normalized characters,
    in order, with nothing collapsed or reordered.
    """
    text = "Hi! 99"
    assert tokenizer.char_tokens(text) == list(text)


def test_ngrams_count_for_long_enough_tokens(tokenizer: Tokenizer) -> None:
    """ngrams over n tokens with window k yields n - k + 1 grams.

    Matters because the count of shingles a document contributes directly
    determines its Jaccard overlap denominator; an off-by-one here biases every
    similarity estimate.
    """
    grams = Tokenizer.ngrams(["a", "b", "c", "d"], 2)
    assert len(grams) == 3  # 4 - 2 + 1
    assert grams == ["a\x1fb", "b\x1fc", "c\x1fd"]


def test_ngrams_unigram_is_identity_per_token(tokenizer: Tokenizer) -> None:
    """With k=1 every token becomes its own gram.

    Matters because unigram shingling is a valid config; each token must map to
    exactly one shingle so the bag of tokens is faithfully represented.
    """
    assert Tokenizer.ngrams(["a", "b", "c"], 1) == ["a", "b", "c"]


def test_ngrams_degenerate_short_nonempty_returns_one_gram(tokenizer: Tokenizer) -> None:
    """Tokens shorter than k collapse to ONE gram (the whole list joined).

    Matters because a short document must still emit at least one shingle;
    otherwise it would be invisible to the deduplicator and could never be
    matched against anything (degenerate input).
    """
    grams = Tokenizer.ngrams(["only", "two"], 5)
    assert grams == ["only\x1ftwo"]
    assert len(grams) == 1


def test_ngrams_degenerate_empty_returns_empty(tokenizer: Tokenizer) -> None:
    """An empty token list yields NO grams even though it is shorter than k.

    Matters because an empty document carries no content and must produce an
    empty shingle set, not a spurious empty-string shingle that would falsely
    match other empty documents (degenerate input).
    """
    assert Tokenizer.ngrams([], 3) == []


def test_ngrams_k_less_than_one_raises(tokenizer: Tokenizer) -> None:
    """ngrams with k < 1 raises ValueError.

    Matters because a non-positive window is meaningless and signals an internal
    misconfiguration; failing loudly prevents silently producing a garbage
    (e.g. empty or infinite) shingle stream.
    """
    with pytest.raises(ValueError):
        Tokenizer.ngrams(["a", "b"], 0)
    with pytest.raises(ValueError):
        Tokenizer.ngrams(["a", "b"], -3)


def test_tokens_dispatches_word_mode(tokenizer: Tokenizer) -> None:
    """tokens(text, 'word') matches word_tokens(text).

    Matters because the shingler calls tokens() with a mode string; dispatch must
    route 'word' to word tokenization or the configured mode would be ignored.
    """
    assert tokenizer.tokens("a b c", "word") == tokenizer.word_tokens("a b c")
    assert tokenizer.tokens("a b c", "word") == ["a", "b", "c"]


def test_tokens_dispatches_char_mode(tokenizer: Tokenizer) -> None:
    """tokens(text, 'char') matches char_tokens(text).

    Matters because CJK and char-shingle configs depend on 'char' routing to the
    character splitter; misrouting would silently change the shingle universe.
    """
    assert tokenizer.tokens("a b", "char") == tokenizer.char_tokens("a b")
    assert tokenizer.tokens("a b", "char") == ["a", " ", "b"]


def test_tokens_unknown_mode_raises(tokenizer: Tokenizer) -> None:
    """tokens() with an unrecognised mode raises ValueError.

    Matters because an unknown mode means the config contract was violated;
    raising stops the pipeline from defaulting to the wrong tokenization and
    corrupting every signature.
    """
    with pytest.raises(ValueError):
        tokenizer.tokens("a b c", "sentence")


def test_pathological_empty_string_yields_no_tokens(tokenizer: Tokenizer) -> None:
    """The empty string produces empty word AND char token lists.

    Matters because empty records must flow through tokenization without error
    and contribute no shingles (pathological input).
    """
    assert tokenizer.word_tokens("") == []
    assert tokenizer.char_tokens("") == []


def test_pathological_punctuation_only_yields_no_word_tokens(
    tokenizer: Tokenizer,
) -> None:
    """A punctuation-only string has zero word tokens.

    Matters because content-free punctuation runs must not masquerade as words;
    otherwise documents containing only symbols could spuriously match
    (pathological input).
    """
    assert tokenizer.word_tokens("!!! ???") == []
    # Char tokenization, by contrast, still sees the raw characters.
    assert tokenizer.char_tokens("!!! ???") == list("!!! ???")
