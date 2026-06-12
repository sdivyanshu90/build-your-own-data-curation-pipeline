"""Unit tests for :class:`TextNormalizer`.

Normalization is the first transform every document passes through; if it is
inconsistent, semantically identical documents produce different shingles and
escape deduplication. These tests pin the exact, ordered cleaning behaviour.
"""

from __future__ import annotations

import pytest

from dedup_pipeline.text_processing.normalizer import TextNormalizer


@pytest.fixture
def normalizer() -> TextNormalizer:
    """A normalizer with the default CJK threshold."""
    return TextNormalizer(cjk_ratio_threshold=0.2)


def test_nfkc_ligature(normalizer: TextNormalizer) -> None:
    """The 'fi' ligature must decompose to 'fi'.

    Matters because copies of a document may differ only in ligature vs. plain
    letters; without NFKC they would not shingle alike.
    """
    assert normalizer.normalize("ﬁle") == "file"


def test_nfkc_fullwidth(normalizer: TextNormalizer) -> None:
    """Full-width Latin characters must fold to ASCII.

    Matters because CJK-locale text often uses full-width ASCII; folding lets it
    match the same content typed in plain ASCII.
    """
    assert normalizer.normalize("ＦＵＬＬ") == "full"


def test_nfkc_combining_characters(normalizer: TextNormalizer) -> None:
    """A base letter + combining accent must compose to one code point.

    Matters because the same accented word can be encoded composed or decomposed;
    NFKC unifies them so they shingle identically.
    """
    decomposed = "é"  # 'e' + combining acute accent
    assert normalizer.normalize(decomposed) == "é"  # 'é'


def test_strip_simple_html(normalizer: TextNormalizer) -> None:
    """Simple tags are removed, inner text retained.

    Matters because boilerplate HTML wrappers must not make otherwise-identical
    articles look different.
    """
    assert normalizer.normalize("<p>Hello</p>") == "hello"


def test_strip_nested_html(normalizer: TextNormalizer) -> None:
    """Nested tags are fully removed.

    Matters because real web documents nest markup arbitrarily deep.
    """
    assert normalizer.normalize("<div><b><i>Hi</i></b></div>") == "hi"


def test_strip_html_with_attributes(normalizer: TextNormalizer) -> None:
    """Tag attributes are dropped along with the tag.

    Matters because attribute noise (classes, hrefs) is not document content.
    """
    assert normalizer.normalize('<a href="http://x.com" class="z">link</a>') == "link"


def test_malformed_html_does_not_crash(normalizer: TextNormalizer) -> None:
    """Malformed markup is tolerated, never raising.

    Matters because crawled HTML is frequently broken; one bad document must not
    abort a batch.
    """
    result = normalizer.normalize("<p>unclosed <b>bold</p> text")
    assert "bold" in result and "text" in result


def test_html_entities_decoded(normalizer: TextNormalizer) -> None:
    """HTML entities resolve to their characters.

    Matters because '&amp;' and '&' must be treated as the same content.
    """
    assert normalizer.normalize("a &amp; b") == "a & b"


def test_collapse_tabs_newlines_spaces(normalizer: TextNormalizer) -> None:
    """Any run of whitespace collapses to a single space.

    Matters because reformatting (line wraps, indentation) is a common near-dup
    difference that should be erased before shingling.
    """
    assert normalizer.normalize("a\t\tb\n\n\nc   d") == "a b c d"


def test_non_breaking_space_normalized(normalizer: TextNormalizer) -> None:
    """Non-breaking spaces are treated as whitespace.

    Matters because NBSP frequently appears in HTML and would otherwise create
    spurious shingle differences.
    """
    assert normalizer.normalize("a  b") == "a b"


def test_lowercasing(normalizer: TextNormalizer) -> None:
    """Case is folded.

    Matters because casing differences are not meaningful for deduplication.
    """
    assert normalizer.normalize("HeLLo WORLD") == "hello world"


def test_empty_string(normalizer: TextNormalizer) -> None:
    """Empty input yields empty output (pathological input).

    Matters because empty records must flow through without error.
    """
    assert normalizer.normalize("") == ""


def test_whitespace_only(normalizer: TextNormalizer) -> None:
    """Whitespace-only input collapses to empty (pathological input).

    Matters because such records carry no content and must not crash the cleaner.
    """
    assert normalizer.normalize("   \t\n  ") == ""


def test_punctuation_only(normalizer: TextNormalizer) -> None:
    """Punctuation-only input is preserved (minus case/whitespace).

    Matters because the normalizer must not silently delete non-letter content.
    """
    assert normalizer.normalize("!!! ??? ...") == "!!! ??? ..."


def test_numbers_only(normalizer: TextNormalizer) -> None:
    """Numeric-only input is preserved.

    Matters because numbers are valid content (tables, logs) and must survive.
    """
    assert normalizer.normalize("123 456 789") == "123 456 789"


def test_very_long_string(normalizer: TextNormalizer) -> None:
    """A 100,000-character string normalizes without error (pathological input).

    Matters because production documents can be very large; the cleaner must
    scale linearly and not blow up.
    """
    text = "A " * 50_000  # 100,000 characters
    result = normalizer.normalize(text)
    assert result == ("a " * 50_000).strip()


def test_cjk_passthrough(normalizer: TextNormalizer) -> None:
    """CJK text is detected as CJK and its characters survive normalization.

    Matters because CJK documents must route to character shingling rather than
    (meaningless) whitespace word tokenization.
    """
    text = "これは日本語のテキスト"
    assert normalizer.is_cjk(text)
    assert normalizer.normalize(text) == text


def test_rtl_text_preserved(normalizer: TextNormalizer) -> None:
    """Right-to-left (Arabic/Hebrew) text is preserved (non-ASCII input).

    Matters because the pipeline must be language-agnostic; RTL content must not
    be mangled or dropped.
    """
    arabic = "مرحبا بالعالم"
    hebrew = "שלום עולם"
    assert normalizer.normalize(arabic) == arabic
    assert normalizer.normalize(hebrew) == hebrew


def test_cjk_ratio_mixed(normalizer: TextNormalizer) -> None:
    """The CJK ratio counts only non-space characters.

    Matters because the routing decision (word vs. char shingling) depends on a
    correct ratio even when text is padded with spaces.
    """
    # 2 CJK chars out of 7 non-space chars -> ~0.286.
    assert normalizer.cjk_ratio("hello 世界") == pytest.approx(2 / 7, abs=1e-6)


def test_non_cjk_text_not_flagged(normalizer: TextNormalizer) -> None:
    """Latin text is not flagged as CJK.

    Matters because mis-flagging English as CJK would disable word shingling.
    """
    assert not normalizer.is_cjk("the quick brown fox")
