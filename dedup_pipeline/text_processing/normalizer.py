"""Text normalization: the first transform applied to every document.

Normalization collapses superficial differences (case, Unicode form, HTML
markup, whitespace) so that two documents that are "the same" to a human map
to highly overlapping shingle sets. Without it, ``"Hello&nbsp;World"`` and
``"hello world"`` would look like different documents and slip past the
deduplicator [Lee et al. 2022].

Responsibility:
    * Provide :class:`TextNormalizer`, which applies a fixed, ordered cleaning
      pipeline and exposes script detection used to choose a shingling mode.

Inputs:
    * Raw document text (``str``), possibly containing HTML, mixed Unicode
      forms, and irregular whitespace.

Outputs:
    * A cleaned ``str`` and a boolean CJK-script verdict.
"""

from __future__ import annotations

import re
import unicodedata
from html.parser import HTMLParser

# Whitespace runs (including Unicode spaces such as U+00A0) collapse to a
# single ASCII space. Compiled once at import for speed.
_WHITESPACE_RUN = re.compile(r"\s+")

# Unicode code-point ranges that count as CJK for script detection. These are
# the blocks where whitespace word-tokenization is meaningless because the
# script is written without spaces between words.
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF),  # Hangul Syllables
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
)


class _HTMLTextExtractor(HTMLParser):
    """Collect the text content of an HTML fragment, discarding tags.

    Subclasses :class:`html.parser.HTMLParser` (standard library, no external
    dependency). ``convert_charrefs=True`` makes the parser resolve entities
    like ``&amp;`` and ``&#65;`` into plain text automatically, so
    :meth:`handle_data` receives already-decoded strings.

    Thread-safety:
        Not thread-safe. Create one instance per call (as
        :func:`TextNormalizer.strip_html` does); never share an instance
        across threads, because the parser keeps mutable feed state.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        """Accumulate a run of character data between tags.

        Args:
            data: A decoded text chunk emitted by the parser.
        """
        self._chunks.append(data)

    def get_text(self) -> str:
        """Return all accumulated text concatenated in document order.

        Returns:
            The tag-free text content seen so far.
        """
        return "".join(self._chunks)


class TextNormalizer:
    """Apply a deterministic cleaning pipeline to raw document text.

    The transform order is fixed and significant (it matches Stage 2 of
    :class:`~dedup_pipeline.pipeline.pipeline.DedupPipeline`):

        1. Unicode **NFKC** normalization — unify compatibility variants
           (ligatures, full-width forms, non-breaking spaces).
        2. **Lowercase** — fold case so ``"The"`` and ``"the"`` shingle alike.
        3. **Strip HTML** — remove tags, keep text and resolved entities.
        4. **Collapse whitespace** — turn any run of whitespace into one space.
        5. **Strip** leading/trailing whitespace.

    Thread-safety:
        Instances are stateless after construction (they hold only the
        immutable CJK ratio threshold) and create a fresh HTML parser per
        call, so a single :class:`TextNormalizer` may be shared across threads
        and processes without locking.

    Args:
        cjk_ratio_threshold: Fraction of CJK characters at or above which
            :meth:`is_cjk` returns ``True``. Sourced from
            :attr:`~dedup_pipeline.config.PipelineConfig.cjk_ratio_threshold`.

    Example:
        >>> norm = TextNormalizer(cjk_ratio_threshold=0.2)
        >>> norm.normalize("  <p>Hello&nbsp;<b>World</b></p>  ")
        'hello world'
    """

    def __init__(self, cjk_ratio_threshold: float) -> None:
        self._cjk_ratio_threshold = cjk_ratio_threshold

    def strip_html(self, text: str) -> str:
        """Remove HTML tags, returning only text content with entities decoded.

        Args:
            text: A string that may contain HTML markup.

        Returns:
            The text with all tags removed. Plain (non-HTML) input is returned
            essentially unchanged because the parser treats it as character
            data.

        Example:
            >>> TextNormalizer(0.2).strip_html("<a href='x'>link</a> &amp; more")
            'link & more'
        """
        parser = _HTMLTextExtractor()
        # html.parser is lenient: malformed markup is treated as data rather
        # than raising, so no try/except is needed here. close() flushes any
        # buffered trailing text.
        parser.feed(text)
        parser.close()
        return parser.get_text()

    def normalize(self, text: str) -> str:
        """Run the full five-step cleaning pipeline.

        Args:
            text: Raw document text.

        Returns:
            The normalized text. An all-whitespace or empty input yields an
            empty string.

        Example:
            >>> TextNormalizer(0.2).normalize("ＦＵＬＬ\twidth\n\n  TEXT")
            'full width text'
        """
        # Step 1: NFKC unifies compatibility characters (e.g. full-width 'Ａ'
        # -> 'A', the ligature 'ﬁ' -> 'fi', NBSP U+00A0 -> ' ').
        text = unicodedata.normalize("NFKC", text)
        # Step 2: case fold before HTML stripping so tag names lowercase too.
        text = text.lower()
        # Step 3: drop markup; keep human-visible text.
        text = self.strip_html(text)
        # Step 4: any whitespace run (tabs, newlines, repeated spaces) -> ' '.
        text = _WHITESPACE_RUN.sub(" ", text)
        # Step 5: remove the single leading/trailing space the collapse may add.
        return text.strip()

    def cjk_ratio(self, text: str) -> float:
        """Compute the fraction of non-space characters that are CJK.

        Whitespace is excluded from the denominator so that mostly-CJK text
        padded with spaces is still detected as CJK.

        Args:
            text: The text to inspect (normalized or raw).

        Returns:
            A value in ``[0.0, 1.0]``; ``0.0`` for empty/whitespace-only input.

        Example:
            >>> round(TextNormalizer(0.2).cjk_ratio("hello 世界"), 3)
            0.286
        """
        non_space = [ch for ch in text if not ch.isspace()]
        if not non_space:
            return 0.0
        cjk = sum(1 for ch in non_space if self._is_cjk_char(ch))
        return cjk / len(non_space)

    def is_cjk(self, text: str) -> bool:
        """Decide whether to treat ``text`` as a CJK document.

        When ``True``, the pipeline forces character-level shingling because
        whitespace word tokenization does not segment CJK words.

        Args:
            text: The text to classify.

        Returns:
            ``True`` if the CJK character ratio meets the configured threshold.

        Example:
            >>> TextNormalizer(0.2).is_cjk("これは日本語のテキストです")
            True
        """
        return self.cjk_ratio(text) >= self._cjk_ratio_threshold

    @staticmethod
    def _is_cjk_char(char: str) -> bool:
        """Return whether a single character falls in a CJK Unicode block.

        Args:
            char: A one-character string.

        Returns:
            ``True`` if the code point lies in any range of :data:`_CJK_RANGES`.
        """
        code_point = ord(char)
        return any(low <= code_point <= high for low, high in _CJK_RANGES)
