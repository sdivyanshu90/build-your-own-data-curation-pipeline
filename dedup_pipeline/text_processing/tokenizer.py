"""Tokenization: turn normalized text into the unit stream for shingling.

Shingling builds *k*-grams over a sequence of base units. This module produces
that base sequence in two modes:

    * **word** units — Unicode word tokens (``re`` ``\\w+`` runs), the natural
      unit for word n-gram shingling.
    * **char** units — individual characters, the natural unit for character
      n-gram shingling, which is tokenizer-free and robust for multilingual
      and CJK text [Leskovec et al. 2014].

Responsibility:
    * Provide :class:`Tokenizer` with deterministic word/char tokenization and
      a generic sliding-window n-gram helper used by the shingler.

Inputs:
    * Normalized text (``str``) from
      :class:`~dedup_pipeline.text_processing.normalizer.TextNormalizer`.

Outputs:
    * Lists of token strings and lists of joined n-gram strings.
"""

from __future__ import annotations

import re

# A word token is a maximal run of Unicode "word" characters (letters, digits,
# underscore). This keeps URLs/punctuation from fragmenting tokens while
# staying language-agnostic. Compiled once at import.
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Separator joined between units when forming an n-gram string. Chosen to be a
# character that cannot appear inside a word token, so the n-gram "a\x1fb"
# cannot be confused with a single token "a\x1fb".
_NGRAM_JOIN = "\x1f"  # ASCII Unit Separator control character


class Tokenizer:
    """Produce word/character token streams and n-grams from normalized text.

    Thread-safety:
        Stateless and immutable; the only field is the compiled regex shared at
        module level. A single :class:`Tokenizer` is safe to share across
        threads and processes.

    Example:
        >>> tok = Tokenizer()
        >>> tok.word_tokens("the quick brown fox")
        ['the', 'quick', 'brown', 'fox']
        >>> tok.char_tokens("abc")
        ['a', 'b', 'c']
    """

    def word_tokens(self, text: str) -> list[str]:
        """Split text into Unicode word tokens.

        Args:
            text: Normalized text.

        Returns:
            Word tokens in left-to-right order. Punctuation and whitespace are
            dropped; an empty or punctuation-only string yields ``[]``.

        Example:
            >>> Tokenizer().word_tokens("e-mail: a@b.com, ok?")
            ['e', 'mail', 'a', 'b', 'com', 'ok']
        """
        return _WORD_RE.findall(text)

    def char_tokens(self, text: str) -> list[str]:
        """Split text into individual characters.

        Whitespace characters are preserved because, after normalization, the
        single spaces between words carry boundary information that improves
        character-shingle discrimination.

        Args:
            text: Normalized text.

        Returns:
            A list of one-character strings (possibly empty).

        Example:
            >>> Tokenizer().char_tokens("a b")
            ['a', ' ', 'b']
        """
        return list(text)

    def tokens(self, text: str, mode: str) -> list[str]:
        """Dispatch to word or character tokenization by mode.

        Args:
            text: Normalized text.
            mode: Either ``"word"`` or ``"char"``.

        Returns:
            The token list for the requested mode.

        Raises:
            ValueError: If ``mode`` is not ``"word"`` or ``"char"``. This is an
                internal programming error (the config layer constrains the
                value to a Literal), so a plain ValueError is appropriate.

        Example:
            >>> Tokenizer().tokens("a b", "word")
            ['a', 'b']
        """
        if mode == "word":
            return self.word_tokens(text)
        if mode == "char":
            return self.char_tokens(text)
        raise ValueError(f"Unknown tokenization mode: {mode!r}")

    @staticmethod
    def ngrams(tokens: list[str], k: int) -> list[str]:
        """Build contiguous k-grams from a token list via a sliding window.

        Args:
            tokens: The base unit sequence (words or characters).
            k: The n-gram size (number of units per shingle).

        Returns:
            A list of ``len(tokens) - k + 1`` joined n-gram strings. If the
            token list is shorter than ``k``, the whole token list is returned
            as a single n-gram so that short documents still produce at least
            one shingle (otherwise they would be invisible to the deduplicator).

        Raises:
            ValueError: If ``k < 1``. The config layer guarantees ``k >= 1``,
                so this guards against internal misuse.

        Example:
            >>> Tokenizer.ngrams(["a", "b", "c", "d"], 2)
            ['a\\x1fb', 'b\\x1fc', 'c\\x1fd']
        """
        if k < 1:
            raise ValueError(f"n-gram size k must be >= 1, got {k}")
        if len(tokens) < k:
            # Degenerate-but-safe: a document with fewer than k units still gets
            # one shingle (its whole content) instead of an empty set.
            return [_NGRAM_JOIN.join(tokens)] if tokens else []
        return [
            _NGRAM_JOIN.join(tokens[i : i + k])
            for i in range(len(tokens) - k + 1)
        ]
