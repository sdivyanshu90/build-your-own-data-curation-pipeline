"""Shingling: convert normalized text into a set of integer shingle ids.

A *shingle* is a contiguous k-gram of tokens. Representing a document as the
*set* of its shingles is what lets Jaccard similarity capture content overlap
in an order-insensitive way [Broder 1997]. Each shingle string is hashed to a
64-bit integer so downstream MinHash operates on compact integers instead of
strings.

Responsibility:
    * Provide :class:`Shingler`, which maps text -> ``set[int]`` using the
      configured k and mode, with automatic CJK fallback to character shingling.

Inputs:
    * Normalized text (``str``).

Outputs:
    * A ``set[int]`` of 64-bit shingle ids (empty only for empty input).
"""

from __future__ import annotations

from dedup_pipeline.minhash.hash_functions import stable_hash64
from dedup_pipeline.text_processing.normalizer import TextNormalizer
from dedup_pipeline.text_processing.tokenizer import Tokenizer


class Shingler:
    """Generate integer shingle sets from text.

    The shingle mode (``"char"`` or ``"word"``) and size ``k`` come from
    configuration. When word shingling is requested but the text is detected as
    CJK, the shingler transparently falls back to character shingling, because
    whitespace word tokenization does not segment CJK scripts.

    Thread-safety:
        Stateless after construction (it holds only immutable parameters and
        shared, stateless helper objects). Safe to share across threads and
        processes.

    Args:
        shingle_size: The k-gram size ``k`` (from
            :attr:`~dedup_pipeline.config.PipelineConfig.shingle_size`).
        shingle_mode: ``"char"`` or ``"word"`` (from
            :attr:`~dedup_pipeline.config.PipelineConfig.shingle_mode`).
        seed: Seed mixed into the shingle content hash (from
            :attr:`~dedup_pipeline.config.PipelineConfig.random_seed`).
        tokenizer: The shared :class:`Tokenizer`.
        normalizer: The shared :class:`TextNormalizer`, used only for its CJK
            detection.

    Example:
        >>> from dedup_pipeline.text_processing.tokenizer import Tokenizer
        >>> from dedup_pipeline.text_processing.normalizer import TextNormalizer
        >>> sh = Shingler(3, "word", 42, Tokenizer(), TextNormalizer(0.2))
        >>> s = sh.shingle_text("the quick brown fox jumps")
        >>> len(s)  # 5 words -> 3 word-trigrams
        3
    """

    def __init__(
        self,
        shingle_size: int,
        shingle_mode: str,
        seed: int,
        tokenizer: Tokenizer,
        normalizer: TextNormalizer,
    ) -> None:
        self._k = shingle_size
        self._mode = shingle_mode
        self._seed = seed
        self._tokenizer = tokenizer
        self._normalizer = normalizer

    def _resolve_mode(self, text: str) -> str:
        """Choose the effective shingle mode for a given text.

        Args:
            text: Normalized text.

        Returns:
            ``"char"`` if word shingling was requested but the text is CJK,
            otherwise the configured mode.
        """
        if self._mode == "word" and self._normalizer.is_cjk(text):
            # CJK fallback: word tokenization is meaningless without spaces.
            return "char"
        return self._mode

    def shingle_text(self, text: str) -> set[int]:
        """Convert one document's text into a set of integer shingle ids.

        Args:
            text: Normalized document text.

        Returns:
            A ``set[int]`` of 64-bit shingle ids. Empty input yields an empty
            set; the set is naturally de-duplicated, so a document repeating the
            same k-gram contributes that shingle only once (this is what makes
            Jaccard a *set* similarity).

        Example:
            >>> sh = Shingler(2, "char", 42, Tokenizer(), TextNormalizer(0.2))
            >>> sorted_ids = sorted(sh.shingle_text("aaa"))
            >>> len(sorted_ids)  # bigrams of "aaa" are {"aa", "aa"} -> 1 unique
            1
        """
        mode = self._resolve_mode(text)
        tokens = self._tokenizer.tokens(text, mode)
        grams = Tokenizer.ngrams(tokens, self._k)
        # set comprehension de-duplicates repeated shingles automatically.
        return {stable_hash64(gram, self._seed) for gram in grams}

    def shingle_batch(self, texts: list[str]) -> list[set[int]]:
        """Shingle a batch of documents.

        Args:
            texts: Normalized texts, one per document.

        Returns:
            A list of shingle sets aligned positionally with ``texts``.

        Example:
            >>> sh = Shingler(2, "word", 42, Tokenizer(), TextNormalizer(0.2))
            >>> out = sh.shingle_batch(["a b c", "x y"])
            >>> [len(s) for s in out]
            [2, 1]
        """
        return [self.shingle_text(text) for text in texts]
