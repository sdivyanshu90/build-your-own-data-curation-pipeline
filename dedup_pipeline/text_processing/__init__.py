"""Text preprocessing: normalization, tokenization, and shingling."""

from __future__ import annotations

from dedup_pipeline.text_processing.normalizer import TextNormalizer
from dedup_pipeline.text_processing.shingler import Shingler
from dedup_pipeline.text_processing.tokenizer import Tokenizer

__all__ = ["Shingler", "TextNormalizer", "Tokenizer"]
