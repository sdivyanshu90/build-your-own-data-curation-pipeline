"""dedup_pipeline: production MinHash/LSH near-duplicate detection and removal.

This package implements the canonical scalable deduplication pipeline used to
clean LLM pretraining corpora: shingling -> MinHash signatures -> LSH banding ->
candidate pairs -> Union-Find clustering -> representative selection
[Broder 1997; Indyk & Motwani 1998; Leskovec et al. 2014; Lee et al. 2022].

Public API:
    * :class:`~dedup_pipeline.config.PipelineConfig` — all tunable parameters.
    * :class:`~dedup_pipeline.pipeline.pipeline.DedupPipeline` — the orchestrator.
    * :class:`~dedup_pipeline.pipeline.reader.Document` — the document record.
    * :class:`~dedup_pipeline.minhash.minhash.MinHasher` — signatures.
    * The exception hierarchy rooted at
      :class:`~dedup_pipeline.exceptions.DedupError`.
"""

from __future__ import annotations

from dedup_pipeline.config import PipelineConfig
from dedup_pipeline.exceptions import (
    CheckpointError,
    ConfigError,
    DedupError,
    EvaluationError,
    HashingError,
    IOError,  # noqa: A004 - intentional domain-specific exception name
)
from dedup_pipeline.minhash.minhash import MinHasher
from dedup_pipeline.pipeline.pipeline import DedupPipeline
from dedup_pipeline.pipeline.reader import Document

__version__ = "0.1.0"

__all__ = [
    "CheckpointError",
    "ConfigError",
    "DedupError",
    "DedupPipeline",
    "Document",
    "EvaluationError",
    "HashingError",
    "IOError",
    "MinHasher",
    "PipelineConfig",
    "__version__",
]
