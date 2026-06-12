"""Pipeline: orchestration, IO, and checkpointing."""

from __future__ import annotations

from dedup_pipeline.pipeline.checkpointer import StageCheckpointer
from dedup_pipeline.pipeline.pipeline import DedupPipeline
from dedup_pipeline.pipeline.reader import Document, DocumentReader
from dedup_pipeline.pipeline.writer import DeduplicatedWriter

__all__ = [
    "DedupPipeline",
    "DeduplicatedWriter",
    "Document",
    "DocumentReader",
    "StageCheckpointer",
]
