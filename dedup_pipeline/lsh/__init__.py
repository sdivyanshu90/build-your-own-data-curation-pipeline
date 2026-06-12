"""LSH: banding, bucket indexing, and candidate-pair generation."""

from __future__ import annotations

from dedup_pipeline.lsh.banding import (
    BandingScheme,
    candidate_probability,
    recommend_bands,
)
from dedup_pipeline.lsh.bucket_index import (
    BloomFilter,
    BucketIndex,
    build_bucket_index,
)
from dedup_pipeline.lsh.candidate_pairs import enumerate_candidate_pairs

__all__ = [
    "BandingScheme",
    "BloomFilter",
    "BucketIndex",
    "build_bucket_index",
    "candidate_probability",
    "enumerate_candidate_pairs",
    "recommend_bands",
]
