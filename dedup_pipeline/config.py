"""Central configuration for the deduplication pipeline.

This module defines :class:`PipelineConfig`, a ``pydantic-settings``
``BaseSettings`` model that holds *every* tunable constant used anywhere in
the package. The hard rule for this codebase is: **no magic numbers in
business logic** — if a number influences behaviour, it lives here as a named,
documented field.

Responsibility:
    * Declare, validate, and serialise all pipeline parameters.

Inputs:
    * Keyword arguments, environment variables (prefixed ``DEDUP_``), or a
      JSON/TOML file loaded by the caller.

Outputs:
    * A validated, immutable-by-convention configuration object consumed by
      every stage.

Notes:
    The Mersenne prime ``2**61 - 1`` used by the universal hash family is a
    *mathematical* constant (the largest prime that fits in 61 bits), not a
    tunable knob, so it is defined as a named module constant in
    :mod:`dedup_pipeline.minhash.hash_functions` rather than as a config field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dedup_pipeline.exceptions import ConfigError


class PipelineConfig(BaseSettings):
    """Validated configuration for a deduplication run.

    All numeric and categorical knobs live here. Construct it directly with
    keyword arguments, or let it read environment variables prefixed with
    ``DEDUP_`` (for example ``DEDUP_NUM_HASH_FUNCTIONS=256``).

    The signature length invariant ``num_hash_functions == lsh_bands *
    lsh_rows`` is enforced at construction time; violating it raises
    :class:`~dedup_pipeline.exceptions.ConfigError`.

    Thread-safety:
        Instances are treated as read-only after construction. The pipeline
        never mutates a config, so a single instance may be shared freely
        across threads and processes. (Pydantic models are not frozen by
        default; do not mutate fields post-construction if you share them.)

    Example:
        >>> cfg = PipelineConfig(num_hash_functions=128, lsh_bands=16, lsh_rows=8)
        >>> cfg.jaccard_threshold
        0.8
        >>> PipelineConfig(num_hash_functions=128, lsh_bands=16, lsh_rows=7)
        Traceback (most recent call last):
        ...
        dedup_pipeline.exceptions.ConfigError: num_hash_functions (128) must ...
    """

    model_config = SettingsConfigDict(
        env_prefix="DEDUP_",
        extra="forbid",
        validate_assignment=True,
    )

    # ----- Shingling -------------------------------------------------------
    shingle_size: int = Field(
        default=5,
        ge=1,
        description=(
            "Character (or word) n-gram size k used to convert a document into "
            "a set of shingles. Default 5 is the canonical choice for character "
            "shingling of English prose [Leskovec et al. 2014]: large enough that "
            "random 5-grams rarely collide across unrelated documents, small "
            "enough to stay robust to local edits. Raise it for code/markup, "
            "lower it for very short texts."
        ),
    )
    shingle_mode: Literal["char", "word"] = Field(
        default="char",
        description=(
            "Shingling strategy. 'char' n-grams are tokenizer-free and robust for "
            "multilingual/CJK text; 'word' n-grams are more semantic but sensitive "
            "to stopwords and tokenization. Default 'char' is the safest general "
            "choice."
        ),
    )
    cjk_ratio_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of CJK (Chinese/Japanese/Korean) characters above which a "
            "document is treated as CJK and forced to character-level shingling, "
            "because whitespace word tokenization is meaningless for unsegmented "
            "scripts. Default 0.2."
        ),
    )

    # ----- MinHash ---------------------------------------------------------
    num_hash_functions: int = Field(
        default=128,
        ge=1,
        description=(
            "Number n of MinHash hash functions = signature length. The estimator "
            "standard error is ~1/sqrt(n), so 128 gives ~0.044 absolute error on a "
            "Jaccard estimate — a good accuracy/memory trade-off [Broder 1997]. "
            "Must equal lsh_bands * lsh_rows."
        ),
    )
    use_numba: bool = Field(
        default=False,
        description=(
            "If True, use the Numba-JIT inner loop for signature computation. "
            "Default False keeps a pure-NumPy path that needs no compilation and "
            "is fully deterministic; enable for large corpora where the one-time "
            "JIT cost is amortised."
        ),
    )

    # ----- LSH banding -----------------------------------------------------
    lsh_bands: int = Field(
        default=16,
        ge=1,
        description=(
            "Number b of LSH bands. With r rows per band, two documents become a "
            "candidate pair with probability 1-(1-s^r)^b at Jaccard s [Indyk & "
            "Motwani 1998]. More bands -> higher recall, lower precision. Default "
            "16 (with 8 rows) places the S-curve knee near Jaccard 0.69."
        ),
    )
    lsh_rows: int = Field(
        default=8,
        ge=1,
        description=(
            "Number r of rows per band. num_hash_functions must equal "
            "lsh_bands * lsh_rows. Larger r sharpens the S-curve (steeper "
            "precision/recall transition). Default 8."
        ),
    )

    # ----- Similarity / verification --------------------------------------
    jaccard_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum estimated Jaccard similarity for two documents to be judged "
            "duplicates. 0.8 is a common near-duplicate threshold for web text "
            "[Lee et al. 2022]; lower it to catch looser paraphrases at the cost "
            "of precision."
        ),
    )
    high_precision_mode: bool = Field(
        default=False,
        description=(
            "If True, every LSH candidate pair is re-verified by re-estimating "
            "Jaccard from the full signatures and dropping pairs below "
            "jaccard_threshold. Default False trades a little precision for speed; "
            "enable when false positives are costly."
        ),
    )

    # ----- Representative selection ---------------------------------------
    representative_strategy: Literal["longest", "first", "random"] = Field(
        default="longest",
        description=(
            "Which member of a duplicate cluster to keep. 'longest' keeps the most "
            "information (default, recommended for pretraining corpora); 'first' "
            "keeps the earliest-seen (stable, order-dependent); 'random' samples "
            "uniformly using random_seed."
        ),
    )

    # ----- Batching / IO ---------------------------------------------------
    batch_size: int = Field(
        default=10_000,
        ge=1,
        description=(
            "Number of documents processed per vectorised batch. Default 10000 "
            "balances NumPy throughput against peak memory; lower it on "
            "memory-constrained machines, raise it for better vectorisation."
        ),
    )
    random_seed: int = Field(
        default=42,
        ge=0,
        description=(
            "Seed for all hash-family coefficients and any random representative "
            "selection. Fixing it makes the entire pipeline deterministic and "
            "reproducible. Default 42."
        ),
    )
    checkpoint_dir: Path | None = Field(
        default=None,
        description=(
            "Directory where per-stage checkpoints are written for resumable runs. "
            "Default None disables checkpointing (a single in-memory run)."
        ),
    )
    output_format: Literal["jsonl", "parquet"] = Field(
        default="jsonl",
        description=(
            "Output container for the deduplicated corpus. 'jsonl' is "
            "human-readable and append-friendly; 'parquet' is columnar, "
            "compressed, and ~3-5x smaller. Default 'jsonl'."
        ),
    )
    text_field: str = Field(
        default="text",
        description=(
            "Record key holding the document text in JSONL/Parquet/HuggingFace "
            "sources. Default 'text' matches most HF text datasets (e.g. AG News)."
        ),
    )
    id_field: str = Field(
        default="id",
        description=(
            "Record key holding a stable document id. If absent in a record, the "
            "reader synthesises an id from the running index. Default 'id'."
        ),
    )

    # ----- Bloom-filter candidate-pair de-duplication ---------------------
    use_bloom_filter: bool = Field(
        default=True,
        description=(
            "If True, candidate-pair enumeration uses a Bloom filter to suppress "
            "re-emitting the same pair found in multiple bands, bounding memory "
            "for very large candidate sets. Default True."
        ),
    )
    bloom_expected_pairs: int = Field(
        default=1_000_000,
        ge=1,
        description=(
            "Expected number of distinct candidate pairs, used to size the Bloom "
            "filter bit array. Default 1e6; set near your true candidate volume so "
            "the realised false-positive rate matches bloom_false_positive_rate."
        ),
    )
    bloom_false_positive_rate: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description=(
            "Target Bloom-filter false-positive probability. A false positive "
            "merely *drops* a duplicate pair (a recall cost), never a false merge, "
            "so 0.01 is safe. Lower it to raise recall at the cost of memory."
        ),
    )

    # ----- Clustering ------------------------------------------------------
    min_cluster_size: int = Field(
        default=2,
        ge=2,
        description=(
            "Smallest connected component considered a duplicate group. Singletons "
            "(size 1) are unique documents and are always kept. Default 2."
        ),
    )

    @model_validator(mode="after")
    def _check_signature_invariant(self) -> PipelineConfig:
        """Enforce ``num_hash_functions == lsh_bands * lsh_rows``.

        Returns:
            The validated config instance (unchanged).

        Raises:
            ConfigError: If the signature length does not factor exactly into
                ``lsh_bands`` bands of ``lsh_rows`` rows. A mismatch would leave
                trailing hash values unused or over-index the signature, silently
                corrupting LSH bucketing.
        """
        expected = self.lsh_bands * self.lsh_rows
        if self.num_hash_functions != expected:
            raise ConfigError(
                f"num_hash_functions ({self.num_hash_functions}) must equal "
                f"lsh_bands ({self.lsh_bands}) * lsh_rows ({self.lsh_rows}) = "
                f"{expected}"
            )
        return self

    def to_serializable_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the config.

        Paths become strings and enums become their values, so the result can
        be embedded directly in the statistics JSON written by stage 10.

        Returns:
            A plain ``dict`` safe to pass to :func:`json.dumps`.

        Example:
            >>> cfg = PipelineConfig()
            >>> d = cfg.to_serializable_dict()
            >>> d["num_hash_functions"]
            128
        """
        return self.model_dump(mode="json")
