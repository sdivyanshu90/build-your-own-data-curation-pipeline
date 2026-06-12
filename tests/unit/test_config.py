"""Unit tests for :class:`PipelineConfig`.

The config object is the single source of truth for every tunable constant in
the pipeline. If its defaults drift, its cross-field invariants stop being
enforced, or its serialisation breaks, then either every downstream stage runs
with silently wrong parameters or reproducibility/statistics output is lost.
These tests pin the documented defaults, the signature-length invariant, field
range validation, and JSON round-tripping so configuration mistakes fail loudly
at construction time rather than corrupting a multi-hour deduplication run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dedup_pipeline.config import PipelineConfig
from dedup_pipeline.exceptions import ConfigError


def test_defaults_match_documented_values() -> None:
    """The default config exposes exactly the documented constants.

    Matters because these defaults define the canonical MinHash/LSH operating
    point (128 hashes = 16 bands x 8 rows, Jaccard 0.8). Any unnoticed drift
    would change every run's recall/precision and break reproducibility against
    published baselines.
    """
    cfg = PipelineConfig()
    assert cfg.shingle_size == 5
    assert cfg.num_hash_functions == 128
    assert cfg.lsh_bands == 16
    assert cfg.lsh_rows == 8
    assert cfg.jaccard_threshold == 0.8
    assert cfg.batch_size == 10000
    assert cfg.random_seed == 42
    assert cfg.shingle_mode == "char"
    assert cfg.output_format == "jsonl"
    assert cfg.representative_strategy == "longest"


def test_default_satisfies_signature_invariant() -> None:
    """The default config already satisfies num_hash_functions == bands * rows.

    Matters because the out-of-the-box config must construct without error;
    16 * 8 == 128 is the load-bearing factorisation that makes LSH banding
    index the signature exactly, with no leftover or over-indexed hash values.
    """
    cfg = PipelineConfig()
    assert cfg.num_hash_functions == cfg.lsh_bands * cfg.lsh_rows


def test_signature_invariant_violation_raises_config_error() -> None:
    """A num_hash_functions != bands * rows mismatch raises ConfigError.

    Matters because a mismatch would leave trailing hash values unused or
    over-index the signature, silently corrupting LSH bucketing. The
    model_validator raises ConfigError directly (pydantic propagates it), so
    callers can trap the precise failure mode rather than a generic
    ValidationError.
    """
    with pytest.raises(ConfigError):
        PipelineConfig(num_hash_functions=128, lsh_bands=16, lsh_rows=7)


def test_valid_custom_config_constructs() -> None:
    """A consistent custom factorisation (64 = 8 * 8) constructs cleanly.

    Matters because operators routinely retune the signature length and banding
    for different corpora; a self-consistent override must be accepted so the
    config is genuinely tunable, not frozen to the defaults.
    """
    cfg = PipelineConfig(num_hash_functions=64, lsh_bands=8, lsh_rows=8)
    assert cfg.num_hash_functions == 64
    assert cfg.lsh_bands == 8
    assert cfg.lsh_rows == 8


def test_to_serializable_dict_round_trips_through_json() -> None:
    """to_serializable_dict() returns a dict that survives json.dumps/loads.

    Matters because the config is embedded verbatim in the run's statistics
    JSON; if any field were a non-serialisable type the entire stats write would
    crash at the end of a long run, losing the provenance of the output corpus.
    """
    cfg = PipelineConfig()
    d = cfg.to_serializable_dict()
    assert isinstance(d, dict)
    encoded = json.dumps(d)
    decoded = json.loads(encoded)
    assert decoded["num_hash_functions"] == 128
    assert decoded["jaccard_threshold"] == 0.8


def test_to_serializable_dict_path_field_becomes_string_or_null(tmp_path: Path) -> None:
    """Path fields serialise to a JSON string (or null when unset).

    Matters because a raw pathlib.Path is not JSON-serialisable; the config must
    coerce checkpoint_dir to text so the provenance record is portable across
    machines and the stats file can always be written.
    """
    default_dict = PipelineConfig().to_serializable_dict()
    assert default_dict["checkpoint_dir"] is None

    cfg = PipelineConfig(checkpoint_dir=tmp_path)
    set_dict = cfg.to_serializable_dict()
    assert isinstance(set_dict["checkpoint_dir"], str)
    # The serialised form must round-trip through JSON without raising.
    json.dumps(set_dict)


def test_jaccard_threshold_above_one_raises_validation_error() -> None:
    """jaccard_threshold > 1 is rejected with a pydantic ValidationError.

    Matters because Jaccard similarity is bounded in [0, 1]; an out-of-range
    threshold is meaningless and would make the duplicate decision either never
    or always fire, silently destroying the deduplication result.
    """
    with pytest.raises(ValidationError):
        PipelineConfig(jaccard_threshold=1.5)


def test_jaccard_threshold_below_zero_raises_validation_error() -> None:
    """jaccard_threshold < 0 is rejected with a pydantic ValidationError.

    Matters because a negative similarity threshold is nonsensical and would
    classify every pair as a duplicate, collapsing the corpus. Range validation
    must catch this at construction time.
    """
    with pytest.raises(ValidationError):
        PipelineConfig(jaccard_threshold=-0.01)


def test_num_hash_functions_below_one_raises_validation_error() -> None:
    """num_hash_functions < 1 is rejected with a pydantic ValidationError.

    Matters because a zero-length MinHash signature carries no information and
    would make every document collide; the ge=1 bound guarantees a usable
    signature length.
    """
    with pytest.raises(ValidationError):
        PipelineConfig(num_hash_functions=0, lsh_bands=1, lsh_rows=1)


def test_bloom_false_positive_rate_must_be_open_interval() -> None:
    """bloom_false_positive_rate must lie strictly in (0, 1).

    Matters because a Bloom filter sized for a 0 or 1 false-positive rate is
    degenerate (infinite or zero bits); the open-interval bound keeps the filter
    well-defined so candidate-pair suppression behaves predictably. Both
    boundary values must be rejected.
    """
    with pytest.raises(ValidationError):
        PipelineConfig(bloom_false_positive_rate=0.0)
    with pytest.raises(ValidationError):
        PipelineConfig(bloom_false_positive_rate=1.0)


def test_checkpoint_dir_accepts_and_stores_path(tmp_path: Path) -> None:
    """checkpoint_dir given a Path is stored as a Path instance.

    Matters because resumable runs depend on the checkpoint directory being a
    usable filesystem path object; storing it as a Path lets every stage write
    and reload checkpoints without re-parsing strings.
    """
    cfg = PipelineConfig(checkpoint_dir=tmp_path)
    assert isinstance(cfg.checkpoint_dir, Path)
    assert cfg.checkpoint_dir == tmp_path


# ----- Pathological / edge cases -----------------------------------------


def test_invalid_shingle_mode_literal_raises_validation_error() -> None:
    """An out-of-set shingle_mode literal is rejected with ValidationError.

    Pathological case: a typo like "characters" instead of "char" must not be
    silently accepted. Matters because the shingling strategy is a closed enum;
    an unrecognised mode would have no defined behaviour and could fall through
    to a wrong tokenisation path, corrupting every signature.
    """
    with pytest.raises(ValidationError):
        PipelineConfig(shingle_mode="characters")


def test_min_cluster_size_below_two_raises_validation_error() -> None:
    """min_cluster_size < 2 is rejected with a pydantic ValidationError.

    Pathological case: a cluster of size 1 is a unique document, not a duplicate
    group. Matters because allowing size < 2 would treat singletons as
    duplicates and start dropping unique documents, the worst possible
    deduplication error.
    """
    with pytest.raises(ValidationError):
        PipelineConfig(min_cluster_size=1)


def test_extra_unknown_field_is_forbidden() -> None:
    """An unknown keyword argument is rejected (extra="forbid").

    Pathological case: a misspelled field name such as "jaccard_treshold" must
    fail loudly rather than be silently dropped. Matters because a typo'd knob
    that is ignored would leave the pipeline running on a stale default while the
    operator believes they retuned it.
    """
    with pytest.raises(ValidationError):
        PipelineConfig(jaccard_treshold=0.9)
