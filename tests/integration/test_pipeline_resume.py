"""Integration test: checkpoint-based resume.

Verifies that interrupting a run after stage 5 and resuming produces byte-for-byte
identical output while skipping the already-completed early stages.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pytest

from dedup_pipeline.config import PipelineConfig
from dedup_pipeline.evaluation.synthetic_injector import inject_exact_duplicates
from dedup_pipeline.pipeline.pipeline import DedupPipeline

# Checkpoint artifacts written after stage 5 (the ones a post-stage-5 crash loses).
_POST_STAGE5_CHECKPOINTS = ("candidate_pairs", "clusters", "keep_indices")


def _corpus(seed: int = 5) -> list[dict[str, str]]:
    """A small corpus with a few exact duplicates so clustering is non-trivial."""
    rng = random.Random(seed)
    base = [
        {"id": f"u{i}", "text": " ".join(f"w{rng.randint(0, 500)}" for _ in range(40))}
        for i in range(60)
    ]
    records, _ = inject_exact_duplicates(base, num_pairs=8, seed=seed)
    return records


@pytest.fixture
def resume_config(tmp_path: Path) -> PipelineConfig:
    """A deterministic config with checkpointing enabled."""
    return PipelineConfig(
        num_hash_functions=64,
        lsh_bands=16,
        lsh_rows=4,
        shingle_mode="word",
        shingle_size=3,
        jaccard_threshold=0.7,
        random_seed=42,
        checkpoint_dir=tmp_path / "ckpt",
    )


@pytest.mark.integration
def test_resume_is_byte_identical_and_skips_early_stages(
    resume_config: PipelineConfig, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A resumed run reproduces the output exactly and skips stages 1-5.

    Matters because resumability is worthless if it changes results or silently
    recomputes (and possibly diverges on) completed stages.
    """
    records = _corpus()

    # 1. Full run to completion; this writes every stage checkpoint.
    dest_full = tmp_path / "out_full.jsonl"
    DedupPipeline(resume_config).run(records, dest_full, resume=False)
    output_full = dest_full.read_bytes()
    assert resume_config.checkpoint_dir is not None
    ckpt_dir = resume_config.checkpoint_dir

    # 2. Simulate an interruption *after stage 5*: drop the later checkpoints,
    #    keeping documents/signatures/bucket_index (stages 1-5).
    for name in _POST_STAGE5_CHECKPOINTS:
        (ckpt_dir / f"{name}.pkl").unlink()
    assert (ckpt_dir / "documents.pkl").exists()
    assert (ckpt_dir / "signatures.pkl").exists()
    assert (ckpt_dir / "bucket_index.pkl").exists()

    # 3. Resume into a fresh destination, capturing the pipeline's logs.
    dest_resume = tmp_path / "out_resume.jsonl"
    with caplog.at_level(logging.INFO, logger="dedup_pipeline.pipeline.pipeline"):
        DedupPipeline(resume_config).run(records, dest_resume, resume=True)
    output_resume = dest_resume.read_bytes()

    # Output must be byte-for-byte identical.
    assert output_resume == output_full

    # Stages 1-5 must have been skipped (loaded from checkpoint), verified via logs.
    log_text = caplog.text
    assert "skipping stages stream_documents, normalize_batch" in log_text
    assert "skipping stages shingle_batch, compute_signatures" in log_text
    assert "skipping stage build_bucket_index" in log_text


@pytest.mark.integration
def test_resume_without_checkpoints_recomputes(resume_config: PipelineConfig, tmp_path: Path) -> None:
    """Resuming with no checkpoints present simply runs everything (edge case).

    Matters because a first-ever run with resume=True must not fail just because
    nothing has been checkpointed yet.
    """
    records = _corpus(seed=9)
    dest = tmp_path / "out.jsonl"
    # Fresh checkpoint dir (created empty by the config fixture); resume=True.
    stats = DedupPipeline(resume_config).run(records, dest, resume=True)
    assert stats["output_count"] == stats["input_count"] - 8
