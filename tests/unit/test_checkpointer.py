"""Unit tests for :class:`StageCheckpointer`.

The checkpointer is the foundation of resumable runs: an interrupted multi-hour
deduplication must be able to restart at the first incomplete stage rather than
recomputing everything. For that to be safe, three properties must hold without
exception: (1) a checkpoint that *exists* is complete and loads back the exact
artifact that was saved (atomic, lossless round-trip); (2) the disabled mode is
a transparent no-op so the identical pipeline code runs with or without
checkpointing; and (3) a missing or corrupt checkpoint surfaces as a typed
:class:`CheckpointError` so a damaged resume fails loudly instead of feeding
garbage into later stages. These tests pin all three.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dedup_pipeline.exceptions import CheckpointError
from dedup_pipeline.pipeline.checkpointer import StageCheckpointer

# ----- Disabled mode (checkpoint_dir=None) -------------------------------


def test_disabled_checkpointer_reports_not_enabled() -> None:
    """With checkpoint_dir=None, enabled is False.

    Matters because the pipeline branches on `enabled` to decide whether to
    consult checkpoints at all; if a disabled checkpointer reported enabled it
    would try to load non-existent files and crash an ordinary in-memory run.
    """
    ck = StageCheckpointer(None)
    assert ck.enabled is False


def test_disabled_has_stage_always_false() -> None:
    """A disabled checkpointer reports has_stage False for any name.

    Matters because the resume logic gates recomputation on has_stage; a
    disabled checkpointer must never claim a stage is already done, otherwise
    the pipeline would skip real work and try to load an artifact that was
    never written.
    """
    ck = StageCheckpointer(None)
    assert ck.has_stage("anything") is False
    assert ck.has_stage("stage_1") is False


def test_disabled_save_is_silent_no_op() -> None:
    """save() on a disabled checkpointer does nothing and does not raise.

    Matters because the exact same pipeline code path calls save() after every
    stage; in disabled mode that call must be a harmless no-op so checkpointing
    can be turned off without touching the stage code.
    """
    ck = StageCheckpointer(None)
    # Must not raise even though there is nowhere to write.
    ck.save("stage_1", {"a": 1})
    assert ck.has_stage("stage_1") is False


def test_disabled_completed_stages_is_empty() -> None:
    """A disabled checkpointer reports an empty completed_stages set.

    Matters because resume picks the first stage not in completed_stages; a
    disabled run must report none complete so it always executes from the start.
    """
    ck = StageCheckpointer(None)
    assert ck.completed_stages() == set()


def test_disabled_delete_is_silent_no_op() -> None:
    """delete() on a disabled checkpointer does nothing and does not raise.

    Matters because cleanup paths call delete() unconditionally; in disabled
    mode it must be a safe no-op so teardown code need not special-case it.
    """
    ck = StageCheckpointer(None)
    ck.delete("stage_1")  # must not raise


# ----- Enabled: save / has_stage / load round-trip -----------------------


def test_save_then_has_stage_true(tmp_path: Path) -> None:
    """After save(), has_stage() returns True for that stage.

    Matters because resume detects completed work via has_stage; a saved
    checkpoint must be immediately visible so the pipeline knows it can skip
    recomputing that stage.
    """
    ck = StageCheckpointer(tmp_path)
    ck.save("stage_1", {"value": 1})
    assert ck.has_stage("stage_1") is True


def test_round_trip_dict_returns_exact_object(tmp_path: Path) -> None:
    """A saved dict loads back equal to the original.

    Matters because stage artifacts (e.g. id->index maps) must survive the
    pickle round-trip byte-for-byte; a lossy reload would silently corrupt every
    downstream stage that consumes the checkpoint.
    """
    ck = StageCheckpointer(tmp_path)
    obj = {"a": 1, "b": [2, 3], "c": {"nested": True}}
    ck.save("stage_dict", obj)
    assert ck.load("stage_dict") == obj


def test_round_trip_list_returns_exact_object(tmp_path: Path) -> None:
    """A saved list loads back equal to the original.

    Matters because many stages emit ordered lists (e.g. signatures per
    document); the reload must preserve both contents and order so resumed runs
    are identical to uninterrupted ones.
    """
    ck = StageCheckpointer(tmp_path)
    obj = [10, 20, 30, 40]
    ck.save("stage_list", obj)
    assert ck.load("stage_list") == obj


def test_round_trip_numpy_array_preserves_values(tmp_path: Path) -> None:
    """A saved NumPy array loads back element-equal via numpy.array_equal.

    Matters because MinHash signatures are stored as NumPy arrays; the
    checkpoint must preserve the exact integer values (not just shape), or a
    resumed run would band different signatures and produce a different dedup
    result.
    """
    ck = StageCheckpointer(tmp_path)
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64)
    ck.save("stage_arr", arr)
    loaded = ck.load("stage_arr")
    assert np.array_equal(loaded, arr)
    assert loaded.dtype == arr.dtype


def test_saving_twice_overwrites_and_load_returns_latest(tmp_path: Path) -> None:
    """Re-saving a stage overwrites the prior artifact; load returns the latest.

    Matters because a stage may be recomputed and re-checkpointed; the
    checkpointer must always reflect the most recent save so a resume never
    picks up a stale artifact from an earlier attempt.
    """
    ck = StageCheckpointer(tmp_path)
    ck.save("stage_x", {"version": 1})
    ck.save("stage_x", {"version": 2})
    assert ck.load("stage_x") == {"version": 2}


def test_delete_removes_checkpoint(tmp_path: Path) -> None:
    """After delete(), has_stage() reports the stage as absent.

    Matters because invalidating a stage (e.g. after a config change) relies on
    delete() truly removing the artifact; a lingering file would let a resume
    reuse an out-of-date checkpoint.
    """
    ck = StageCheckpointer(tmp_path)
    ck.save("stage_x", [1, 2, 3])
    assert ck.has_stage("stage_x") is True
    ck.delete("stage_x")
    assert ck.has_stage("stage_x") is False


def test_delete_missing_stage_is_silent(tmp_path: Path) -> None:
    """delete() of a never-saved stage does not raise.

    Matters because cleanup runs unconditionally; deleting a checkpoint that was
    never written must be tolerated so teardown logic stays branch-free.
    """
    ck = StageCheckpointer(tmp_path)
    ck.delete("never_saved")  # must not raise


def test_completed_stages_returns_set_of_saved_names(tmp_path: Path) -> None:
    """completed_stages() returns exactly the set of saved stage names.

    Matters because resume computes the first incomplete stage by set
    membership; the returned names must match the saved names (without the .pkl
    suffix) or the pipeline would either redo finished work or skip unfinished
    work.
    """
    ck = StageCheckpointer(tmp_path)
    ck.save("stage_a", 1)
    ck.save("stage_b", 2)
    ck.save("stage_c", 3)
    assert ck.completed_stages() == {"stage_a", "stage_b", "stage_c"}


def test_completed_stages_reflects_deletion(tmp_path: Path) -> None:
    """A deleted stage drops out of completed_stages().

    Matters because completed_stages() is the authoritative resume manifest;
    after invalidating a stage it must no longer appear, so the resume recomputes
    it.
    """
    ck = StageCheckpointer(tmp_path)
    ck.save("stage_a", 1)
    ck.save("stage_b", 2)
    ck.delete("stage_a")
    assert ck.completed_stages() == {"stage_b"}


# ----- Error handling: missing and corrupt checkpoints -------------------


def test_load_missing_stage_raises_checkpoint_error(tmp_path: Path) -> None:
    """load() of a stage that was never saved raises CheckpointError.

    Matters because a resume must fail loudly with a typed error if it expects a
    checkpoint that is not there, rather than returning None and letting a later
    stage operate on missing data.
    """
    ck = StageCheckpointer(tmp_path)
    with pytest.raises(CheckpointError):
        ck.load("does_not_exist")


def test_load_corrupt_file_raises_checkpoint_error(tmp_path: Path) -> None:
    """load() of a corrupt .pkl file raises CheckpointError, not a raw unpickle error.

    Pathological case: a checkpoint truncated by a crash mid-write (here
    simulated by writing garbage bytes to the .pkl path) must surface as the
    package's typed CheckpointError so the resume aborts cleanly instead of
    propagating an opaque UnpicklingError/EOFError from deep in the stage code.
    """
    ck = StageCheckpointer(tmp_path)
    corrupt_path = tmp_path / "stage_bad.pkl"
    corrupt_path.write_bytes(b"\x00\x01not a valid pickle stream\xff")
    assert ck.has_stage("stage_bad") is True  # the file exists...
    with pytest.raises(CheckpointError):
        ck.load("stage_bad")  # ...but it cannot be unpickled.


# ----- Pathological / edge cases -----------------------------------------


def test_save_empty_list_and_dict_round_trip(tmp_path: Path) -> None:
    """Empty containers round-trip and are reported as completed stages.

    Pathological case: a stage may legitimately produce an empty artifact (e.g.
    a corpus with no duplicate clusters yields an empty cluster list). Matters
    because an empty result is still a *completed* stage; treating it as missing
    would force a needless recomputation on resume. The falsy value must not be
    confused with "no checkpoint".
    """
    ck = StageCheckpointer(tmp_path)
    ck.save("empty_list", [])
    ck.save("empty_dict", {})
    assert ck.has_stage("empty_list") is True
    assert ck.has_stage("empty_dict") is True
    assert ck.load("empty_list") == []
    assert ck.load("empty_dict") == {}
    assert {"empty_list", "empty_dict"} <= ck.completed_stages()


def test_constructor_creates_nonexistent_nested_dir(tmp_path: Path) -> None:
    """Constructing with a deep, non-existent path creates the directory tree.

    Pathological case: the configured checkpoint_dir may point several levels
    below an existing root that has not been created yet. Matters because the
    pipeline must not require the operator to pre-create the directory; the
    checkpointer must materialise the full path (parents=True) so the first
    save() has somewhere to write.
    """
    nested = tmp_path / "a" / "b" / "c" / "checkpoints"
    assert not nested.exists()
    ck = StageCheckpointer(nested)
    assert nested.is_dir()
    ck.save("stage_1", {"ok": True})
    assert ck.load("stage_1") == {"ok": True}
