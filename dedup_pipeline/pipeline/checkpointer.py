"""Per-stage checkpointing for resumable pipeline runs.

Each pipeline stage may persist its output artifact so that an interrupted run
can resume at the first incomplete stage instead of recomputing everything.
Writes are **atomic** — the artifact is serialised to a temporary file and then
``os.replace``-d into place — so a checkpoint file that exists is guaranteed to
be complete and loadable, never half-written.

Responsibility:
    * Save/load/detect/delete named stage artifacts under ``checkpoint_dir``.

Inputs:
    * A stage name and a picklable artifact.

Outputs:
    * Persisted ``.pkl`` files and reload of their contents.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

from dedup_pipeline.exceptions import CheckpointError

logger = logging.getLogger(__name__)

# Suffix for checkpoint artifact files.
_CHECKPOINT_SUFFIX: str = ".pkl"
# Suffix for the in-progress temporary file (renamed atomically on success).
_TEMP_SUFFIX: str = ".tmp"


class StageCheckpointer:
    """Save and restore stage artifacts for resumable runs.

    When constructed with ``checkpoint_dir=None`` the checkpointer is *disabled*:
    :attr:`enabled` is ``False`` and all save/has/load calls become no-ops or
    return falsy, so the same pipeline code runs unchanged with or without
    checkpointing.

    Thread-safety:
        Not thread-safe for concurrent writes to the *same* stage name (the
        pipeline writes each stage from one thread). Different stage names use
        different files and may be written concurrently.

    Args:
        checkpoint_dir: Directory for checkpoint files, or ``None`` to disable.

    Raises:
        CheckpointError: If a non-``None`` directory cannot be created.

    Example:
        >>> import tempfile, pathlib
        >>> d = pathlib.Path(tempfile.mkdtemp())
        >>> ck = StageCheckpointer(d)
        >>> ck.save("stage_x", {"a": 1})
        >>> ck.has_stage("stage_x")
        True
        >>> ck.load("stage_x")
        {'a': 1}
    """

    def __init__(self, checkpoint_dir: Path | None) -> None:
        self._dir = checkpoint_dir
        if checkpoint_dir is not None:
            try:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise CheckpointError(
                    f"could not create checkpoint dir {checkpoint_dir}: {exc}"
                ) from exc

    @property
    def enabled(self) -> bool:
        """Whether checkpointing is active (a directory was supplied)."""
        return self._dir is not None

    def _path_for(self, stage_name: str) -> Path:
        """Return the checkpoint file path for a stage.

        Args:
            stage_name: The stage's checkpoint name.

        Returns:
            The full path to its ``.pkl`` file.

        Raises:
            CheckpointError: If checkpointing is disabled.
        """
        if self._dir is None:
            raise CheckpointError("checkpointing is disabled (no checkpoint_dir)")
        return self._dir / f"{stage_name}{_CHECKPOINT_SUFFIX}"

    def has_stage(self, stage_name: str) -> bool:
        """Return whether a completed checkpoint exists for a stage.

        Args:
            stage_name: The stage's checkpoint name.

        Returns:
            ``True`` if checkpointing is enabled and the artifact file exists.
        """
        if self._dir is None:
            return False
        return self._path_for(stage_name).exists()

    def save(self, stage_name: str, artifact: Any) -> None:
        """Atomically persist a stage artifact.

        Args:
            stage_name: The stage's checkpoint name.
            artifact: Any picklable object (lists, NumPy arrays, dicts, etc.).

        Raises:
            CheckpointError: If serialisation or the atomic rename fails.

        Example:
            >>> import tempfile, pathlib
            >>> ck = StageCheckpointer(pathlib.Path(tempfile.mkdtemp()))
            >>> ck.save("s", [1, 2, 3])
        """
        if self._dir is None:
            return  # disabled: silently skip (documented no-op behaviour)
        final_path = self._path_for(stage_name)
        temp_path = final_path.with_suffix(_CHECKPOINT_SUFFIX + _TEMP_SUFFIX)
        try:
            with temp_path.open("wb") as handle:
                pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())  # durability before the rename
            os.replace(temp_path, final_path)  # atomic on POSIX and Windows
            logger.debug("Saved checkpoint %r (%s)", stage_name, final_path)
        except (OSError, pickle.PicklingError) as exc:
            # Clean up a partial temp file; never leave a corrupt artifact behind.
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise CheckpointError(
                f"failed to save checkpoint {stage_name!r}: {exc}"
            ) from exc

    def load(self, stage_name: str) -> Any:
        """Load a previously saved stage artifact.

        Args:
            stage_name: The stage's checkpoint name.

        Returns:
            The deserialised artifact.

        Raises:
            CheckpointError: If the checkpoint is missing or corrupt.

        Example:
            >>> import tempfile, pathlib
            >>> ck = StageCheckpointer(pathlib.Path(tempfile.mkdtemp()))
            >>> ck.save("s", 99)
            >>> ck.load("s")
            99
        """
        path = self._path_for(stage_name)
        if not path.exists():
            raise CheckpointError(f"no checkpoint found for stage {stage_name!r}")
        try:
            with path.open("rb") as handle:
                return pickle.load(handle)
        except (OSError, pickle.UnpicklingError, EOFError) as exc:
            raise CheckpointError(
                f"failed to load checkpoint {stage_name!r} from {path}: {exc}"
            ) from exc

    def delete(self, stage_name: str) -> None:
        """Delete a stage checkpoint if it exists.

        Args:
            stage_name: The stage's checkpoint name.
        """
        if self._dir is None:
            return
        self._path_for(stage_name).unlink(missing_ok=True)

    def completed_stages(self) -> set[str]:
        """Return the set of stage names with existing checkpoints.

        Returns:
            Stage names (without the ``.pkl`` suffix); empty if disabled.
        """
        if self._dir is None:
            return set()
        return {
            p.name[: -len(_CHECKPOINT_SUFFIX)]
            for p in self._dir.glob(f"*{_CHECKPOINT_SUFFIX}")
        }
