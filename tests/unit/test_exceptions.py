"""Unit tests for the deduplication exception hierarchy.

Every module in the package raises errors from a single rooted tree so that a
caller can catch :class:`DedupError` to trap any pipeline failure, or catch a
specific subclass to handle one narrow failure mode (retry on IOError, abort on
ConfigError). These tests pin the subclass relationships, the catch-as-base
behaviour, the package-level re-exports, and the deliberate fact that the
domain ``IOError`` is *not* the builtin ``OSError`` — so that exception handling
in caller code stays correct and unambiguous.
"""

from __future__ import annotations

import pytest

import dedup_pipeline
import dedup_pipeline.exceptions as exc_mod
from dedup_pipeline.exceptions import (
    CheckpointError,
    ConfigError,
    DedupError,
    EvaluationError,
    HashingError,
    IOError,
)

_SUBCLASSES = [ConfigError, HashingError, IOError, CheckpointError, EvaluationError]


def test_dedup_error_subclasses_exception() -> None:
    """DedupError is a subclass of the builtin Exception.

    Matters because library code never raises bare Exception; rooting the tree
    at Exception lets generic ``except Exception`` handlers (and test runners)
    still see pipeline errors, while the dedicated base enables precise trapping.
    """
    assert issubclass(DedupError, Exception)


def test_all_concrete_errors_subclass_dedup_error() -> None:
    """Every concrete error type derives from DedupError.

    Matters because the whole point of the hierarchy is that catching
    DedupError traps *any* pipeline failure; a stray subclass rooted elsewhere
    would leak past that catch and crash the caller.
    """
    for err in _SUBCLASSES:
        assert issubclass(err, DedupError)


def test_catching_dedup_error_traps_each_subclass() -> None:
    """Raising any subclass can be caught as DedupError.

    Matters because callers that wrap a pipeline run in ``except DedupError``
    rely on this to convert every internal failure into a single handled path;
    if one subclass escaped, an otherwise-recoverable run would abort uncaught.
    """
    for err in _SUBCLASSES:
        with pytest.raises(DedupError):
            raise err("boom")


def test_subclasses_are_distinct_for_narrow_handling() -> None:
    """A specific subclass does not catch a sibling subclass.

    Matters because the hierarchy promises narrow handling (e.g. retry only on
    IOError); if ConfigError were also catchable as IOError, a config bug would
    be silently retried forever instead of aborting the run.
    """
    with pytest.raises(ConfigError):
        raise ConfigError("config is wrong")
    # A ConfigError must NOT be an instance of an unrelated sibling.
    assert not issubclass(ConfigError, IOError)
    assert not issubclass(IOError, ConfigError)


def test_package_reexports_exception_hierarchy() -> None:
    """The top-level package re-exports the full exception tree.

    Matters because the documented public API is ``from dedup_pipeline import
    DedupError, IOError, ...``; if a re-export were dropped, downstream code
    would break at import time and the namespaced ``dedup_pipeline.IOError``
    contract (distinct from the builtin) would be unavailable.
    """
    from dedup_pipeline import (
        CheckpointError as PkgCheckpointError,
    )
    from dedup_pipeline import (
        ConfigError as PkgConfigError,
    )
    from dedup_pipeline import (
        DedupError as PkgDedupError,
    )
    from dedup_pipeline import (
        EvaluationError as PkgEvaluationError,
    )
    from dedup_pipeline import (
        HashingError as PkgHashingError,
    )
    from dedup_pipeline import (
        IOError as PkgIOError,
    )

    assert PkgDedupError is DedupError
    assert PkgConfigError is ConfigError
    assert PkgHashingError is HashingError
    assert PkgIOError is IOError
    assert PkgCheckpointError is CheckpointError
    assert PkgEvaluationError is EvaluationError
    for name in (
        "DedupError",
        "ConfigError",
        "HashingError",
        "IOError",
        "CheckpointError",
        "EvaluationError",
    ):
        assert name in dedup_pipeline.__all__


# ----- Pathological / edge cases -----------------------------------------


def test_each_exception_str_returns_its_message() -> None:
    """str(exc) returns exactly the message passed to the constructor.

    Pathological case: an empty message must round-trip as the empty string,
    not "None" or a repr. Matters because log lines and re-raised wrappers embed
    str(exc); a mangled message would obscure the real cause of a failed run.
    """
    for err in [DedupError, *_SUBCLASSES]:
        message = f"{err.__name__}: detailed failure context"
        assert str(err(message)) == message
    # Empty-message edge case round-trips cleanly.
    assert str(DedupError("")) == ""


def test_domain_io_error_is_not_builtin_os_error() -> None:
    """The domain IOError is distinct from the builtin OSError.

    Pathological case: the package deliberately shadows the builtin ``IOError``
    name (an alias of OSError) within its own namespace. Matters because a
    caller writing ``except OSError`` for true OS faults must NOT accidentally
    swallow a pipeline IOError, and vice versa; the domain class subclasses
    DedupError/Exception only, never OSError.
    """
    assert not issubclass(exc_mod.IOError, OSError)
    assert issubclass(exc_mod.IOError, DedupError)
    assert issubclass(exc_mod.IOError, Exception)
    # The instance must not be catchable as a builtin OSError.
    instance = exc_mod.IOError("unreadable shard")
    assert not isinstance(instance, OSError)
