"""Custom exception hierarchy for the deduplication pipeline.

This module defines the single rooted exception tree that the rest of the
package raises. Centralising the hierarchy lets callers catch
:class:`DedupError` to trap *any* pipeline failure, or catch a specific
subclass to handle a narrow failure mode (for example, retry on
:class:`IOError` but abort on :class:`ConfigError`).

Responsibility:
    * Define the base error and its five concrete subclasses.

Inputs:
    * None (this module declares types only).

Outputs:
    * Exception classes imported by every other module.
"""

from __future__ import annotations


class DedupError(Exception):
    """Base class for every error raised by the deduplication pipeline.

    Catching this class traps all errors that originate inside
    ``dedup_pipeline``. Library code never raises a bare :class:`Exception`;
    it always raises this class or one of its subclasses so that callers can
    distinguish *our* failures from unrelated runtime errors (for example, a
    ``KeyboardInterrupt`` or a third-party library bug).

    Example:
        >>> try:
        ...     raise DedupError("something went wrong")
        ... except DedupError as exc:
        ...     print(str(exc))
        something went wrong
    """


class ConfigError(DedupError):
    """Raised when configuration is invalid or internally inconsistent.

    Typical triggers:
        * ``num_hash_functions`` is not equal to ``lsh_bands * lsh_rows``.
        * A field receives a value outside its permitted range.

    Example:
        >>> raise ConfigError("num_hash_functions must equal bands * rows")
        Traceback (most recent call last):
        ...
        dedup_pipeline.exceptions.ConfigError: num_hash_functions must equal bands * rows
    """


class HashingError(DedupError):
    """Raised when hash-function construction or evaluation fails.

    Typical triggers:
        * Requesting more hash functions than the prime modulus can support.
        * A hash backend (``xxhash``/``mmh3``) is missing or returns an
          unexpected type.
    """


class IOError(DedupError):  # noqa: A001 - intentional domain-specific name
    """Raised on file read/write or data-format errors.

    This deliberately shadows the builtin ``IOError`` *within this package's
    namespace* so that callers can write ``except dedup_pipeline.IOError``.
    The builtin alias ``OSError`` remains available for true OS-level errors;
    we only raise this class for *pipeline* I/O problems (an unreadable shard,
    a malformed Parquet schema, an unsupported source type).

    Typical triggers:
        * The source path does not exist or matches no files.
        * A record cannot be decoded into a ``Document``.
        * An unsupported output format is requested.
    """


class CheckpointError(DedupError):
    """Raised on checkpoint save/load/corruption errors.

    Typical triggers:
        * A checkpoint file is present but truncated or unreadable.
        * The checkpoint manifest references a stage that the current code
          version does not recognise.
    """


class EvaluationError(DedupError):
    """Raised when a metric computation cannot be performed.

    Typical triggers:
        * Precision/recall requested with an empty prediction *and* empty
          ground-truth set (the metric is undefined).
        * Ground-truth pairs reference document ids absent from the corpus.
    """
