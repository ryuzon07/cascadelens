"""Errors raised while building or verifying splits."""


class SplitError(Exception):
    """A split could not be built or verified."""


class LeakageError(SplitError):
    """Two partitions that must stay apart share a group."""
