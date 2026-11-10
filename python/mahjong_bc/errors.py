"""Package-specific error types."""


class MahjongBCError(Exception):
    """Base class for package errors."""


class ConfigurationError(MahjongBCError, ValueError):
    """A dataset identity or package configuration is invalid."""


class DataError(MahjongBCError, ValueError):
    """A dataset batch or array layout violates the shared contract."""


class CacheContractError(MahjongBCError):
    """The prepared NumPy cache cannot be consumed under this contract."""


class CheckpointError(MahjongBCError):
    """A policy checkpoint is corrupt, incompatible, or not supported here."""
