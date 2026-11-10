"""Package-specific error types."""


class MahjongBCError(Exception):
    """Base class for package errors."""


class ConfigurationError(MahjongBCError, ValueError):
    """A model, training, dataset, or checkpoint configuration is invalid."""


class DataError(MahjongBCError, ValueError):
    """A dataset batch or array layout violates the shared contract."""


class CheckpointError(MahjongBCError):
    """A checkpoint is missing, corrupt, or unsupported."""


class ResumeError(MahjongBCError):
    """A resume checkpoint does not match the requested run identity."""


class CacheContractError(MahjongBCError):
    """The prepared NumPy cache cannot be consumed under this contract."""
