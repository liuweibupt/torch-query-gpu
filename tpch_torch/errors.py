"""Shared execution and compilation errors."""


class UnsupportedPlanError(ValueError):
    """Raised when a frontend-admitted plan is outside the PyTorch backend subset."""
