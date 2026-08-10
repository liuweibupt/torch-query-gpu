"""Execution mode names for TQP SQL runners."""

from __future__ import annotations

from typing import Literal

ExecutionMode = Literal["strict", "universal"]


def validate_execution_mode(mode: str) -> ExecutionMode:
    if mode not in {"strict", "universal"}:
        raise ValueError(f"unknown execution mode: {mode}")
    return mode  # type: ignore[return-value]
