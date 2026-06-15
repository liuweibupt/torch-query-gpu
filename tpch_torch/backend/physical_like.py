"""LIKE pattern helpers for encoded physical string columns."""

from __future__ import annotations

import re


def like_matches(value: str, pattern: str) -> bool:
    """Return whether a SQL LIKE pattern matches a string value."""

    escaped = re.escape(pattern).replace("%", ".*").replace("_", ".")
    return re.fullmatch(escaped, value, flags=re.S) is not None
