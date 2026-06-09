"""Validation helpers comparing PyTorch Q1 output to DuckDB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import duckdb

from tpch_torch.duckdb_bridge import export_substrait_json, fetch_lineitem_tensor_table, run_duckdb_q1
from tpch_torch.queries.q01 import Q1_RESULT_COLUMNS, execute_q1
from tpch_torch.substrait import Q1Plan, compile_q1_substrait_plan

NUMERIC_COLUMNS = tuple(name for name in Q1_RESULT_COLUMNS if name not in {"l_returnflag", "l_linestatus"})


@dataclass(frozen=True)
class ValidationResult:
    """Summary of a DuckDB-vs-PyTorch Q1 comparison."""

    row_count: int
    max_abs_error: float
    duckdb_rows: list[dict[str, Any]]
    pytorch_rows: list[dict[str, Any]]


def validate_q1(
    con: duckdb.DuckDBPyConnection, device: str = "cpu", plan: Q1Plan | None = None
) -> ValidationResult:
    """Compile DuckDB Substrait Q1 and compare PyTorch execution to DuckDB."""

    if plan is None:
        plan_json = export_substrait_json(con)
        plan = compile_q1_substrait_plan(plan_json)
    table = fetch_lineitem_tensor_table(con, device=device)
    pytorch_rows = execute_q1(table, plan)
    duckdb_rows = run_duckdb_q1(con)
    max_abs_error = compare_q1_rows(duckdb_rows, pytorch_rows)
    return ValidationResult(
        row_count=len(duckdb_rows),
        max_abs_error=max_abs_error,
        duckdb_rows=duckdb_rows,
        pytorch_rows=pytorch_rows,
    )


def compare_q1_rows(
    duckdb_rows: Sequence[dict[str, Any]], pytorch_rows: Sequence[dict[str, Any]]
) -> float:
    """Return max numeric absolute error or raise on schema/key mismatch."""

    if len(duckdb_rows) != len(pytorch_rows):
        raise AssertionError(f"row count mismatch: DuckDB={len(duckdb_rows)} PyTorch={len(pytorch_rows)}")

    max_abs_error = 0.0
    for index, (duckdb_row, pytorch_row) in enumerate(zip(duckdb_rows, pytorch_rows)):
        _assert_group_keys_equal(index, duckdb_row, pytorch_row)
        for column_name in NUMERIC_COLUMNS:
            error = abs(float(duckdb_row[column_name]) - float(pytorch_row[column_name]))
            max_abs_error = max(max_abs_error, error)
    return max_abs_error


def _assert_group_keys_equal(
    index: int, duckdb_row: dict[str, Any], pytorch_row: dict[str, Any]
) -> None:
    duckdb_key = (duckdb_row["l_returnflag"], duckdb_row["l_linestatus"])
    pytorch_key = (pytorch_row["l_returnflag"], pytorch_row["l_linestatus"])
    if duckdb_key != pytorch_key:
        raise AssertionError(f"group key mismatch at row {index}: DuckDB={duckdb_key} PyTorch={pytorch_key}")
