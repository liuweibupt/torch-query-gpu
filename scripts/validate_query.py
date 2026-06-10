"""Validate supported SQL through DuckDB Substrait and PyTorch."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tpch_torch.duckdb_bridge import connect_database
from tpch_torch.relational import SQLValidationResult
from tpch_torch.runner import load_sql, validate_sql
from tpch_torch.sql import get_tpch_query

DEFAULT_SQL_TOLERANCE = 1e-2
QueryLoader = Callable[[object, int], str]
QueryValidator = Callable[[object, str, str], SQLValidationResult]


@dataclass(frozen=True)
class BatchValidationRecord:
    query_id: int
    ok: bool
    message: str
    row_count: int = 0
    max_abs_error: float | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate supported SQL against DuckDB")
    parser.add_argument("--db", type=Path, required=True, help="Input DuckDB database path")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query", type=int, help="TPC-H query number")
    source.add_argument("--queries", help="TPC-H query ids as comma-separated numbers")
    source.add_argument("--sql", help="Inline SQL text")
    source.add_argument("--sql-file", type=Path, help="SQL file path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Execution device")
    parser.add_argument("--keep-going", action="store_true", help="Continue batch validation after a query fails")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_SQL_TOLERANCE)
    return parser


def parse_query_ids(raw: str) -> tuple[int, ...]:
    query_ids = tuple(int(item) for item in raw.split(",") if item)
    if not query_ids:
        raise ValueError("at least one query id is required")
    return query_ids


def validate_queries(
    con: object,
    query_ids: tuple[int, ...],
    *,
    device: str,
    tolerance: float,
    keep_going: bool,
    load_query: QueryLoader = get_tpch_query,
    validator: QueryValidator = validate_sql,
) -> list[BatchValidationRecord]:
    records: list[BatchValidationRecord] = []
    for query_id in query_ids:
        try:
            record = _validate_one_query(con, query_id, device, tolerance, load_query, validator)
        except Exception as exc:
            if not keep_going:
                raise
            records.append(BatchValidationRecord(query_id=query_id, ok=False, message=str(exc)))
            continue
        records.append(record)
    return records


def _validate_one_query(
    con: object,
    query_id: int,
    device: str,
    tolerance: float,
    load_query: QueryLoader,
    validator: QueryValidator,
) -> BatchValidationRecord:
    sql = load_query(con, query_id)
    result = validator(con, sql, device)
    if result.max_abs_error > tolerance:
        raise AssertionError(
            f"Q{result.query_id} validation failed: "
            f"max_abs_error={result.max_abs_error} tolerance={tolerance}"
        )
    return BatchValidationRecord(
        query_id=result.query_id,
        ok=True,
        message="validated",
        row_count=result.row_count,
        max_abs_error=result.max_abs_error,
    )


def main() -> None:
    args = build_parser().parse_args()
    con = connect_database(args.db)
    try:
        sql = load_sql(con, query=args.query, sql=args.sql, sql_file=args.sql_file)
        result = validate_sql(con, sql, device=args.device)
    finally:
        con.close()
    if result.max_abs_error > args.tolerance:
        raise AssertionError(
            f"Q{result.query_id} validation failed: "
            f"max_abs_error={result.max_abs_error} tolerance={args.tolerance}"
        )
    print(
        f"validated query={result.query_id} rows={result.row_count} "
        f"max_abs_error={result.max_abs_error:.6g}"
    )


if __name__ == "__main__":
    main()
