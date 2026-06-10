"""Validate supported SQL through DuckDB Substrait and PyTorch."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tpch_torch.duckdb_bridge import connect_database
from tpch_torch.relational import SQLValidationResult
from tpch_torch.runner import PlanSource, load_sql, validate_sql_with_plan_source
from tpch_torch.sql import get_tpch_query

DEFAULT_SQL_TOLERANCE = 1e-2
FIRST_TPCH_QUERY_ID = 1
LAST_TPCH_QUERY_ID = 22
ALL_TPCH_QUERY_IDS = tuple(range(FIRST_TPCH_QUERY_ID, LAST_TPCH_QUERY_ID + 1))
QueryLoader = Callable[[object, int], str]
QueryValidator = Callable[[object, str, str, PlanSource], SQLValidationResult]


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
    parser.add_argument(
        "--plan-source",
        choices=("substrait", "duckdb-logical", "auto"),
        default="substrait",
        help="Plan admission path before PyTorch execution",
    )
    parser.add_argument("--keep-going", action="store_true", help="Continue batch validation after a query fails")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_SQL_TOLERANCE)
    return parser


def parse_query_ids(raw: str) -> tuple[int, ...]:
    if raw == "all":
        return ALL_TPCH_QUERY_IDS
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
    plan_source: PlanSource = "substrait",
    load_query: QueryLoader = get_tpch_query,
    validator: QueryValidator = validate_sql_with_plan_source,
) -> list[BatchValidationRecord]:
    records: list[BatchValidationRecord] = []
    for query_id in query_ids:
        try:
            record = _validate_one_query(
                con,
                query_id,
                device=device,
                tolerance=tolerance,
                plan_source=plan_source,
                load_query=load_query,
                validator=validator,
            )
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
    *,
    device: str,
    tolerance: float,
    plan_source: PlanSource,
    load_query: QueryLoader,
    validator: QueryValidator,
) -> BatchValidationRecord:
    sql = load_query(con, query_id)
    result = validator(con, sql, device, plan_source)
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
        if args.queries is not None:
            records = validate_queries(
                con,
                parse_query_ids(args.queries),
                device=args.device,
                tolerance=args.tolerance,
                keep_going=args.keep_going,
                plan_source=args.plan_source,
            )
            _print_batch_records(records)
            _raise_on_batch_failures(records)
            return
        sql = load_sql(con, query=args.query, sql=args.sql, sql_file=args.sql_file)
        result = validate_sql_with_plan_source(
            con,
            sql,
            device=args.device,
            plan_source=args.plan_source,
        )
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


def _print_batch_records(records: list[BatchValidationRecord]) -> None:
    for record in records:
        if record.ok:
            print(
                f"validated query={record.query_id} rows={record.row_count} "
                f"max_abs_error={record.max_abs_error:.6g}"
            )
            continue
        print(f"failed query={record.query_id} {record.message}")


def _raise_on_batch_failures(records: list[BatchValidationRecord]) -> None:
    failed_query_ids = [record.query_id for record in records if not record.ok]
    if failed_query_ids:
        formatted = ",".join(str(query_id) for query_id in failed_query_ids)
        raise AssertionError(f"batch validation failed: Q{formatted}")


if __name__ == "__main__":
    main()
