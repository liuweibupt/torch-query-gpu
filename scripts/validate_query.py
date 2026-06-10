"""Validate supported SQL through DuckDB Substrait and PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path

from tpch_torch.duckdb_bridge import connect_database
from tpch_torch.runner import load_sql, validate_sql

DEFAULT_SQL_TOLERANCE = 1e-2


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
