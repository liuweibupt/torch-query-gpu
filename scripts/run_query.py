"""Run supported TPC-H SQL through a TQP frontend and PyTorch backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.validate_query import resolve_frontend
from tpch_torch.duckdb_bridge import connect_database
from tpch_torch.runner import load_sql, timed_run_sql


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run supported SQL with PyTorch tensors")
    parser.add_argument("--db", type=Path, required=True, help="Input DuckDB database path")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query", type=int, help="TPC-H query number")
    source.add_argument("--sql", help="Inline SQL text")
    source.add_argument("--sql-file", type=Path, help="SQL file path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Execution device")
    parser.add_argument(
        "--frontend",
        choices=("sirius", "substrait", "auto"),
        default="sirius",
        help="TQP frontend used before PyTorch execution",
    )
    parser.add_argument(
        "--plan-source",
        choices=("substrait", "duckdb-logical", "auto"),
        default=None,
        help="Legacy alias for --frontend: duckdb-logical maps to sirius",
    )
    parser.add_argument("--json", action="store_true", help="Print result rows as JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frontend = resolve_frontend(args.frontend, args.plan_source)
    con = connect_database(args.db)
    try:
        sql = load_sql(con, query=args.query, sql=args.sql, sql_file=args.sql_file)
        result, elapsed_ms = timed_run_sql(
            con,
            sql,
            device=args.device,
            frontend=frontend,
        )
    finally:
        con.close()
    if args.json:
        print(json.dumps(result.rows, indent=2, sort_keys=True))
    else:
        for row in result.rows:
            print(row)
    label = "generic" if result.query_id is None else f"q{result.query_id:02d}"
    print(f"{label}_pytorch_ms={elapsed_ms:.3f}")


if __name__ == "__main__":
    main()
