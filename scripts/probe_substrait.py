"""Probe native DuckDB Substrait export support for original TPC-H SQL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tpch_torch.capabilities import probe_tpch_substrait_exports
from tpch_torch.duckdb_bridge import connect_database

ALL_TPCH_QUERIES = tuple(range(1, 23))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe native DuckDB Substrait export support")
    parser.add_argument("--db", type=Path, required=True, help="Input DuckDB database path")
    parser.add_argument(
        "--queries",
        required=True,
        help="TPC-H query ids as comma-separated numbers, or 'all'",
    )
    parser.add_argument("--json", action="store_true", help="Print probe results as JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    con = connect_database(args.db)
    try:
        statuses = probe_tpch_substrait_exports(con, parse_query_ids(args.queries))
    finally:
        con.close()
    if args.json:
        print(json.dumps([status.to_dict() for status in statuses], indent=2, sort_keys=True))
        return
    for status in statuses:
        state = "OK" if status.export_ok else "FAIL"
        message = status.error_message or ""
        print(
            f"Q{status.query_id:02d} export={state} "
            f"executor_supported={status.executor_supported} {message}"
        )


def parse_query_ids(raw: str) -> tuple[int, ...]:
    if raw == "all":
        return ALL_TPCH_QUERIES
    query_ids = tuple(int(item) for item in raw.split(",") if item)
    if not query_ids:
        raise ValueError("at least one query id is required")
    return query_ids


if __name__ == "__main__":
    main()
