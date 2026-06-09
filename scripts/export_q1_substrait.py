"""Export DuckDB Substrait JSON for canonical TPC-H Q1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tpch_torch.duckdb_bridge import connect_database, export_substrait_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export TPC-H Q1 Substrait JSON from DuckDB")
    parser.add_argument("--db", type=Path, required=True, help="Input DuckDB database path")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    con = connect_database(args.db)
    try:
        plan_json = export_substrait_json(con)
    finally:
        con.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan_json, indent=2, sort_keys=True))
    print(f"wrote Q1 Substrait JSON to {args.out}")


if __name__ == "__main__":
    main()
