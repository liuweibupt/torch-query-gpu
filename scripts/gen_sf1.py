"""Generate a DuckDB database containing TPC-H data."""

from __future__ import annotations

import argparse
from pathlib import Path

from tpch_torch.duckdb_bridge import connect_database, generate_tpch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TPC-H data in DuckDB")
    parser.add_argument("--db", type=Path, required=True, help="Output DuckDB database path")
    parser.add_argument("--sf", type=float, default=1.0, help="TPC-H scale factor, default: 1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    con = connect_database(args.db)
    try:
        generate_tpch(con, scale_factor=args.sf)
    finally:
        con.close()
    print(f"generated TPC-H SF{args.sf:g} at {args.db}")


if __name__ == "__main__":
    main()
