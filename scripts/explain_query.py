"""Explain framework-level SQL admission and strict TQP coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tpch_torch.duckdb_bridge import connect_database
from tpch_torch.runner import load_sql
from tpch_torch.sql_admission import SQLAdmission, admit_sql


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explain SQL lowering into the TQP operator graph")
    parser.add_argument("--db", type=Path, required=True, help="Input DuckDB database path")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query", type=int, help="TPC-H query number")
    source.add_argument("--sql", help="Inline SQL text")
    source.add_argument("--sql-file", type=Path, help="SQL file path")
    parser.add_argument("--json", action="store_true", help="Print machine-readable admission JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    con = connect_database(args.db)
    try:
        sql = load_sql(con, query=args.query, sql=args.sql, sql_file=args.sql_file)
        admission = admit_sql(con, sql)
    finally:
        con.close()
    if args.json:
        print(json.dumps(admission.to_dict(), indent=2, sort_keys=True))
        return
    _print_admission(admission)


def _print_admission(admission: SQLAdmission) -> None:
    graph = admission.graph
    coverage = admission.strict_coverage
    output = ", ".join(f"{column.name}:{column.type_name}" for column in graph.output_schema)
    print(f"frontend={admission.plan.frontend}")
    print(f"query_id={admission.plan.query_id}")
    print(f"root={graph.root_id} {graph.root.name}")
    print(f"nodes={coverage.node_count}")
    print(f"output_schema=[{output}]")
    print(f"strict_admissible={str(coverage.strict_admissible).lower()}")
    if coverage.strict_admissible:
        return
    print("strict_gaps:")
    for gap in coverage.gaps:
        print(f"- {gap.node_id} {gap.node_name} ({gap.node_kind}): {gap.reason}")


if __name__ == "__main__":
    main()
