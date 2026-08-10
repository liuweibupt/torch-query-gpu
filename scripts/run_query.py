"""Run supported TPC-H SQL through a TQP frontend and PyTorch backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tpch_torch.backend.physical_partitionable import PartitionConfig
from tpch_torch.duckdb_bridge import connect_database
from tpch_torch.execution_mode import validate_execution_mode
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
        choices=("sirius", "substrait"),
        default="sirius",
        help="TQP frontend used before PyTorch execution",
    )
    parser.add_argument("--json", action="store_true", help="Print result rows as JSON")
    parser.add_argument(
        "--compressed-masks",
        action="store_true",
        help="Use explicit compressed mask execution where implemented, currently TPC-H Q6",
    )
    parser.add_argument("--partition-table", help="Enable partitionable execution over this table")
    parser.add_argument("--partition-chunk-size", type=int, help="Rows per partitionable chunk")
    parser.add_argument(
        "--execution-mode",
        choices=("strict", "universal"),
        default="strict",
        help=(
            "strict uses only implemented TQP operators; universal explicitly "
            "materializes unsupported SQL through TensorRecordBatch"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    con = connect_database(args.db)
    try:
        sql = load_sql(con, query=args.query, sql=args.sql, sql_file=args.sql_file)
        result, elapsed_ms = timed_run_sql(
            con,
            sql,
            device=args.device,
            frontend=args.frontend,
            use_compressed_masks=args.compressed_masks,
            partition_config=_partition_config(args),
            execution_mode=validate_execution_mode(args.execution_mode),
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


def _partition_config(args: argparse.Namespace) -> PartitionConfig | None:
    if args.partition_table is None and args.partition_chunk_size is None:
        return None
    if args.partition_table is None or args.partition_chunk_size is None:
        raise SystemExit("--partition-table and --partition-chunk-size must be provided together")
    return PartitionConfig(args.partition_table, args.partition_chunk_size)


if __name__ == "__main__":
    main()
