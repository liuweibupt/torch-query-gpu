"""Run TPC-H Q1 through DuckDB Substrait compilation and PyTorch execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from tpch_torch.duckdb_bridge import connect_database, export_substrait_json, fetch_lineitem_tensor_table
from tpch_torch.queries.q01 import execute_q1
from tpch_torch.substrait import compile_q1_substrait_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TPC-H Q1 with PyTorch tensors")
    parser.add_argument("--db", type=Path, required=True, help="Input DuckDB database path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Execution device")
    parser.add_argument("--substrait-json", type=Path, help="Use an existing Substrait JSON file")
    parser.add_argument("--json", action="store_true", help="Print result rows as JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_device(args.device)
    con = connect_database(args.db)
    try:
        plan_json = _load_or_export_plan(con, args.substrait_json)
        plan = compile_q1_substrait_plan(plan_json)
        table = fetch_lineitem_tensor_table(con, device=args.device)
        rows, elapsed_ms = _timed_execute(table, plan, args.device)
    finally:
        con.close()
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            print(row)
    print(f"q1_pytorch_ms={elapsed_ms:.3f}")


def _load_or_export_plan(con: Any, substrait_json: Path | None) -> dict[str, Any]:
    if substrait_json is not None:
        return json.loads(substrait_json.read_text())
    return export_substrait_json(con)


def _timed_execute(table: Any, plan: Any, device: str) -> tuple[list[dict[str, Any]], float]:
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        rows = execute_q1(table, plan)
        end.record()
        torch.cuda.synchronize()
        return rows, float(start.elapsed_time(end))
    start_time = perf_counter()
    rows = execute_q1(table, plan)
    elapsed_ms = (perf_counter() - start_time) * 1000.0
    return rows, elapsed_ms


def _validate_device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")


if __name__ == "__main__":
    main()
