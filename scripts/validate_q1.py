"""Validate PyTorch TPC-H Q1 output against DuckDB."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tpch_torch.duckdb_bridge import connect_database
from tpch_torch.validate import validate_q1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate PyTorch Q1 against DuckDB")
    parser.add_argument("--db", type=Path, required=True, help="Input DuckDB database path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Execution device")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Max absolute numeric error")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_device(args.device)
    con = connect_database(args.db)
    try:
        result = validate_q1(con, device=args.device)
    finally:
        con.close()
    if result.max_abs_error > args.tolerance:
        raise AssertionError(
            f"Q1 validation failed: max_abs_error={result.max_abs_error} tolerance={args.tolerance}"
        )
    print(f"validated rows={result.row_count} max_abs_error={result.max_abs_error:.6g}")


def _validate_device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")


if __name__ == "__main__":
    main()
