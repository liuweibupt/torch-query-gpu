"""Benchmark cold/hot SQL execution through TQP and PyTorch."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tpch_torch.benchmark import (
    DEFAULT_COLD_RUNS,
    DEFAULT_HOT_RUNS,
    DEFAULT_WARMUP_RUNS,
    BenchmarkConfig,
    BenchmarkReport,
    benchmark_sql,
)
from tpch_torch.duckdb_bridge import connect_database
from tpch_torch.runner import load_sql


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark cold/hot SQL execution with PyTorch tensors")
    parser.add_argument("--db", type=Path, required=True, help="Input DuckDB database path")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query", type=int, help="TPC-H query number")
    source.add_argument("--sql", help="Inline SQL text")
    source.add_argument("--sql-file", type=Path, help="SQL file path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Execution device")
    parser.add_argument("--frontend", choices=("sirius", "substrait"), default="sirius")
    parser.add_argument("--cold-runs", type=int, default=DEFAULT_COLD_RUNS)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--hot-runs", type=int, default=DEFAULT_HOT_RUNS)
    parser.add_argument(
        "--compressed-masks",
        action="store_true",
        help="Use explicit compressed mask execution where implemented, currently TPC-H Q6",
    )
    parser.add_argument("--json", action="store_true", help="Print benchmark report as JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sql = _load_sql(args)
    report = benchmark_sql(
        BenchmarkConfig(
            db_path=args.db,
            sql=sql,
            device=args.device,
            frontend=args.frontend,
            cold_runs=args.cold_runs,
            warmup_runs=args.warmup_runs,
            hot_runs=args.hot_runs,
            use_compressed_masks=args.compressed_masks,
        )
    )
    if args.json:
        print(json.dumps(_report_dict(report), indent=2, sort_keys=True))
        return
    _print_report(report)


def _load_sql(args: argparse.Namespace) -> str:
    con = connect_database(args.db)
    try:
        return load_sql(con, query=args.query, sql=args.sql, sql_file=args.sql_file)
    finally:
        con.close()


def _print_report(report: BenchmarkReport) -> None:
    config = report.config
    print(
        "benchmark "
        f"device={config.device} frontend={config.frontend} "
        f"compressed_masks={config.use_compressed_masks}"
    )
    _print_summary("cold", report.cold)
    print(f"warmup_runs={config.warmup_runs}")
    _print_summary("hot", report.hot)


def _print_summary(name: str, summary) -> None:
    if summary.count == 0:
        print(f"{name} count=0")
        return
    print(
        f"{name} count={summary.count} "
        f"median_ms={summary.median_ms:.3f} "
        f"mean_ms={summary.mean_ms:.3f} "
        f"p95_ms={summary.p95_ms:.3f} "
        f"min_ms={summary.min_ms:.3f} "
        f"max_ms={summary.max_ms:.3f} "
        f"stdev_ms={summary.stdev_ms:.3f}"
    )


def _report_dict(report: BenchmarkReport) -> dict:
    raw = asdict(report)
    raw["config"]["db_path"] = str(report.config.db_path)
    return raw


if __name__ == "__main__":
    main()
