#!/usr/bin/env python3
"""Close the TPC-H Q6 HBF selective-return time denominator gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import duckdb
import numpy as np
import torch

from scripts.profile_hbf_tpch import _minimum_modules

DEFAULT_DB = Path("data/tpch_sf1.duckdb")
DEFAULT_OUT_DIR = Path("data/hbf_q6_time_gate")
DEFAULT_PAGE_BYTES = 4096
DEFAULT_ROWID_BYTES = 8
DEFAULT_STACK_COUNT = 3
DEFAULT_SF_SCALE = 100.0
DEFAULT_GRADE_FLOOR_GB_S = 400.0
DEFAULT_PAYLOAD_EFFICIENCY = 0.85
DEFAULT_TARGET_MS = "1,5,10,14,20,50,100"
DEFAULT_ROW_GROUP_ROWS = "4096,16384,65536"
DEFAULT_WARMUP = 3
DEFAULT_HOT_RUNS = 10
REQUIRED_COLUMNS = ("l_discount", "l_extendedprice", "l_quantity", "l_shipdate")
PREDICATE_COLUMNS = ("l_discount", "l_quantity", "l_shipdate")
PROJECTED_COLUMNS = ("l_discount", "l_extendedprice")
DATE_LOWER = 19940101
DATE_UPPER = 19950101
DISCOUNT_LOWER = 0.05
DISCOUNT_UPPER = 0.07
QUANTITY_UPPER = 24.0
NANOSECONDS_PER_MILLISECOND = 1_000_000.0


@dataclass(frozen=True, slots=True)
class TimingSummary:
    stage: str
    count: int
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True, slots=True)
class BaselineBytes:
    name: str
    row_group_rows: int
    sf1_return_bytes: int
    sf100_return_gb: float
    amplification_vs_selective: float
    notes: str


@dataclass(frozen=True, slots=True)
class ModuleSweepRow:
    baseline: str
    row_group_rows: int
    target_scan_time_ms: float
    sf100_return_gb: float
    required_gb_s_per_stack: float
    x16_modules_per_stack: int
    x64_modules_per_stack: int


def main() -> int:
    args = _parse_args()
    _require_cuda_if_requested(args.device)
    con = duckdb.connect(str(args.db))
    try:
        columns, duckdb_timing = _measure_duckdb_fetch(con, args)
        selected = _q6_mask(columns)
        compression = _compression_ratios(con, args.page_bytes, args.parquet_row_group_rows)
    finally:
        con.close()
    baselines = _build_baselines(columns, selected, compression, args)
    module_sweep = _module_sweep(baselines, _parse_floats(args.target_scan_time_ms), args)
    timings = [duckdb_timing, *_measure_q6_timing(columns, args)]
    _write_outputs(args.out_dir, columns, selected, compression, baselines, module_sweep, timings, args)
    print(args.out_dir)
    print(_summary_markdown(baselines, module_sweep, timings))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--page-bytes", type=int, default=DEFAULT_PAGE_BYTES)
    parser.add_argument("--rowid-bytes", type=int, default=DEFAULT_ROWID_BYTES)
    parser.add_argument("--stack-count", type=int, default=DEFAULT_STACK_COUNT)
    parser.add_argument("--sf-scale", type=float, default=DEFAULT_SF_SCALE)
    parser.add_argument("--grade-floor-gb-s", type=float, default=DEFAULT_GRADE_FLOOR_GB_S)
    parser.add_argument("--payload-efficiency", type=float, default=DEFAULT_PAYLOAD_EFFICIENCY)
    parser.add_argument("--target-scan-time-ms", default=DEFAULT_TARGET_MS)
    parser.add_argument("--row-group-rows", default=DEFAULT_ROW_GROUP_ROWS)
    parser.add_argument("--parquet-row-group-rows", type=int, default=16_384)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--hot-runs", type=int, default=DEFAULT_HOT_RUNS)
    return parser.parse_args()


def _load_q6_columns(con: duckdb.DuckDBPyConnection) -> dict[str, np.ndarray]:
    sql = """
    select
        l_discount::double as l_discount,
        l_extendedprice::double as l_extendedprice,
        l_quantity::double as l_quantity,
        strftime(l_shipdate, '%Y%m%d')::integer as l_shipdate
    from lineitem
    """
    return {name: values for name, values in con.execute(sql).fetchnumpy().items()}


def _q6_mask(columns: dict[str, np.ndarray]) -> np.ndarray:
    return (
        (columns["l_shipdate"] >= DATE_LOWER)
        & (columns["l_shipdate"] < DATE_UPPER)
        & (columns["l_discount"] >= DISCOUNT_LOWER)
        & (columns["l_discount"] <= DISCOUNT_UPPER)
        & (columns["l_quantity"] < QUANTITY_UPPER)
    )


def _compression_ratios(
    con: duckdb.DuckDBPyConnection,
    page_bytes: int,
    row_group_rows: int,
) -> dict[str, float]:
    ratios: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="tqg_q6_parquet_") as temp:
        temp_dir = Path(temp)
        for column in REQUIRED_COLUMNS:
            path = temp_dir / f"{column}.parquet"
            con.execute(_copy_column_sql(column, path, row_group_rows))
            rows = int(con.execute("select count(*) from lineitem").fetchone()[0])
            logical = _column_full_pages(rows, _column_width(column), page_bytes)
            ratios[column] = path.stat().st_size / logical
    return ratios


def _copy_column_sql(column: str, path: Path, row_group_rows: int) -> str:
    expression = f"strftime({column}, '%Y%m%d')::integer as {column}" if column == "l_shipdate" else column
    escaped = str(path).replace("'", "''")
    return (
        f"COPY (SELECT {expression} FROM lineitem) TO '{escaped}' "
        f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {row_group_rows})"
    )


def _build_baselines(
    columns: dict[str, np.ndarray],
    selected: np.ndarray,
    compression: dict[str, float],
    args: argparse.Namespace,
) -> list[BaselineBytes]:
    rows = _row_count(columns)
    selected_ids = np.flatnonzero(selected)
    selective = _selective_return_bytes(rows, int(selected.sum()), args.rowid_bytes)
    baselines = [_baseline("selective_return", 0, selective, selective, args, "HBF-side Q6 filter returns projected values + metadata")]
    full_columns = _full_required_column_bytes(rows, args.page_bytes)
    full = sum(full_columns.values())
    baselines.append(_baseline("full_required_columns", 0, full, selective, args, "columnar required-column page return"))
    baselines.append(_baseline("full_required_columns_zstd_proxy", 0, _compress_column_bytes(full_columns, compression), selective, args, "per-column Parquet ZSTD size proxy"))
    for group_rows in _parse_ints(args.row_group_rows):
        groups = _candidate_groups(columns, group_rows)
        zone_columns = _zone_required_column_bytes(rows, groups, args.page_bytes)
        late_columns = _late_materialization_column_bytes(rows, groups, selected_ids, args.page_bytes)
        zone = sum(zone_columns.values())
        late = sum(late_columns.values())
        baselines.extend(
            [
                _baseline("zone_map_required_columns", group_rows, zone, selective, args, "row-group min/max skipping, then return required columns"),
                _baseline("zone_map_late_materialization", group_rows, late, selective, args, "zone-map predicate columns + selected projected pages"),
                _baseline("zone_map_late_materialization_zstd_proxy", group_rows, _compress_column_bytes(late_columns, compression), selective, args, "late materialization with per-column Parquet ZSTD proxy"),
            ]
        )
    baselines.append(_baseline("generic_predicate_pushdown_oracle", 0, selective, selective, args, "same return bytes as selective; included as a kill baseline"))
    return baselines


def _baseline(name: str, group_rows: int, bytes_value: int, selective: int, args: argparse.Namespace, notes: str) -> BaselineBytes:
    return BaselineBytes(
        name=name,
        row_group_rows=group_rows,
        sf1_return_bytes=bytes_value,
        sf100_return_gb=bytes_value * args.sf_scale / 1_000_000_000,
        amplification_vs_selective=_safe_div(bytes_value, selective),
        notes=notes,
    )


def _full_required_column_bytes(rows: int, page_bytes: int) -> dict[str, int]:
    return {column: _column_full_pages(rows, _column_width(column), page_bytes) for column in REQUIRED_COLUMNS}


def _zone_required_column_bytes(rows: int, groups: list[tuple[int, int]], page_bytes: int) -> dict[str, int]:
    return {
        column: _column_pages_for_ranges(rows, _column_width(column), groups, page_bytes)
        for column in REQUIRED_COLUMNS
    }


def _late_materialization_column_bytes(
    rows: int,
    groups: list[tuple[int, int]],
    selected_ids: np.ndarray,
    page_bytes: int,
) -> dict[str, int]:
    values = {
        column: _column_pages_for_ranges(rows, _column_width(column), groups, page_bytes)
        for column in PREDICATE_COLUMNS
    }
    values["l_extendedprice"] = _column_pages_for_ids(rows, _column_width("l_extendedprice"), selected_ids, page_bytes)
    return values


def _candidate_groups(columns: dict[str, np.ndarray], group_rows: int) -> list[tuple[int, int]]:
    rows = _row_count(columns)
    ranges: list[tuple[int, int]] = []
    for start in range(0, rows, group_rows):
        end = min(start + group_rows, rows)
        if _group_can_match(columns, start, end):
            ranges.append((start, end))
    return ranges


def _group_can_match(columns: dict[str, np.ndarray], start: int, end: int) -> bool:
    dates = columns["l_shipdate"][start:end]
    discounts = columns["l_discount"][start:end]
    quantities = columns["l_quantity"][start:end]
    return bool(
        dates.max() >= DATE_LOWER
        and dates.min() < DATE_UPPER
        and discounts.max() >= DISCOUNT_LOWER
        and discounts.min() <= DISCOUNT_UPPER
        and quantities.min() < QUANTITY_UPPER
    )


def _column_pages_for_ranges(rows: int, width: int, ranges: list[tuple[int, int]], page_bytes: int) -> int:
    pages: set[int] = set()
    rows_per_page = max(page_bytes // width, 1)
    for start, end in ranges:
        if end <= start:
            continue
        pages.update(range(start // rows_per_page, (end - 1) // rows_per_page + 1))
    max_pages = math.ceil(rows / rows_per_page)
    return min(len(pages), max_pages) * page_bytes


def _column_pages_for_ids(rows: int, width: int, ids: np.ndarray, page_bytes: int) -> int:
    if ids.size == 0:
        return 0
    rows_per_page = max(page_bytes // width, 1)
    max_pages = math.ceil(rows / rows_per_page)
    return min(np.unique(ids // rows_per_page).size, max_pages) * page_bytes


def _compress_column_bytes(column_bytes: dict[str, int], compression: dict[str, float]) -> int:
    return math.ceil(sum(bytes_value * compression[column] for column, bytes_value in column_bytes.items()))


def _selective_return_bytes(rows: int, selected_rows: int, rowid_bytes: int) -> int:
    projected_row_bytes = sum(_column_width(column) for column in PROJECTED_COLUMNS)
    metadata = min(math.ceil(rows / 8), selected_rows * rowid_bytes)
    return (selected_rows * projected_row_bytes) + metadata


def _module_sweep(
    baselines: list[BaselineBytes],
    target_ms_values: tuple[float, ...],
    args: argparse.Namespace,
) -> list[ModuleSweepRow]:
    rows: list[ModuleSweepRow] = []
    for baseline in baselines:
        for target_ms in target_ms_values:
            required = baseline.sf100_return_gb / (target_ms / 1000.0) / args.stack_count
            x16 = _minimum_modules(required, 16, args).modules_per_stack
            x64 = _minimum_modules(required, 64, args).modules_per_stack
            rows.append(ModuleSweepRow(baseline.name, baseline.row_group_rows, target_ms, baseline.sf100_return_gb, required, x16, x64))
    return rows


def _measure_duckdb_fetch(
    con: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], TimingSummary]:
    for _ in range(args.warmup_runs):
        _load_q6_columns(con)
    columns: dict[str, np.ndarray] | None = None
    samples = []
    for _ in range(args.hot_runs):
        start = perf_counter_ns()
        columns = _load_q6_columns(con)
        samples.append((perf_counter_ns() - start) / NANOSECONDS_PER_MILLISECOND)
    if columns is None:
        raise ValueError("--hot-runs must be positive")
    return columns, _summarize("duckdb_required_column_fetchnumpy", samples)


def _measure_q6_timing(
    columns: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> list[TimingSummary]:
    timings = [_time_stage("cpu_tensor_wrap", args.hot_runs, lambda: _cpu_tensor_wrap(columns))]
    if args.device != "cuda":
        return timings
    cpu_tensors = _cpu_tensor_wrap(columns)
    timings.append(_time_h2d(cpu_tensors, args))
    gpu_tensors = {name: tensor.to(args.device) for name, tensor in cpu_tensors.items()}
    timings.append(_time_resident_kernel(gpu_tensors, args))
    return timings


def _cpu_tensor_wrap(columns: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        "l_discount": torch.as_tensor(columns["l_discount"], dtype=torch.float64),
        "l_extendedprice": torch.as_tensor(columns["l_extendedprice"], dtype=torch.float64),
        "l_quantity": torch.as_tensor(columns["l_quantity"], dtype=torch.float64),
        "l_shipdate": torch.as_tensor(columns["l_shipdate"], dtype=torch.int32),
    }


def _time_h2d(cpu_tensors: dict[str, torch.Tensor], args: argparse.Namespace) -> TimingSummary:
    def copy_once() -> dict[str, torch.Tensor]:
        return {name: tensor.to(args.device) for name, tensor in cpu_tensors.items()}

    for _ in range(args.warmup_runs):
        copy_once()
        torch.cuda.synchronize()
    samples = []
    for _ in range(args.hot_runs):
        torch.cuda.synchronize()
        start = perf_counter_ns()
        copy_once()
        torch.cuda.synchronize()
        samples.append((perf_counter_ns() - start) / NANOSECONDS_PER_MILLISECOND)
    return _summarize("h2d_required_columns", samples)


def _time_resident_kernel(gpu_tensors: dict[str, torch.Tensor], args: argparse.Namespace) -> TimingSummary:
    for _ in range(args.warmup_runs):
        _q6_resident_tensor(gpu_tensors)
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.hot_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _q6_resident_tensor(gpu_tensors)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return _summarize("gpu_resident_q6_plain_mask_sum", samples)


def _q6_resident_tensor(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = (
        (tensors["l_shipdate"] >= DATE_LOWER)
        & (tensors["l_shipdate"] < DATE_UPPER)
        & (tensors["l_discount"] >= DISCOUNT_LOWER)
        & (tensors["l_discount"] <= DISCOUNT_UPPER)
        & (tensors["l_quantity"] < QUANTITY_UPPER)
    )
    return (tensors["l_extendedprice"][mask] * tensors["l_discount"][mask]).sum()


def _time_stage(name: str, runs: int, func) -> TimingSummary:
    samples = []
    for _ in range(runs):
        start = perf_counter_ns()
        func()
        samples.append((perf_counter_ns() - start) / NANOSECONDS_PER_MILLISECOND)
    return _summarize(name, samples)


def _summarize(name: str, samples: list[float]) -> TimingSummary:
    return TimingSummary(name, len(samples), statistics.median(samples), statistics.mean(samples), min(samples), max(samples))


def _write_outputs(
    out_dir: Path,
    columns: dict[str, np.ndarray],
    selected: np.ndarray,
    compression: dict[str, float],
    baselines: list[BaselineBytes],
    sweep: list[ModuleSweepRow],
    timings: list[TimingSummary],
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(columns, selected, compression, args)
    _write_json(out_dir / "metadata.json", metadata)
    _write_json(out_dir / "baselines.json", [asdict(row) for row in baselines])
    _write_json(out_dir / "module_sweep.json", [asdict(row) for row in sweep])
    _write_json(out_dir / "timings.json", [asdict(row) for row in timings])
    (out_dir / "baselines.csv").write_text(_csv([asdict(row) for row in baselines], tuple(field.name for field in fields(BaselineBytes))) + "\n")
    (out_dir / "module_sweep.csv").write_text(_csv([asdict(row) for row in sweep], tuple(field.name for field in fields(ModuleSweepRow))) + "\n")
    (out_dir / "timings.csv").write_text(_csv([asdict(row) for row in timings], tuple(field.name for field in fields(TimingSummary))) + "\n")
    (out_dir / "summary.md").write_text(_summary_markdown(baselines, sweep, timings) + "\n")


def _metadata(
    columns: dict[str, np.ndarray],
    selected: np.ndarray,
    compression: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = _row_count(columns)
    return {
        "rows_sf1": rows,
        "selected_rows_sf1": int(selected.sum()),
        "selectivity": _safe_div(float(selected.sum()), rows),
        "compression_ratios": compression,
        "args": {key: _json_scalar(value) for key, value in vars(args).items()},
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }


def _summary_markdown(baselines: list[BaselineBytes], sweep: list[ModuleSweepRow], timings: list[TimingSummary]) -> str:
    lines = ["# Q6 HBF time gate", "", "## Baselines", "", "| baseline | group | SF100 GB | amp vs sel |", "| --- | ---: | ---: | ---: |"]
    for row in baselines:
        lines.append(f"| {row.name} | {row.row_group_rows} | {row.sf100_return_gb:.3f} | {row.amplification_vs_selective:.2f} |")
    lines.extend(["", "## x64 modules per stack", "", "| baseline | group | target ms | GB/s/stack | x64 |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in sweep:
        lines.append(f"| {row.baseline} | {row.row_group_rows} | {row.target_scan_time_ms:g} | {row.required_gb_s_per_stack:.1f} | {row.x64_modules_per_stack} |")
    lines.extend(["", "## Local timing", "", "| stage | median ms | mean ms |", "| --- | ---: | ---: |"])
    for row in timings:
        lines.append(f"| {row.stage} | {row.median_ms:.3f} | {row.mean_ms:.3f} |")
    return "\n".join(lines)


def _csv(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    return "\n".join([",".join(columns), *(",".join(_csv_value(row[column]) for column in columns) for row in rows)])


def _csv_value(value: Any) -> str:
    text = str(value)
    if any(char in text for char in ',\n"'):
        return '"' + text.replace('"', '""') + '"'
    return text




def _json_scalar(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value

def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _column_width(column: str) -> int:
    return 4 if column == "l_shipdate" else 8


def _column_full_pages(rows: int, width: int, page_bytes: int) -> int:
    return math.ceil(rows * width / page_bytes) * page_bytes


def _row_count(columns: dict[str, np.ndarray]) -> int:
    return int(next(iter(columns.values())).shape[0])


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


def _require_cuda_if_requested(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")


if __name__ == "__main__":
    raise SystemExit(main())
