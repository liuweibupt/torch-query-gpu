#!/usr/bin/env python3
"""Profile TQP/TPC-H scans for HBF selective-return workload screening."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import duckdb

from tpch_torch.runner import load_sql

DEFAULT_DB = Path("data/tpch_sf1.duckdb")
DEFAULT_OUT_DIR = Path("data/hbf_tpch_profile")
DEFAULT_QUERIES = "1,6,12,14,19"
DEFAULT_PAGE_BYTES = 4096
DEFAULT_ROWID_BYTES = 8
DEFAULT_STACK_COUNT = 3
DEFAULT_SF_SCALE = 100.0
DEFAULT_HBF_GB_S_PER_STACK = 1600.0
DEFAULT_TARGET_HBF_DUTY = 0.50
DEFAULT_GRADE_FLOOR_GB_S = 400.0
DEFAULT_PAYLOAD_EFFICIENCY = 0.85
DEFAULT_DATA_RATES_GT_S = (16.0, 32.0, 48.0, 64.0)
MODULE_WIDTHS = (16, 64)


@dataclass(frozen=True, slots=True)
class ScanProfile:
    query_id: int
    scan_index: int
    table: str
    table_rows: int
    selected_rows: int
    selectivity: float
    projected_columns: tuple[str, ...]
    predicate_columns: tuple[str, ...]
    required_columns: tuple[str, ...]
    filters: tuple[str, ...]
    filter_status: str
    projected_row_bytes: int
    required_row_bytes: int
    sf1_page_return_bytes: int
    sf1_selective_return_bytes: int
    sf1_metadata_bytes: int
    effective_amplification: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QueryProfile:
    query_id: int
    scan_count: int
    sf1_page_return_bytes: int
    sf1_selective_return_bytes: int
    scaled_page_return_bytes: float
    scaled_selective_return_bytes: float
    effective_amplification: float
    max_scan_selectivity: float
    min_scan_selectivity: float
    qps_at_target_hbf_duty: float
    page_return_gb_s_per_stack_at_target_duty: float
    selective_gb_s_per_stack_at_target_duty: float
    x16_page_modules_per_stack: int
    x16_selective_modules_per_stack: int
    x16_module_reduction: float
    x64_page_modules_per_stack: int
    x64_selective_modules_per_stack: int
    x64_module_reduction: float
    candidate_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModuleConfig:
    width: int
    data_rate_gt_s: float
    modules_per_stack: int
    payload_gb_s_per_stack: float


def main() -> int:
    args = _parse_args()
    con = duckdb.connect(str(args.db))
    try:
        scans: list[ScanProfile] = []
        queries: list[QueryProfile] = []
        for query_id in _parse_queries(args.queries):
            query_scans = _profile_query(con, query_id, args)
            scans.extend(query_scans)
            queries.append(_summarize_query(query_id, query_scans, args))
    finally:
        con.close()
    _write_outputs(args.out_dir, scans, queries)
    print(args.out_dir)
    print(_summary_markdown(queries))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--page-bytes", type=int, default=DEFAULT_PAGE_BYTES)
    parser.add_argument("--rowid-bytes", type=int, default=DEFAULT_ROWID_BYTES)
    parser.add_argument("--stack-count", type=int, default=DEFAULT_STACK_COUNT)
    parser.add_argument("--sf-scale", type=float, default=DEFAULT_SF_SCALE)
    parser.add_argument("--hbf-gb-s-per-stack", type=float, default=DEFAULT_HBF_GB_S_PER_STACK)
    parser.add_argument("--target-hbf-duty", type=float, default=DEFAULT_TARGET_HBF_DUTY)
    parser.add_argument("--grade-floor-gb-s", type=float, default=DEFAULT_GRADE_FLOOR_GB_S)
    parser.add_argument("--payload-efficiency", type=float, default=DEFAULT_PAYLOAD_EFFICIENCY)
    return parser.parse_args()


def _profile_query(
    con: duckdb.DuckDBPyConnection,
    query_id: int,
    args: argparse.Namespace,
) -> list[ScanProfile]:
    sql = load_sql(con, query=query_id, sql=None, sql_file=None)
    plan_json = con.execute("EXPLAIN (FORMAT JSON) " + sql).fetchone()[1]
    nodes = json.loads(plan_json)
    scan_nodes = [node for root in nodes for node in _walk_scan_nodes(root)]
    column_types = _all_column_types(con)
    return [
        _profile_scan(con, query_id, index, node, column_types, args)
        for index, node in enumerate(scan_nodes)
    ]


def _profile_scan(
    con: duckdb.DuckDBPyConnection,
    query_id: int,
    scan_index: int,
    node: dict[str, Any],
    column_types: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> ScanProfile:
    extra = node.get("extra_info", {})
    table = str(extra["Table"])
    table_rows = int(con.execute(f"select count(*) from {table}").fetchone()[0])
    filters, filter_status = _scan_filters(extra)
    selected_rows = _selected_rows(con, table, filters)
    projected = _scan_columns(extra.get("Projections"))
    predicates = _predicate_columns(filters, column_types[table])
    required = tuple(sorted(set(projected) | set(predicates)))
    projected_row_bytes = sum(_column_width(column_types[table][column]) for column in projected)
    required_row_bytes = sum(_column_width(column_types[table][column]) for column in required)
    page_bytes = _page_return_bytes(table_rows, required, column_types[table], args.page_bytes)
    selective_bytes, metadata_bytes = _selective_bytes(
        selected_rows,
        table_rows,
        projected_row_bytes,
        args.rowid_bytes,
    )
    return ScanProfile(
        query_id=query_id,
        scan_index=scan_index,
        table=table,
        table_rows=table_rows,
        selected_rows=selected_rows,
        selectivity=_safe_div(selected_rows, table_rows),
        projected_columns=projected,
        predicate_columns=predicates,
        required_columns=required,
        filters=filters,
        filter_status=filter_status,
        projected_row_bytes=projected_row_bytes,
        required_row_bytes=required_row_bytes,
        sf1_page_return_bytes=page_bytes,
        sf1_selective_return_bytes=selective_bytes,
        sf1_metadata_bytes=metadata_bytes,
        effective_amplification=_safe_div(page_bytes, selective_bytes),
    )


def _summarize_query(
    query_id: int,
    scans: list[ScanProfile],
    args: argparse.Namespace,
) -> QueryProfile:
    page_bytes = sum(scan.sf1_page_return_bytes for scan in scans)
    selective_bytes = sum(scan.sf1_selective_return_bytes for scan in scans)
    scaled_page = page_bytes * args.sf_scale
    scaled_selective = selective_bytes * args.sf_scale
    qps = _safe_div(
        args.target_hbf_duty * args.stack_count * args.hbf_gb_s_per_stack * 1_000_000_000,
        scaled_page,
    )
    page_gb_s = scaled_page * qps / args.stack_count / 1_000_000_000
    selective_gb_s = scaled_selective * qps / args.stack_count / 1_000_000_000
    x16_page = _minimum_modules(page_gb_s, 16, args)
    x16_selective = _minimum_modules(selective_gb_s, 16, args)
    x64_page = _minimum_modules(page_gb_s, 64, args)
    x64_selective = _minimum_modules(selective_gb_s, 64, args)
    amp = _safe_div(page_bytes, selective_bytes)
    return QueryProfile(
        query_id=query_id,
        scan_count=len(scans),
        sf1_page_return_bytes=page_bytes,
        sf1_selective_return_bytes=selective_bytes,
        scaled_page_return_bytes=scaled_page,
        scaled_selective_return_bytes=scaled_selective,
        effective_amplification=amp,
        max_scan_selectivity=max((scan.selectivity for scan in scans), default=0.0),
        min_scan_selectivity=min((scan.selectivity for scan in scans), default=0.0),
        qps_at_target_hbf_duty=qps,
        page_return_gb_s_per_stack_at_target_duty=page_gb_s,
        selective_gb_s_per_stack_at_target_duty=selective_gb_s,
        x16_page_modules_per_stack=x16_page.modules_per_stack,
        x16_selective_modules_per_stack=x16_selective.modules_per_stack,
        x16_module_reduction=_safe_div(x16_page.modules_per_stack, x16_selective.modules_per_stack),
        x64_page_modules_per_stack=x64_page.modules_per_stack,
        x64_selective_modules_per_stack=x64_selective.modules_per_stack,
        x64_module_reduction=_safe_div(x64_page.modules_per_stack, x64_selective.modules_per_stack),
        candidate_status=_candidate_status(amp, x64_page, x64_selective),
    )


def _walk_scan_nodes(node: dict[str, Any]):
    if node.get("name", "").strip() == "SEQ_SCAN":
        yield node
    for child in node.get("children", []):
        yield from _walk_scan_nodes(child)


def _all_column_types(con: duckdb.DuckDBPyConnection) -> dict[str, dict[str, str]]:
    tables = [row[0] for row in con.execute("show tables").fetchall()]
    return {
        table: {str(row[1]): str(row[2]) for row in con.execute(f"pragma table_info('{table}')").fetchall()}
        for table in tables
    }


def _scan_filters(extra: dict[str, Any]) -> tuple[tuple[str, ...], str]:
    raw = extra.get("Filters")
    if raw is None:
        return (), "none"
    items = raw if isinstance(raw, list) else [str(raw)]
    filters: list[str] = []
    stripped_optional = False
    for item in items:
        text = str(item).strip()
        if text.lower().startswith("optional:"):
            text = text.split(":", maxsplit=1)[1].strip()
            stripped_optional = True
        filters.append(text)
    return tuple(filters), "exact_optional_stripped" if stripped_optional else "exact"


def _selected_rows(con: duckdb.DuckDBPyConnection, table: str, filters: tuple[str, ...]) -> int:
    if not filters:
        return int(con.execute(f"select count(*) from {table}").fetchone()[0])
    where = " AND ".join(f"({item})" for item in filters)
    return int(con.execute(f"select count(*) from {table} where {where}").fetchone()[0])


def _scan_columns(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    items = value if isinstance(value, list) else [str(value)]
    return tuple(str(item).strip() for item in items if str(item).strip())


def _predicate_columns(filters: tuple[str, ...], column_types: dict[str, str]) -> tuple[str, ...]:
    text = " ".join(filters)
    return tuple(sorted(column for column in column_types if re.search(rf"\b{column}\b", text)))


def _column_width(duckdb_type: str) -> int:
    normalized = duckdb_type.upper()
    if normalized == "DATE":
        return 4
    if normalized in {"BOOLEAN", "BOOL"}:
        return 1
    return 8


def _page_return_bytes(
    row_count: int,
    columns: tuple[str, ...],
    column_types: dict[str, str],
    page_bytes: int,
) -> int:
    return sum(
        math.ceil(row_count * _column_width(column_types[column]) / page_bytes) * page_bytes
        for column in columns
    )


def _selective_bytes(
    selected_rows: int,
    table_rows: int,
    projected_row_bytes: int,
    rowid_bytes: int,
) -> tuple[int, int]:
    if projected_row_bytes == 0:
        return 0, 0
    bitmap_bytes = math.ceil(table_rows / 8)
    rowid_bytes_total = selected_rows * rowid_bytes
    metadata_bytes = min(bitmap_bytes, rowid_bytes_total)
    return selected_rows * projected_row_bytes + metadata_bytes, metadata_bytes


def _minimum_modules(required_gb_s: float, width: int, args: argparse.Namespace) -> ModuleConfig:
    required = max(required_gb_s, args.grade_floor_gb_s)
    best: ModuleConfig | None = None
    for rate in DEFAULT_DATA_RATES_GT_S:
        payload = rate / 8.0 * width * args.payload_efficiency
        modules = math.ceil(required / payload)
        config = ModuleConfig(width, rate, modules, payload * modules)
        if best is None or _module_key(config) < _module_key(best):
            best = config
    if best is None:
        raise RuntimeError("module search failed")
    return best


def _module_key(config: ModuleConfig) -> tuple[int, int, float]:
    return (config.modules_per_stack, config.width * config.modules_per_stack, config.data_rate_gt_s)


def _candidate_status(amp: float, page: ModuleConfig, selective: ModuleConfig) -> str:
    if amp < 16.0:
        return "reject_low_amplification"
    if page.modules_per_stack <= selective.modules_per_stack:
        return "reject_no_module_crossing"
    return "candidate_trace_followup"


def _write_outputs(out_dir: Path, scans: list[ScanProfile], queries: list[QueryProfile]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_dicts = [scan.to_dict() for scan in scans]
    query_dicts = [query.to_dict() for query in queries]
    (out_dir / "scan_profiles.json").write_text(json.dumps(scan_dicts, indent=2) + "\n")
    (out_dir / "query_profiles.json").write_text(json.dumps(query_dicts, indent=2) + "\n")
    (out_dir / "scan_profiles.csv").write_text(_csv(scan_dicts, tuple(field.name for field in fields(ScanProfile))) + "\n")
    (out_dir / "query_profiles.csv").write_text(_csv(query_dicts, tuple(field.name for field in fields(QueryProfile))) + "\n")
    (out_dir / "summary.md").write_text(_summary_markdown(queries) + "\n")


def _summary_markdown(queries: list[QueryProfile]) -> str:
    headers = ("Q", "scans", "amp", "SF100 pageGB", "SF100 selGB", "qps@50%duty", "x16", "x64", "status")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for query in queries:
        values = (
            str(query.query_id),
            str(query.scan_count),
            f"{query.effective_amplification:.2f}",
            f"{query.scaled_page_return_bytes / 1_000_000_000:.2f}",
            f"{query.scaled_selective_return_bytes / 1_000_000_000:.2f}",
            f"{query.qps_at_target_hbf_duty:.1f}",
            f"{query.x16_page_modules_per_stack}->{query.x16_selective_modules_per_stack}",
            f"{query.x64_page_modules_per_stack}->{query.x64_selective_modules_per_stack}",
            query.candidate_status,
        )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _csv(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(_csv_value(row[column]) for column in columns))
    return "\n".join(lines)


def _csv_value(value: Any) -> str:
    if isinstance(value, tuple):
        text = ";".join(str(item) for item in value)
    else:
        text = str(value)
    if any(char in text for char in ',\n"'):
        return '"' + text.replace('"', '""') + '"'
    return text


def _parse_queries(value: str) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        return tuple(range(1, 23))
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


if __name__ == "__main__":
    raise SystemExit(main())
