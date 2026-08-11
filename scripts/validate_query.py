"""Validate supported SQL through a TQP frontend and PyTorch backend."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tpch_torch.backend.physical_partitionable import PartitionConfig
from tpch_torch.duckdb_bridge import connect_database
from tpch_torch.execution_mode import ExecutionMode, validate_execution_mode
from tpch_torch.ir import FrontendName
from tpch_torch.relational import SQLValidationResult
from tpch_torch.runner import load_sql, validate_sql_with_frontend
from tpch_torch.sql import get_tpch_query

DEFAULT_SQL_TOLERANCE = 1e-2
FIRST_TPCH_QUERY_ID = 1
LAST_TPCH_QUERY_ID = 22
ALL_TPCH_QUERY_IDS = tuple(range(FIRST_TPCH_QUERY_ID, LAST_TPCH_QUERY_ID + 1))
QueryLoader = Callable[[object, int], str]
QueryValidator = Callable[..., SQLValidationResult]
BatchProgress = Callable[["BatchValidationRecord"], None]


@dataclass(frozen=True)
class BatchValidationRecord:
    query_id: int
    ok: bool
    message: str
    row_count: int = 0
    max_abs_error: float | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate supported SQL against DuckDB")
    parser.add_argument("--db", type=Path, required=True, help="Input DuckDB database path")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query", type=int, help="TPC-H query number")
    source.add_argument("--queries", help="TPC-H query ids as comma-separated numbers or 'all'")
    source.add_argument("--sql", help="Inline SQL text")
    source.add_argument("--sql-file", type=Path, help="SQL file path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Execution device")
    parser.add_argument(
        "--frontend",
        choices=("sirius", "substrait"),
        default="sirius",
        help="TQP frontend used before PyTorch execution",
    )
    parser.add_argument("--keep-going", action="store_true", help="Continue batch validation after a query fails")
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
    parser.add_argument("--tolerance", type=float, default=DEFAULT_SQL_TOLERANCE)
    return parser


def parse_query_ids(raw: str) -> tuple[int, ...]:
    if raw == "all":
        return ALL_TPCH_QUERY_IDS
    query_ids = tuple(int(item) for item in raw.split(",") if item)
    if not query_ids:
        raise ValueError("at least one query id is required")
    return query_ids


def validate_queries(
    con: object,
    query_ids: tuple[int, ...],
    *,
    device: str,
    tolerance: float,
    keep_going: bool,
    frontend: FrontendName = "sirius",
    use_compressed_masks: bool = False,
    partition_config: PartitionConfig | None = None,
    execution_mode: ExecutionMode = "strict",
    load_query: QueryLoader = get_tpch_query,
    validator: QueryValidator = validate_sql_with_frontend,
    on_record: BatchProgress | None = None,
) -> list[BatchValidationRecord]:
    records: list[BatchValidationRecord] = []
    for query_id in query_ids:
        try:
            record = _validate_one_query(
                con,
                query_id,
                device=device,
                tolerance=tolerance,
                frontend=frontend,
                use_compressed_masks=use_compressed_masks,
                partition_config=partition_config,
                execution_mode=execution_mode,
                load_query=load_query,
                validator=validator,
            )
        except Exception as exc:
            if not keep_going:
                raise
            record = BatchValidationRecord(query_id=query_id, ok=False, message=str(exc))
            records.append(record)
            _emit_progress(record, on_record)
            continue
        records.append(record)
        _emit_progress(record, on_record)
    return records


def _validate_one_query(
    con: object,
    query_id: int,
    *,
    device: str,
    tolerance: float,
    frontend: FrontendName,
    load_query: QueryLoader,
    validator: QueryValidator,
    use_compressed_masks: bool,
    partition_config: PartitionConfig | None,
    execution_mode: ExecutionMode,
) -> BatchValidationRecord:
    sql = load_query(con, query_id)
    result = validator(
        con,
        sql,
        device=device,
        frontend=frontend,
        use_compressed_masks=use_compressed_masks,
        partition_config=partition_config,
        execution_mode=execution_mode,
    )
    result_query_id = result.query_id if result.query_id is not None else query_id
    if result.max_abs_error > tolerance:
        raise AssertionError(
            f"Q{result_query_id} validation failed: "
            f"max_abs_error={result.max_abs_error} tolerance={tolerance}"
        )
    return BatchValidationRecord(
        query_id=result_query_id,
        ok=True,
        message="validated",
        row_count=result.row_count,
        max_abs_error=result.max_abs_error,
    )


def main() -> None:
    args = build_parser().parse_args()
    con = connect_database(args.db)
    try:
        if args.queries is not None:
            printed_query_ids: set[int] = set()
            records = validate_queries(
                con,
                parse_query_ids(args.queries),
                device=args.device,
                tolerance=args.tolerance,
                keep_going=args.keep_going,
                frontend=args.frontend,
                use_compressed_masks=args.compressed_masks,
                partition_config=_partition_config(args),
                execution_mode=validate_execution_mode(args.execution_mode),
                on_record=_progress_printer(printed_query_ids),
            )
            _print_batch_records(
                [record for record in records if record.query_id not in printed_query_ids]
            )
            _raise_on_batch_failures(records)
            return
        sql = load_sql(con, query=args.query, sql=args.sql, sql_file=args.sql_file)
        result = validate_sql_with_frontend(
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
    if result.max_abs_error > args.tolerance:
        raise AssertionError(
            f"Q{result.query_id} validation failed: "
            f"max_abs_error={result.max_abs_error} tolerance={args.tolerance}"
        )
    print(
        f"validated query={result.query_id} rows={result.row_count} "
        f"max_abs_error={result.max_abs_error:.6g}"
    )


def _partition_config(args: argparse.Namespace) -> PartitionConfig | None:
    if args.partition_table is None and args.partition_chunk_size is None:
        return None
    if args.partition_table is None or args.partition_chunk_size is None:
        raise SystemExit("--partition-table and --partition-chunk-size must be provided together")
    return PartitionConfig(args.partition_table, args.partition_chunk_size)


def _print_batch_records(records: list[BatchValidationRecord]) -> None:
    for record in records:
        _print_batch_record(record)


def _print_batch_record(record: BatchValidationRecord) -> None:
    if record.ok:
        print(
            f"validated query={record.query_id} rows={record.row_count} "
            f"max_abs_error={record.max_abs_error:.6g}",
            flush=True,
        )
        return
    print(f"failed query={record.query_id} {record.message}", flush=True)


def _progress_printer(printed_query_ids: set[int]) -> BatchProgress:
    def print_record(record: BatchValidationRecord) -> None:
        printed_query_ids.add(record.query_id)
        _print_batch_record(record)

    return print_record


def _emit_progress(record: BatchValidationRecord, on_record: BatchProgress | None) -> None:
    if on_record is not None:
        on_record(record)


def _raise_on_batch_failures(records: list[BatchValidationRecord]) -> None:
    failed_query_ids = [record.query_id for record in records if not record.ok]
    if failed_query_ids:
        formatted = ",".join(str(query_id) for query_id in failed_query_ids)
        raise AssertionError(f"batch validation failed: Q{formatted}")


if __name__ == "__main__":
    main()
