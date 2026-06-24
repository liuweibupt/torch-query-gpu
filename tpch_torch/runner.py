"""SQL -> TQP frontend -> PyTorch backend execution dispatch."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import duckdb
import torch

from tpch_torch.backend import PyTorchBackend
from tpch_torch.backend.physical_partitionable import PartitionConfig
from tpch_torch.frontend import compile_sirius_plan, compile_substrait_plan
from tpch_torch.ir import FrontendName, TQPPlan
from tpch_torch.relational import QueryResult, SQLValidationResult, compare_rows, run_duckdb_sql
from tpch_torch.sql import get_tpch_query


def load_sql(
    con: duckdb.DuckDBPyConnection,
    query: int | None = None,
    sql: str | None = None,
    sql_file: Path | None = None,
) -> str:
    sources = [query is not None, sql is not None, sql_file is not None]
    if sum(sources) != 1:
        raise ValueError("exactly one of query, sql, or sql_file is required")
    if query is not None:
        return get_tpch_query(con, query)
    if sql is not None:
        return sql
    if sql_file is None:
        raise ValueError("sql_file is required")
    return sql_file.read_text()


def run_sql(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    device: str = "cpu",
    use_compressed_masks: bool = False,
    partition_config: PartitionConfig | None = None,
) -> QueryResult:
    return run_sql_with_frontend(
        con,
        sql,
        device=device,
        frontend="sirius",
        use_compressed_masks=use_compressed_masks,
        partition_config=partition_config,
    )


def run_sql_with_frontend(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    device: str = "cpu",
    frontend: FrontendName = "sirius",
    use_compressed_masks: bool = False,
    partition_config: PartitionConfig | None = None,
) -> QueryResult:
    _validate_device(device)
    plan = compile_tqp_plan(con, sql, frontend)
    rows = PyTorchBackend().execute(
        con,
        plan,
        device=device,
        use_compressed_masks=use_compressed_masks,
        partition_config=partition_config,
    )
    return QueryResult(query_id=plan.query_id, rows=rows)


def validate_sql(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    device: str = "cpu",
    use_compressed_masks: bool = False,
    partition_config: PartitionConfig | None = None,
) -> SQLValidationResult:
    return validate_sql_with_frontend(
        con,
        sql,
        device=device,
        frontend="sirius",
        use_compressed_masks=use_compressed_masks,
        partition_config=partition_config,
    )


def validate_sql_with_frontend(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    device: str = "cpu",
    frontend: FrontendName = "sirius",
    use_compressed_masks: bool = False,
    partition_config: PartitionConfig | None = None,
) -> SQLValidationResult:
    result = run_sql_with_frontend(
        con,
        sql,
        device=device,
        frontend=frontend,
        use_compressed_masks=use_compressed_masks,
        partition_config=partition_config,
    )
    duckdb_rows = run_duckdb_sql(con, sql)
    max_abs_error = compare_rows(duckdb_rows, result.rows)
    return SQLValidationResult(
        query_id=result.query_id,
        row_count=len(duckdb_rows),
        max_abs_error=max_abs_error,
        duckdb_rows=duckdb_rows,
        pytorch_rows=result.rows,
    )


def timed_run_sql(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    device: str = "cpu",
    frontend: FrontendName = "sirius",
    use_compressed_masks: bool = False,
    partition_config: PartitionConfig | None = None,
) -> tuple[QueryResult, float]:
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = run_sql_with_frontend(
            con,
            sql,
            device=device,
            frontend=frontend,
            use_compressed_masks=use_compressed_masks,
            partition_config=partition_config,
        )
        end.record()
        torch.cuda.synchronize()
        return result, float(start.elapsed_time(end))
    start_time = perf_counter()
    result = run_sql_with_frontend(
        con,
        sql,
        device=device,
        frontend=frontend,
        use_compressed_masks=use_compressed_masks,
        partition_config=partition_config,
    )
    return result, (perf_counter() - start_time) * 1000.0


def compile_tqp_plan(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    frontend: FrontendName = "sirius",
) -> TQPPlan:
    if frontend == "sirius":
        return compile_sirius_plan(con, sql)
    if frontend == "substrait":
        return compile_substrait_plan(con, sql)
    raise ValueError(f"unknown frontend: {frontend}")


def _validate_device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
