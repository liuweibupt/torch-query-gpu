"""PyTorch backend for executing TQP plans."""

from __future__ import annotations

from typing import Any

import duckdb

from tpch_torch.backend.graph import PyTorchGraphExecutor
from tpch_torch.backend.physical_chunked import ScanChunkConfig
from tpch_torch.backend.physical_partitionable import PartitionConfig
from tpch_torch.backend.universal import execute_universal_sql
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.execution_mode import ExecutionMode, validate_execution_mode
from tpch_torch.ir import TQPPlan


class PyTorchBackend:
    """Execute internal TQP plans with PyTorch tensor query operators."""

    def execute(
        self,
        con: duckdb.DuckDBPyConnection,
        plan: TQPPlan,
        device: str = "cpu",
        use_compressed_masks: bool = False,
        partition_config: PartitionConfig | None = None,
        scan_chunk_config: ScanChunkConfig | None = None,
        execution_mode: ExecutionMode = "strict",
    ) -> list[dict[str, Any]]:
        mode = validate_execution_mode(execution_mode)
        if mode == "universal":
            return self._execute_universal(
                con,
                plan,
                device,
                use_compressed_masks,
                partition_config,
                scan_chunk_config,
            )
        executor_kwargs = {
            "device": device,
            "use_compressed_masks": use_compressed_masks,
            "partition_config": partition_config,
        }
        if scan_chunk_config is not None:
            executor_kwargs["scan_chunk_config"] = scan_chunk_config
        if plan.operator_graph is not None:
            return PyTorchGraphExecutor().execute(con, plan, **executor_kwargs)
        if plan.query_id is not None:
            raise UnsupportedPlanError(
                f"TPC-H Q{plan.query_id} requires a frontend-lowered TQP operator graph"
            )
        return PyTorchGraphExecutor().execute(con, plan, **executor_kwargs)

    def _execute_universal(
        self,
        con: duckdb.DuckDBPyConnection,
        plan: TQPPlan,
        device: str,
        use_compressed_masks: bool,
        partition_config: PartitionConfig | None,
        scan_chunk_config: ScanChunkConfig | None,
    ) -> list[dict[str, Any]]:
        try:
            return self.execute(
                con,
                plan,
                device=device,
                use_compressed_masks=use_compressed_masks,
                partition_config=partition_config,
                scan_chunk_config=scan_chunk_config,
                execution_mode="strict",
            )
        except UnsupportedPlanError as exc:
            has_runtime_transform = (
                use_compressed_masks
                or partition_config is not None
                or scan_chunk_config is not None
            )
            if has_runtime_transform:
                raise UnsupportedPlanError(
                    "universal compatibility materialization cannot be combined "
                    "with compressed, partition, or scan chunk modes"
                ) from exc
            return execute_universal_sql(con, plan.source_sql, device=device)
