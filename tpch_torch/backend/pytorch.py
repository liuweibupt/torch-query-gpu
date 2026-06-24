"""PyTorch backend for executing TQP plans."""

from __future__ import annotations

from typing import Any

import duckdb

from tpch_torch.backend.graph import PyTorchGraphExecutor
from tpch_torch.backend.physical_partitionable import PartitionConfig
from tpch_torch.errors import UnsupportedPlanError
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
    ) -> list[dict[str, Any]]:
        if plan.operator_graph is not None:
            return PyTorchGraphExecutor().execute(
                con,
                plan,
                device=device,
                use_compressed_masks=use_compressed_masks,
                partition_config=partition_config,
            )
        if plan.query_id is not None:
            raise UnsupportedPlanError(
                f"TPC-H Q{plan.query_id} requires a frontend-lowered TQP operator graph"
            )
        return PyTorchGraphExecutor().execute(
            con,
            plan,
            device=device,
            use_compressed_masks=use_compressed_masks,
            partition_config=partition_config,
        )
