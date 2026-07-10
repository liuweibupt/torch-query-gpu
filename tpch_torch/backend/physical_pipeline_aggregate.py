"""Aggregate batch operators for partitionable physical pipelines."""

from __future__ import annotations

from dataclasses import dataclass

from tpch_torch.backend.physical_aggregate import (
    aggregate_specs,
    execute_grouped_aggregate,
    execute_ungrouped_aggregate,
)
from tpch_torch.backend.physical_types import PhysicalTable
from tpch_torch.operator_graph import TQPOperatorNode


@dataclass(frozen=True)
class LocalAggregateBatchOperator:
    """Compute one partial aggregate table for each input batch."""

    child: object
    node: TQPOperatorNode

    def next_batch(self) -> PhysicalTable | None:
        batch = self.child.next_batch()
        if batch is None:
            return None
        specs = aggregate_specs(self.node, batch)
        group_exprs = _metadata_list(self.node, "Groups")
        if group_exprs:
            return execute_grouped_aggregate(batch, group_exprs, specs)
        return execute_ungrouped_aggregate(batch, specs)


def _metadata_list(node: TQPOperatorNode, key: str) -> tuple[str, ...]:
    value = node.metadata.get(key)
    if value is None or value == "":
        return ()
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),)
