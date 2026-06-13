# Graph-Lowered Direct Primitives Design

## Goal

Keep SQL as the only user-facing input while making high-performance direct primitives the backend execution target. The execution contract is:

```text
SQL -> DuckDB/Sirius-like planner -> DuckDB JSON physical plan -> TQPOperatorGraph -> PyTorch physical/primitives backend
```

This explicitly excludes hand-written JSON inputs, DuckDB result-row fallback, and query-id Python template fallback.

## Architecture

The frontend continues to lower every supported query through DuckDB JSON physical plans into `TQPOperatorGraph`. The backend should prefer generic physical interpretation and optimized tensor primitives selected from graph/physical node shape, not from a standalone query template. Query-specific graph recipes remain only for complex TPC-H shapes not yet covered by the physical interpreter.

For this batch, Q6 moves from the default `query_id == 6` direct dispatch to the graph-lowered physical interpreter. The existing `--compressed-masks` Q6 experimental path remains explicit because it tests compressed mask primitives and is not yet represented by DuckDB physical metadata.

## Data Flow

Default Q6 path:

```text
TPC-H Q6 SQL
  -> compile_sirius_plan()
  -> lower_duckdb_json_to_operator_graph()
  -> PyTorchGraphExecutor._execute_tpch_graph()
  -> execute_physical_plan()
  -> SEQ_SCAN / PROJECTION / UNGROUPED_AGGREGATE tensor execution
```

Compressed Q6 path:

```text
TPC-H Q6 SQL + --compressed-masks
  -> same frontend graph admission
  -> explicit Q6 compressed mask primitive
```

## Error Handling

Unsupported DuckDB physical nodes continue to raise `UnsupportedPlanError`. The backend must not silently fall back to DuckDB rows or old `tpch_torch.queries.qXX` modules.

## Testing

- Add a regression test proving default Q6 calls `execute_physical_plan()` and does not call `_execute_q6_graph()`.
- Keep a regression test proving `use_compressed_masks=True` still reaches the compressed-mask primitive path.
- Validate Q6 through `scripts.validate_query` using `--query 6 --frontend sirius`.

## Documentation

Update README and architecture/roadmap docs to say Q1/Q6/Q12/Q14/Q19 use graph-lowered physical interpretation by default, with Q6 compressed masks as an explicit experimental variant.
