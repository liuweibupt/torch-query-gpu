# Scan Chunk Execution Design

## Goal

Add explicit scan-level chunk execution on top of the TensorRecordBatch-backed physical runtime.

## Current State

`fetch_physical_table_chunks()` can already fetch a table as multiple `TensorRecordBatch` chunks, and `PhysicalPlanExecutor` can execute a whole physical graph over a bounded `scan_range`. The missing piece is an explicit public execution config that repeatedly runs safe physical plans over ranges and preserves configured chunk metadata in scan batches.

## Chosen Design

Introduce `ScanChunkConfig(table, chunk_size)` separately from `PartitionConfig`.

- `PartitionConfig` remains the CoddSpeed-style aggregate fragment path with partial aggregate merge.
- `ScanChunkConfig` is scan/filter/project only in the first implementation.
- Unsupported plan shapes such as join, aggregate, sort, limit, CTE, and delim joins fail with `UnsupportedPlanError` instead of silently running the whole table.
- Each chunk reuses `PhysicalPlanExecutor(scan_ranges=...)` so SQL still goes through Sirius/DuckDB physical graph lowering and PyTorch physical operators.
- `PhysicalPlanExecutor` receives optional scan chunk-size metadata so `TensorRecordBatch.batch_meta.chunk_size/chunk_index/source_offset` represent the configured scan chunk rather than just the local slice length.

## Data Flow

```text
SQL -> Sirius physical graph -> PyTorchGraphExecutor(scan_chunk_config)
    -> execute_chunked_physical_plan
       -> row_ranges(table_count, chunk_size)
       -> PhysicalPlanExecutor(scan_ranges={table: (start, end)}, scan_chunk_sizes={table: chunk_size})
       -> scan/filter/project physical operators
       -> concatenate output rows in scan chunk order
```

## Explicit Limits

This is not a general streaming/chunked relational engine yet. It deliberately rejects operators that require global semantics across chunks unless a dedicated merge strategy exists. Aggregates should use `PartitionConfig`; joins need a chunk-aware join strategy before enabling.

## Test Plan

- RED/GREEN unit tests for `ScanChunkConfig` validation.
- RED/GREEN integration test for single-table scan/filter/project, verifying output equality and scan metadata chunk size/index/source offset passed to `fetch_physical_table`.
- RED/GREEN tests that aggregate and join queries are rejected explicitly.
- Regression test that existing `PartitionConfig` passes configured chunk metadata to scans.
