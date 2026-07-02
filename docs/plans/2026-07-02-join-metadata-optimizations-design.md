# Join Metadata Optimizations Design

## Goal

Continue the paper-driven optimization roadmap by making the physical interpreter aware of sorted/unique column properties, then use that metadata to reduce redundant join, sort, and group-by work in the DuckDB physical-plan → PyTorch execution path. Correctness remains the primary requirement; performance improvements are measured but not required to match paper numbers in this batch.

## Current findings

- The TPC-H path is already end-to-end SQL driven: DuckDB emits JSON physical plans, the repository lowers them into `TQPOperatorGraph`, and the physical interpreter executes PyTorch tensor operators.
- Existing inner join code is already tensorized: it sorts the build key, probes with `torch.searchsorted`, and uses a duplicate-free fast path when the sorted build key is strictly increasing.
- The missing piece is persistent physical metadata. Each join currently rediscovers sortedness/uniqueness by scanning tensors; subsequent sort/group-by operators do not know when earlier operators already established order or uniqueness.
- Baseline tests in the current container failed before code changes because the cgroup is near its PID limit (`pids.current` was close to `pids.max`). DuckDB `dbgen` with the default thread count attempted to create many worker threads and reported `Resource temporarily unavailable`. DuckDB's Python `load_extension()` can surface the same resource error as `RuntimeError`. A later check also showed SQL `LOAD substrait` may abort in a `duckdb`-then-`torch` import order, so the safer fix is typed error wrapping for the Python API, not switching APIs.

## Architecture

### Metadata representation

Add immutable metadata fields to `PhysicalValue`:

- `sorted_non_decreasing`: the tensor is known to be sorted ascending with duplicates allowed.
- `unique`: the tensor is known to have no duplicate values.

Metadata lives on `PhysicalValue`, not on string column names, because aliases often point to the same value object (`col`, `table.col`, projection aliases, join-equivalent keys). This keeps alias propagation DRY and avoids a separate name-normalization registry.

### Propagation rules

- Scan-created `rowid` is sorted and unique.
- Fetched scan columns are tagged by a small catalog of TPC-H primary-key columns after validating the tensor once. Validation is explicit: sorted/unique flags are only set when tensor checks pass.
- `filter()` preserves sortedness and uniqueness because boolean masking keeps relative order and cannot introduce duplicates.
- Arbitrary `gather()` drops sortedness/uniqueness because indices may reorder or duplicate rows.
- `ORDER BY` marks the ordered key as sorted for ascending single-key sorts and drops unknown metadata for gathered payload columns.
- Group-by output keys are unique by construction; sorted `unique_consecutive` keys remain sorted.

### Join fast path

Introduce a small immutable `JoinKeyProperties` object and route `join_indices_for_conditions()` through value-aware helpers. For single-key equi joins, if the build-side key is known sorted and unique, the executor can directly `searchsorted` and skip:

- `torch.argsort()` on the build key;
- sortedness validation scan;
- duplicate expansion path.

If metadata is absent, the current generic tensor path remains the correctness path.

### DuckDB helper stability

Keep DuckDB helper failures visible, but make them typed and deterministic:

- keep DuckDB's Python extension API for Substrait loading because SQL `LOAD` can abort in one import-order edge case in this environment;
- wrap both `duckdb.Error` and `RuntimeError` from extension loading in `DuckDBSubstraitError` so tests can skip/report explicitly;
- configure DuckDB `dbgen` worker count through an explicit `TQG_DUCKDB_THREADS` environment variable defaulting to `1` for repository-managed test-data generation. This is documented as a container resource control, not an execution fallback.

## Testing strategy

- Add unit tests for metadata preservation/drop rules on `PhysicalValue` and `PhysicalTable`.
- Add join tests that prove the sorted/unique build key path skips `torch.argsort` while producing the same row-index pairs.
- Add DuckDB bridge tests for typed `RuntimeError` wrapping and helper thread configuration.
- Run focused tests first, then full `pytest` with `timeout 60`.
- Benchmark representative join-heavy TPC-H queries (Q14/Q19 if the existing SF=1 database is present) and record results in docs/README without overstating improvements.
