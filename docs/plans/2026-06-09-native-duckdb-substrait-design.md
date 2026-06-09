# Native DuckDB Substrait Expansion Design

## Goal

Follow the user-selected **B方案**: keep the unmodified TPC-H SQL path and wait for or build against native DuckDB Substrait support for `DELIM_JOIN` / `MARK`-style plans. The repository must not introduce canonical SQL rewrites to bypass DuckDB planner limitations.

The required execution contract remains:

```text
Original SQL / --query N
  → DuckDB get_substrait_json(original_sql)
  → Substrait plan analysis / dispatch
  → PyTorch tensor execution on CPU/GPU
  → optional DuckDB baseline validation
```

## Current evidence

Using this repository's supported DuckDB 1.2.x environment:

- DuckDB 1.2.0, 1.2.1, and 1.2.2 can load the `substrait` extension.
- Original TPC-H Q2, Q4, Q17, Q20, Q21, and Q22 fail at `get_substrait_json` with `DELIM_JOIN`.
- Original TPC-H Q16 fails at `get_substrait_json` with unsupported `MARK` join.
- `enable_optimizer=true` and `enable_optimizer=false` both fail for those seven queries.

A version probe against PyPI DuckDB 1.3.0 through 1.5.3 found no downloadable `substrait` extension package from the default or community extension repositories for `linux_amd64`; therefore those versions are not currently usable as a drop-in fix in this environment.

A source inspection of `substrait-io/duckdb-substrait-extension` at commit `29518b28d19532c659aa2a1271907659f5d8c7e3` shows that current source has explicit handling for `JoinType::MARK` and Substrait join enums include left/right mark, semi, anti, and single joins. It still does not expose a `LOGICAL_DELIM_JOIN` transform in `TransformOp` in the inspected source, so original correlated-query support remains a source-build/probing task rather than an assumed success.

## B方案 policy

1. Do not rewrite user SQL to avoid `DELIM_JOIN` or `MARK`.
2. Do not fabricate Substrait plans in this repository.
3. Do not use DuckDB result rows as the PyTorch result path.
4. Unsupported original SQL remains an explicit `DuckDBSubstraitError` until a real DuckDB Substrait export succeeds.
5. Add tooling/tests that make the native support state visible and reproducible.

## Architecture

### Native capability probe

Add a small, explicit capability layer that asks DuckDB to export each canonical TPC-H SQL query without rewrite. It records:

- DuckDB Python package version.
- Whether `substrait` extension loaded.
- Per-query `export_ok` boolean.
- Error class and first-line message for failures.
- Whether the PyTorch dispatcher has an executor for exported plans.

This layer is diagnostic only. It never changes SQL or falls back to alternate execution.

### Source-build hook for future native support

The repository should document a path for testing a locally built DuckDB Substrait extension:

```text
TQG_SUBSTRAIT_EXTENSION=/path/to/substrait.duckdb_extension
```

When set, `duckdb_bridge` should load exactly that extension path instead of installing the repository extension. If it fails to load, raise `DuckDBSubstraitError`. This keeps experiments with upstream/native support explicit and reproducible.

### PyTorch operator expansion after native export exists

Only after original SQL exports a real Substrait plan should we add PyTorch executors/operators for newly exported query shapes. Expected operators from TQP/TQP++ line:

- mark join output interpretation.
- left/right semi join.
- left/right anti join.
- single join / scalar subquery result.
- count distinct by group.
- grouped min/avg and scalar aggregate broadcast.
- composite-key joins and existence masks.

The executor can remain correctness-first and query-specific at first, but dispatch must be gated by successful native Substrait export of original SQL.

## User-facing behavior

- `tpch-torch-run --query 2` still fails today with `DuckDBSubstraitError` unless the loaded DuckDB Substrait extension can export original Q2.
- A new probe command should make that state obvious, e.g.:

```bash
tpch-torch-probe-substrait --db data/tpch_sf1.duckdb --queries all --json
```

- If `TQG_SUBSTRAIT_EXTENSION` is set, the probe and runner use that local native extension and report its path.

## Testing

1. Unit-test that local extension loading is explicit and errors cleanly when the path is invalid.
2. Unit-test that capability probing reports the current seven native export failures without treating them as test failures.
3. Regression-test that supported exported queries still validate through PyTorch.
4. Future TDD gate: when an upstream/local native extension exports a previously blocked query, add a failing end-to-end test for that query before implementing the PyTorch executor.

## Non-goals

- No SQL canonical rewrite pass.
- No custom Substrait plan generator.
- No vendored DuckDB C++ build inside this Python repository unless separately requested.
- No performance optimization until correctness and native Substrait export are established.
