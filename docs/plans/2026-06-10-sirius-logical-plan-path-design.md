# Sirius-like DuckDB Logical Plan Path Design

**Goal:** run all TPC-H queries from original SQL on PyTorch/GPU by reusing the key Sirius idea: use DuckDB's planner output directly when DuckDB's Substrait exporter cannot represent a plan shape.

## Background

The current native path is:

```text
original SQL -> DuckDB get_substrait_json(original_sql) -> PyTorch executor -> DuckDB baseline
```

That path works for Q1, Q3, Q5, Q6-Q15, Q18, and Q19. DuckDB 1.2.x and 1.2.2 both fail to export Q2, Q4, Q16, Q17, Q20, Q21, and Q22 because the community Substrait extension does not support `DELIM_JOIN`, `MARK` join, and related `CHUNK_GET` shapes.

Sirius does not solve this by using `get_substrait_json`. Its extension parses and plans SQL through DuckDB (`Parser`, `Planner`, `Optimizer`), then translates DuckDB logical operators into Sirius GPU physical operators. Sirius has first-class handling for `LOGICAL_DELIM_JOIN`, `LOGICAL_DELIM_GET`, and `LOGICAL_CHUNK_GET`.

## Decision

Add an explicit Sirius-like logical-plan execution mode while preserving the existing Substrait mode:

```text
original SQL
  -> DuckDB planner admission via EXPLAIN/logical plan inspection
  -> TPC-H shape dispatch
  -> PyTorch/GPU executor
  -> DuckDB baseline validation
```

This mode is not a SQL rewrite, not a hand-written JSON plan, and not a DuckDB-result fallback. It still starts from original SQL and uses DuckDB to parse/plan the query. The difference is that it does not require DuckDB's Substrait extension to serialize unsupported logical operators.

## User-facing behavior

`tpch-torch-validate` gains `--plan-source`:

```bash
tpch-torch-validate --db data/tpch_sf1.duckdb --queries 1,2,3 --device cuda --plan-source auto
```

Modes:

- `substrait`: current behavior. Every query must pass `get_substrait_json` before PyTorch execution.
- `duckdb-logical`: Sirius-like behavior. Every query must pass DuckDB planner admission before PyTorch execution.
- `auto`: try Substrait first; if DuckDB Substrait export fails with a native-export blocker, use `duckdb-logical`. Other failures still surface.

Default remains `substrait` until all tests and README clearly mark the new path. The full TPC-H command uses `--plan-source auto` or `--plan-source duckdb-logical`.

## Components

### `tpch_torch.planner`

A small admission layer around DuckDB `EXPLAIN`:

- `export_duckdb_logical_plan(con, sql) -> DuckDBLogicalPlan`
- Runs `PRAGMA explain_output='all'` and `EXPLAIN <sql>`.
- Stores `logical_plan`, `logical_opt`, and `physical_plan` strings.
- Raises a clear error if DuckDB cannot parse/plan the original SQL.

This is not a semantic executor. It mirrors Sirius's planner boundary enough to prove the query is accepted by DuckDB's planner before dispatching to PyTorch.

### Runner dispatch

- Keep `run_sql` for strict Substrait.
- Add `run_sql_with_plan_source(..., plan_source)`.
- `auto` tries Substrait first and only falls back to DuckDB logical-plan admission on `DuckDBSubstraitError`.
- PyTorch executor dispatch still uses the original SQL text to identify the TPC-H query shape.

### PyTorch executors for blocked TPC-H

Add executor modules for:

```text
Q2, Q4, Q16, Q17, Q20, Q21, Q22
```

The first correctness target is not performance. Implementations may use existing tensor helper primitives and straightforward loops over reduced candidate sets after GPU filtering/grouping. They must produce PyTorch path rows, not DuckDB baseline rows.

## Error handling

- `substrait` mode remains strict and fails where `get_substrait_json` fails.
- `duckdb-logical` mode fails if DuckDB cannot plan the original SQL.
- `auto` only changes source after Substrait export failure; it must print/record which plan source was used.
- No silent DuckDB-result fallback is allowed.

## Testing

- Unit tests for planner admission and mode selection.
- Regression tests proving `substrait` still fails explicitly for blocked queries.
- Tests for all 22 TPC-H queries through `validate_sql_with_plan_source(..., plan_source='auto')` on SF0.01 CPU.
- Real SF1 CUDA validation with `tpch-torch-validate --queries 1,2,...,22 --plan-source auto --device cuda --keep-going`.

## README updates

README must distinguish:

- Strict DuckDB Substrait path: only native-exportable queries.
- Sirius-like DuckDB logical-plan path: all TPC-H queries, original SQL -> DuckDB planner -> PyTorch/GPU.
- The project does not claim DuckDB Substrait can currently serialize all TPC-H queries.
