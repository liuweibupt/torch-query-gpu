# torch-query-gpu

A minimal TQP-style prototype for analytical query execution on PyTorch tensors.
The default execution path is now a clean TQP/Sirius-style compiler stack:

```text
SQL / --query N
  -> Sirius-like DuckDB frontend
       DuckDB parses, binds, plans, and optimizes the original SQL
  -> TQP IR
       internal plan object shared by frontends and backends
  -> PyTorch backend
       correctness-first tensor operators on CPU or CUDA
  -> optional DuckDB baseline validation
```

Substrait is no longer the default execution path. It remains available as an
experimental strict frontend for queries DuckDB's Substrait exporter can emit:

```text
SQL / --query N
  -> DuckDB get_substrait_json(original_sql)
  -> TQP IR carrying the real Substrait JSON
  -> PyTorch backend
```

The project does **not** rewrite SQL to avoid planner limitations, does **not**
fabricate Substrait JSON, and does **not** use DuckDB result rows as PyTorch
output. DuckDB rows are used only as the validation baseline.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

## Run tests

```bash
timeout 60 python -m pytest -q
```

## Generate TPC-H data

```bash
# SF1 by default.
tpch-torch-gen-sf1 --db data/tpch_sf1.duckdb --sf 1
```

## End-to-end SQL commands

The generic commands read SQL directly from `--query`, `--sql`, or
`--sql-file`. They do not require manually exported JSON.

Default Sirius-like frontend:

```bash
tpch-torch-run --db data/tpch_sf1.duckdb --query 21 --device cuda --frontend sirius
tpch-torch-validate --db data/tpch_sf1.duckdb --query 21 --device cuda --frontend sirius
```

All TPC-H queries through the default clean path:

```bash
tpch-torch-validate \
  --db data/tpch_sf1.duckdb \
  --queries all \
  --device cuda \
  --frontend sirius \
  --keep-going
```

Strict Substrait frontend for DuckDB-exportable queries:

```bash
tpch-torch-run --db data/tpch_sf1.duckdb --query 6 --device cuda --frontend substrait --json
tpch-torch-validate --db data/tpch_sf1.duckdb --query 6 --device cuda --frontend substrait
```

Compatibility aliases remain available:

```text
--plan-source duckdb-logical  ->  --frontend sirius
--plan-source substrait       ->  --frontend substrait
--plan-source auto            ->  --frontend auto
```

Use `--device cpu` on machines without CUDA. If `--device cuda` is requested on a
CPU-only machine, the runner raises an explicit error.

Validation compares PyTorch rows with DuckDB's result for the same original SQL.
The default absolute tolerance is `1e-2`, which covers the small
accumulation-order differences between DuckDB decimals and PyTorch reductions at
SF1. Use `--tolerance` to make the check stricter or looser.

## TPC-H support matrix

| Query set | Default Sirius-like frontend | Strict DuckDB Substrait frontend | PyTorch backend |
| --- | --- | --- | --- |
| Q1, Q3, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, Q13, Q14, Q15, Q18, Q19 | yes | yes | yes |
| Q2, Q4, Q16, Q17, Q20, Q21, Q22 | yes | blocked in DuckDB 1.2.x Substrait export | yes |

Current all-query command:

```bash
tpch-torch-validate --db data/tpch_sf1.duckdb --queries all --device cuda --frontend sirius --keep-going
```

The all-query path is correctness-first. It prioritizes keeping the full
DuckDB-admitted SQL -> TQP IR -> PyTorch/GPU execution chain runnable over
performance.

## Architecture

### Frontends

Frontends compile original SQL into `TQPPlan`:

- `sirius`: DuckDB Parser/Planner/Optimizer admission via `EXPLAIN`; this is the
  default and covers Q1-Q22.
- `substrait`: DuckDB's real `get_substrait_json()` export; this remains a strict
  experimental frontend.
- `auto`: tries strict Substrait first and falls back to the Sirius-like frontend
  only when DuckDB's Substrait exporter raises `DuckDBSubstraitError`.

### TQP IR

`TQPPlan` is the internal plan object between frontend and backend. The first IR
version records the source SQL, query id, frontend, DuckDB plan metadata, and
optional real Substrait JSON. It is intentionally small so the project can keep
TPC-H correctness while evolving toward an operator graph IR.

### PyTorch backend

`PyTorchBackend` executes `TQPPlan` with existing correctness-first tensor
operators in `tpch_torch/queries/q01.py` through `q22.py`. This keeps backend
execution separate from SQL admission.

## Native DuckDB Substrait policy (B方案)

For strict Substrait experiments, this repository still follows the native
DuckDB Substrait path selected for B方案:

```text
original SQL / --query N
  -> DuckDB get_substrait_json(original_sql)
  -> TQP IR carrying the real Substrait JSON
  -> PyTorch tensor execution
```

If `--frontend substrait` is selected and DuckDB cannot export the original SQL,
the command fails with `DuckDBSubstraitError`.

Probe the native export state explicitly:

```bash
tpch-torch-probe-substrait --db data/tpch_sf1.duckdb --queries all --json
tpch-torch-probe-substrait --db data/tpch_sf1.duckdb --queries 2,4,16
```

As of DuckDB 1.2.x, original TPC-H Q2, Q4, Q16, Q17, Q20, Q21, and Q22 are
native-export blocked (`DELIM_JOIN` or `MARK` join). The default Sirius-like
frontend exists specifically so the clean TQP path is not blocked by that
exporter limitation.

If you build or obtain a newer native DuckDB Substrait extension, test it without
changing SQL by setting:

```bash
export TQG_SUBSTRAIT_EXTENSION=/path/to/substrait.duckdb_extension
```

When this variable is set, the bridge loads that exact extension path. Missing
or unloadable paths are hard errors.

## Legacy Q1 commands

The original Q1-only commands remain available for focused Substrait compiler
experiments:

```bash
# Export DuckDB's real Substrait JSON plan for Q1.
tpch-torch-export-q1-substrait --db data/tpch_sf1.duckdb --out data/q1_substrait.json

# Run Q1 through the legacy Substrait -> PyTorch path.
tpch-torch-run-q1 --db data/tpch_sf1.duckdb --device cuda

# Validate PyTorch output against DuckDB.
tpch-torch-validate-q1 --db data/tpch_sf1.duckdb --device cuda
```

Pre-exported JSON is only a debugging/caching aid for these legacy Q1 commands;
the generic runner performs SQL -> TQP frontend -> TQP IR -> PyTorch backend in
one command.
