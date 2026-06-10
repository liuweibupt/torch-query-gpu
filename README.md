# torch-query-gpu

A minimal TQP-style prototype for analytical query execution on PyTorch tensors.
The project now has two explicit SQL admission paths before PyTorch execution:

```text
Strict DuckDB Substrait path
  original SQL / --query N
  -> DuckDB get_substrait_json(original_sql)
  -> Substrait plan analysis / dispatch
  -> PyTorch tensor operators on CPU or CUDA
  -> optional DuckDB baseline validation

Sirius-like DuckDB logical-plan path
  original SQL / --query N
  -> DuckDB Parser/Planner/Optimizer admission via EXPLAIN
  -> TPC-H query dispatch
  -> PyTorch tensor operators on CPU or CUDA
  -> optional DuckDB baseline validation
```

The second path mirrors the part of Sirius that avoids relying on DuckDB's
Substrait exporter for queries containing `DELIM_JOIN`, `MARK`, or `CHUNK_GET`:
DuckDB must still parse and plan the original SQL, but the blocked query is not
claimed to have a DuckDB Substrait export.

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

Strict Substrait for DuckDB-exportable queries:

```bash
tpch-torch-run --db data/tpch_sf1.duckdb --query 6 --device cuda --plan-source substrait --json
tpch-torch-validate --db data/tpch_sf1.duckdb --query 6 --device cuda --plan-source substrait
```

Sirius-like logical-plan admission for Substrait-blocked queries:

```bash
tpch-torch-run --db data/tpch_sf1.duckdb --query 21 --device cuda --plan-source duckdb-logical
tpch-torch-validate --db data/tpch_sf1.duckdb --query 21 --device cuda --plan-source duckdb-logical
```

Automatic mode tries strict Substrait first and falls back to DuckDB logical-plan
admission only when DuckDB's Substrait exporter raises `DuckDBSubstraitError`:

```bash
tpch-torch-validate \
  --db data/tpch_sf1.duckdb \
  --queries all \
  --device cuda \
  --plan-source auto \
  --keep-going
```

Use `--device cpu` on machines without CUDA. If `--device cuda` is requested on a
CPU-only machine, the runner raises an explicit error.

Validation compares PyTorch rows with DuckDB's result for the same original SQL.
The default absolute tolerance is `1e-2`, which covers the small
accumulation-order differences between DuckDB decimals and PyTorch reductions at
SF1. Use `--tolerance` to make the check stricter or looser.

## TPC-H support matrix

| Query set | Strict DuckDB Substrait | Sirius-like DuckDB logical-plan admission | PyTorch executor |
| --- | --- | --- | --- |
| Q1, Q3, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, Q13, Q14, Q15, Q18, Q19 | yes | yes | yes |
| Q2, Q4, Q16, Q17, Q20, Q21, Q22 | blocked in DuckDB 1.2.x Substrait export | yes | yes |

Current all-query command:

```bash
tpch-torch-validate --db data/tpch_sf1.duckdb --queries all --device cuda --plan-source auto --keep-going
```

The all-query path is correctness-first. It prioritizes keeping the full
DuckDB-admitted SQL -> PyTorch/GPU execution chain runnable over performance.

## Native DuckDB Substrait policy (B方案)

For strict Substrait work, this repository follows the native DuckDB Substrait
path selected for B方案:

```text
original SQL / --query N
  -> DuckDB get_substrait_json(original_sql)
  -> Substrait plan analysis / dispatch
  -> PyTorch tensor execution
```

The runner does **not** rewrite SQL to avoid DuckDB planner limitations, does
**not** fabricate Substrait plans, and does **not** use DuckDB result rows as the
PyTorch result path. If `--plan-source substrait` is selected and DuckDB cannot
export the original SQL, the command fails with `DuckDBSubstraitError`.

Probe the native export state explicitly:

```bash
tpch-torch-probe-substrait --db data/tpch_sf1.duckdb --queries all --json
tpch-torch-probe-substrait --db data/tpch_sf1.duckdb --queries 2,4,16
```

As of DuckDB 1.2.x, original TPC-H Q2, Q4, Q16, Q17, Q20, Q21, and Q22 are
native-export blocked (`DELIM_JOIN` or `MARK` join). This project keeps those as
explicit strict-Substrait failures and uses the Sirius-like logical-plan path
only when `--plan-source duckdb-logical` or `--plan-source auto` is requested.

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

# Run Q1 through the Substrait -> PyTorch path.
tpch-torch-run-q1 --db data/tpch_sf1.duckdb --device cuda

# Validate PyTorch output against DuckDB.
tpch-torch-validate-q1 --db data/tpch_sf1.duckdb --device cuda
```

Pre-exported JSON is only a debugging/caching aid for this legacy Q1 command;
the generic runner performs SQL -> DuckDB admission -> PyTorch in one command.
