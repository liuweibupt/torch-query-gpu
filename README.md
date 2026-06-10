# torch-query-gpu

A correctness-first TQP-style prototype for analytical query execution on PyTorch
tensors. The default execution path is a clean DuckDB-planned frontend and a
PyTorch backend:

```text
SQL / --query N
  -> Sirius-like DuckDB frontend
       DuckDB parses, binds, plans, and optimizes the original SQL
  -> TQP IR
       immutable frontend/backend boundary object
  -> PyTorch backend
       tensor operators on CPU or CUDA
  -> optional DuckDB baseline validation
```

Substrait is not the default execution path. It remains available only as an
explicit strict frontend for SQL that DuckDB's native Substrait exporter can emit:

```text
SQL / --query N
  -> DuckDB get_substrait_json(original_sql)
  -> TQP IR carrying the real Substrait JSON
  -> PyTorch backend
```

There is no automatic frontend fallback. The project does **not** rewrite SQL to
avoid planner limitations, does **not** fabricate Substrait JSON, and does
**not** use DuckDB result rows as PyTorch output. DuckDB rows are used only as a
validation baseline.

The Sirius-like frontend can admit any SQL that DuckDB can parse and plan. The
PyTorch backend currently executes all TPC-H Q1-Q22 templates plus an explicit
generic SQL subset: single-table `SELECT`, simple `WHERE`, arithmetic projection,
`COUNT(*)`, `SUM(col)`, simple `GROUP BY`, `ORDER BY`, and `LIMIT`. SQL outside
that backend subset is admitted by the frontend but fails at backend execution
with explicit `UnsupportedPlanError`.

For a module-by-module implementation guide with key code snippets, see
[`docs/architecture.md`](docs/architecture.md).

For the paper-derived operator and optimization backlog, see
[`docs/operator-roadmap.md`](docs/operator-roadmap.md).

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

The generic commands read SQL directly from `--query`, `--sql`, or `--sql-file`.
They do not require manually exported JSON.

Default Sirius-like frontend for TPC-H templates:

```bash
tpch-torch-run --db data/tpch_sf1.duckdb --query 21 --device cuda
tpch-torch-validate --db data/tpch_sf1.duckdb --query 21 --device cuda
```

Default Sirius-like frontend for a generic SQL subset:

```bash
tpch-torch-validate \
  --db data/tpch_sf1.duckdb \
  --sql "select count(*) as n from lineitem" \
  --device cuda
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

Supported frontends are explicit:

- `sirius`: default DuckDB parser/planner admission path.
- `substrait`: strict DuckDB native Substrait export path.

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
- `substrait`: DuckDB's real `get_substrait_json()` export; this is a strict
  experimental frontend and has no fallback.

### TQP IR

`TQPPlan` is the internal plan object between frontend and backend. It records
the source SQL, optional TPC-H query id, frontend, DuckDB plan metadata, optional
real Substrait JSON, and optional generic operator plan. TPC-H templates use
`query_id`; non-TPC-H SQL uses `query_id=None` plus `generic_plan` when the SQL
falls inside the current generic executor subset.

### PyTorch backend

`PyTorchBackend` executes `TQPPlan` with correctness-first tensor operators in
`tpch_torch/queries/q01.py` through `q22.py` for TPC-H templates, and with
`tpch_torch/backend/generic.py` for the supported generic SQL subset. This keeps
backend execution separate from SQL admission.

## Native DuckDB Substrait policy (B方案)

For strict Substrait experiments, this repository follows the native DuckDB
Substrait path selected for B方案:

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
frontend exists so the clean TQP path is not blocked by that exporter limitation.

If you build or obtain a newer native DuckDB Substrait extension, test it without
changing SQL by setting:

```bash
export TQG_SUBSTRAIT_EXTENSION=/path/to/substrait.duckdb_extension
```

When this variable is set, the bridge loads that exact extension path. Missing
or unloadable paths are hard errors.
