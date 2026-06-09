# torch-query-gpu

A minimal TQP-style prototype for analytical query execution on PyTorch tensors.
The initial target is TPC-H Q1:

```text
TPC-H Q1 SQL
  -> DuckDB SQL parser
  -> DuckDB Substrait JSON export
  -> Q1 plan validation/compiler
  -> PyTorch tensor operators
  -> DuckDB result validation
```

The first implementation is deliberately narrow. It supports the Q1 operator
shape: scan `lineitem`, filter by `l_shipdate`, compute arithmetic projections,
group by `l_returnflag` and `l_linestatus`, aggregate, and order the final rows.
Unsupported Substrait plans fail explicitly.

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


## Generic direct SQL path

The generic runner reads SQL directly, asks DuckDB to export a real Substrait
plan, and dispatches the supported plan shape to PyTorch tensors:

```bash
tpch-torch-run --db data/tpch_sf1.duckdb --query 6 --device cuda --json
tpch-torch-validate --db data/tpch_sf1.duckdb --query 6 --device cuda

tpch-torch-run --db data/tpch_sf1.duckdb --sql-file path/to/query.sql --device cuda
tpch-torch-run --db data/tpch_sf1.duckdb --sql "select ..." --device cuda
```

Pre-exported JSON is only a debugging/caching aid for the legacy Q1 command; the
generic runner performs SQL -> DuckDB Substrait -> PyTorch in one command.

### TPC-H support matrix

Current DuckDB-exportable queries supported by the PyTorch path are:

```text
Q1, Q3, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, Q13, Q14, Q15, Q18, Q19
```

DuckDB executes Q1-Q22, but DuckDB 1.2.x Substrait export currently fails for
Q2, Q4, Q16, Q17, and Q20-Q22 with unsupported join forms such as `DELIM_JOIN`
or `MARK` joins. Those remain explicit `DuckDBSubstraitError` failures rather
than silent SQL rewrites or DuckDB-result fallbacks.

## SF1 target flow

```bash
# Generate a local DuckDB database with TPC-H SF1 data.
tpch-torch-gen-sf1 --db data/tpch_sf1.duckdb --sf 1

# Export DuckDB's Substrait JSON plan for Q1.
tpch-torch-export-q1-substrait --db data/tpch_sf1.duckdb --out data/q1_substrait.json

# Run Q1 through the Substrait -> PyTorch path.
tpch-torch-run-q1 --db data/tpch_sf1.duckdb --device cuda

# Validate PyTorch output against DuckDB.
tpch-torch-validate-q1 --db data/tpch_sf1.duckdb --device cuda
```

Use `--device cpu` on machines without CUDA. If `--device cuda` is requested on a
CPU-only machine, the runner raises an explicit error.

Validation compares floating point aggregates with an absolute tolerance. The
default tolerance is `1e-2`, which covers the small accumulation-order
differences between DuckDB decimals and PyTorch tensor reductions at SF1. Use
`--tolerance` to make the check stricter or looser.

If DuckDB cannot install/load the community `substrait` extension, export and
validation commands fail with `DuckDBSubstraitError`. This is intentional: the
project does not silently bypass the SQL -> Substrait stage. You can still run
`tpch-torch-run-q1 --substrait-json path/to/q1.json` with a previously exported
real Substrait plan.

## Native DuckDB Substrait policy (B方案)

For the remaining TPC-H gaps, this repository follows the native DuckDB
Substrait path selected for B方案:

```text
original SQL / --query N
  -> DuckDB get_substrait_json(original_sql)
  -> Substrait plan analysis / dispatch
  -> PyTorch tensor execution
```

The runner does **not** rewrite SQL to avoid DuckDB planner limitations, does
not fabricate Substrait plans, and does not use DuckDB result rows as the
PyTorch result path. If DuckDB cannot export the original SQL, the command fails
with `DuckDBSubstraitError`.

Probe the native export state explicitly:

```bash
tpch-torch-probe-substrait --db data/tpch_sf1.duckdb --queries all --json
tpch-torch-probe-substrait --db data/tpch_sf1.duckdb --queries 2,4,16
```

As of DuckDB 1.2.x, original TPC-H Q2, Q4, Q16, Q17, Q20, Q21, and Q22 are
native-export blocked (`DELIM_JOIN` or `MARK` join). If you build or obtain a
newer native DuckDB Substrait extension, test it without changing SQL by setting:

```bash
export TQG_SUBSTRAIT_EXTENSION=/path/to/substrait.duckdb_extension
```

When this variable is set, the bridge loads that exact extension path. Missing
or unloadable paths are hard errors.
