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

If DuckDB cannot install/load the community `substrait` extension, export and
validation commands fail with `DuckDBSubstraitError`. This is intentional: the
project does not silently bypass the SQL -> Substrait stage. You can still run
`tpch-torch-run-q1 --substrait-json path/to/q1.json` with a previously exported
real Substrait plan.
