# DuckDB SQL → Substrait → PyTorch/GPU Framework Design

## Goal

Extend the repository from a Q1-only prototype into a direct SQL execution path that reads TPC-H SQL, asks DuckDB to export Substrait, compiles the supported Substrait plan shape, and executes the query with PyTorch tensors on CPU or GPU.

The primary acceptance target is every TPC-H query that DuckDB's Substrait extension can export in the current supported DuckDB version. Queries that DuckDB cannot export remain explicit `DuckDBSubstraitError` failures.

## User-facing behavior

The main path must not require pre-exported JSON. New generic commands will accept either a TPC-H query number or SQL text/file:

```bash
tpch-torch-run --db data/tpch_sf1.duckdb --query 6 --device cuda
tpch-torch-run --db data/tpch_sf1.duckdb --sql-file queries/q6.sql --device cuda
tpch-torch-validate --db data/tpch_sf1.duckdb --query 6 --device cuda
```

The command flow is always:

```text
SQL text
  → DuckDB get_substrait_json(sql)
  → Substrait analysis / dispatch
  → PyTorch tensor execution
  → optional DuckDB baseline validation
```

Pre-exported JSON remains a debugging/caching option, but it is not the normal execution path.

## Supported-query policy

Current probing shows:

- DuckDB executes Q1-Q22.
- DuckDB Substrait export succeeds for Q1, Q3, Q5-Q15, Q18, and Q19.
- DuckDB Substrait export fails for Q2, Q4, Q16, Q17, and Q20-Q22 because the extension reports unsupported join forms such as `DELIM_JOIN` or `MARK` joins.

This project will only promise PyTorch execution after DuckDB successfully exports a real Substrait plan. It will not silently rewrite SQL or bypass Substrait.

## Architecture

### SQL catalog and bridge

`tpch_torch.sql` will expose `get_tpch_query(query_id)` and `get_sql(...)` helpers backed by DuckDB's `tpch_queries()` output when a database connection is available. The bridge will add generic Substrait export and DuckDB baseline execution for arbitrary SQL.

### Execution dispatch

A new high-level module will expose:

```python
run_sql(con, sql, device="cpu") -> QueryResult
validate_sql(con, sql, device="cpu", tolerance=...) -> ValidationResult
```

`run_sql` exports Substrait from DuckDB and dispatches to supported PyTorch executors. Existing Q1 executor stays supported through this path.

### Initial implementation strategy

Build the framework incrementally:

1. Add generic CLI and SQL loading, preserving Q1 behavior.
2. Add a query dispatcher that can identify canonical TPC-H query numbers from SQL text or root output shape.
3. Add Q6 as the first non-Q1 direct SQL executor because it is single-table filter + aggregate.
4. Add more exported TPC-H queries using query-specific PyTorch executors behind the generic SQL/Substrait dispatcher.

This is intentionally not a complete Substrait engine yet. It is a direct SQL→Substrait→PyTorch framework with explicit support for concrete exported TPC-H query shapes. Unsupported shapes raise clear errors.

## Data and operators

- Numeric columns use `torch.float64` initially for correctness.
- Dates remain encoded as `YYYYMMDD` integers in tensor tables.
- Strings remain dictionary-encoded per loaded column. Query-specific executors decode only final result strings.
- Joins can be implemented with simple correctness-first tensor algorithms, including CPU-assisted mapping where necessary, as long as the actual query result derivation uses tensor tables and unsupported cases fail explicitly.

## Error handling

- DuckDB Substrait export failure raises `DuckDBSubstraitError`.
- A successfully exported but unsupported Substrait plan raises `UnsupportedPlanError` with the query/operator reason.
- CUDA requested without CUDA remains a hard error.
- No fake success paths, mock outputs, or silent DuckDB execution fallbacks are allowed.

## Validation

Every supported query gets an end-to-end validation test:

1. Load SQL directly via `--query` or SQL helper.
2. Export real Substrait in DuckDB.
3. Execute PyTorch path on CPU in unit tests and CUDA in manual/SF1 verification when available.
4. Compare rows against DuckDB baseline with numeric tolerance.

Q1 and Q6 are required first milestones. The final milestone records the exact set of DuckDB-exportable TPC-H queries supported by the PyTorch dispatcher.
