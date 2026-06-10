# Batch Query Validation Design

**Goal:** add one CLI path that validates several TPC-H queries from original DuckDB SQL through DuckDB Substrait and the PyTorch executor without manual plan export.

## Constraints

- The only accepted execution chain is: DuckDB `tpch_queries()` original SQL -> `get_substrait_json(original_sql)` -> PyTorch executor -> DuckDB baseline comparison.
- The CLI must not accept or consume hand-written JSON plans.
- The CLI must not rewrite SQL to avoid DuckDB Substrait limitations.
- The CLI must not fall back to DuckDB results as PyTorch results.
- Unsupported native exports or missing PyTorch executors must fail explicitly.

## User-facing behavior

`tpch-torch-validate` keeps the existing single-source modes:

```bash
tpch-torch-validate --db data/tpch_sf1.duckdb --query 6 --device cuda
tpch-torch-validate --db data/tpch_sf1.duckdb --sql-file q.sql
tpch-torch-validate --db data/tpch_sf1.duckdb --sql "select ..."
```

It gains a batch mode:

```bash
tpch-torch-validate --db data/tpch_sf1.duckdb --queries 1,3,5,6 --device cuda --keep-going
```

`--queries` is mutually exclusive with `--query`, `--sql`, and `--sql-file`. The argument is a comma-separated list of TPC-H query ids. Batch mode reads each original query with `get_tpch_query(con, query_id)`, then calls the same `validate_sql(con, sql, device=...)` path used by single-query validation.

## Data flow

For each query id in `--queries`:

1. `get_tpch_query(con, query_id)` loads DuckDB's canonical TPC-H SQL from `tpch_queries()`.
2. `validate_sql(con, sql, device)` calls `run_sql`.
3. `run_sql` calls `export_substrait_json(con, sql)`, which calls DuckDB `get_substrait_json`.
4. The existing query dispatcher identifies the TPC-H shape and runs the PyTorch executor on the requested device.
5. `validate_sql` runs DuckDB baseline SQL and compares normalized rows.

## Error handling

- Without `--keep-going`, batch mode stops on the first exception or tolerance failure and exits non-zero.
- With `--keep-going`, batch mode prints every query status, continues after failures, and exits non-zero at the end if any query failed.
- Each failed query records the real exception message. Failures are not swallowed or converted to success.
- `--device cuda` keeps the existing strict CUDA availability check.

## Testing

Tests cover parsing and orchestration without faking successful execution:

- `--queries` parses as a batch source and is mutually exclusive with single-query sources.
- `parse_query_ids("1,3,5,6")` returns `(1, 3, 5, 6)` and rejects an empty list.
- Batch validation obtains every SQL string by query id and calls the provided validator once per query.
- `--keep-going` returns failed query records while continuing; without it the original exception propagates.

End-to-end verification uses an existing DuckDB SF1 database and the actual `tpch-torch-validate --queries ...` CLI, so DuckDB Substrait export happens inside the normal execution path.
