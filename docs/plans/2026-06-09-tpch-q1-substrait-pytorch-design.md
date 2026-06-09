# TPC-H Q1 SQL → DuckDB/Substrait → PyTorch Design

## Goal

Initialize this repository as a minimal TQP-style query prototype that can run TPC-H Q1 at SF1 by taking SQL, asking DuckDB to parse/export a Substrait JSON plan, compiling the supported Q1 plan shape to PyTorch tensor operators, and validating results against DuckDB.

## Scope

This first version is intentionally Q1-focused. It supports the relational pattern required by TPC-H Q1:

```text
read lineitem
  → filter l_shipdate <= DATE '1998-09-02'
  → project arithmetic expressions
  → group by l_returnflag, l_linestatus
  → sum/count/avg aggregates
  → order by l_returnflag, l_linestatus
```

The runner should work for SF1 when the environment has enough memory and the required Python dependencies. Tests use small in-memory fixtures so the repository can be verified quickly without generating SF1.

## Architecture

- `tpch_torch.sql`: canonical TPC-H Q1 SQL.
- `tpch_torch.duckdb_bridge`: DuckDB integration for TPC-H data generation, baseline execution, and Substrait JSON export.
- `tpch_torch.substrait`: minimal plan inspection/compilation layer. It validates that DuckDB produced a plan with the required table read, filter, aggregate, and sort nodes. Unsupported plans fail explicitly instead of silently falling back to hard-coded execution.
- `tpch_torch.storage`: converts DuckDB/Arrow/Python column batches into columnar `torch.Tensor` tables. Dates are encoded as `YYYYMMDD` int32. Decimals are scaled integers where possible.
- `tpch_torch.operators`: reusable tensor operators for filter masks, grouping by composite keys, reductions, and stable result formatting.
- `tpch_torch.queries.q01`: Q1 PyTorch implementation driven by a compiled Q1 plan.
- `scripts/`: CLI entry points for generating SF1 DuckDB data, exporting Q1 Substrait JSON, running Q1 on PyTorch, and validating against DuckDB.

## Data representation

- `l_shipdate`: `torch.int32` encoded as `YYYYMMDD`.
- Numeric columns used by Q1: `torch.float64` for direct DuckDB parity in the initial version. Scaled integer decimal support can be added later when performance/correctness trade-offs are evaluated.
- `l_returnflag` and `l_linestatus`: dictionary-encoded int64 ids with vocabularies preserved for result decoding.

## Error handling

No fake success paths or silent fallbacks are allowed. If DuckDB's Substrait extension is unavailable, the bridge raises a clear dependency error. If the Substrait plan does not contain the expected Q1 relational operators, compilation raises `UnsupportedPlanError`. If CUDA is requested but unavailable, the runner raises an explicit error.

## Testing

Tests follow TDD and avoid requiring SF1. They cover:

1. Q1 SQL text contains the expected filter, grouping, and ordering semantics.
2. Substrait plan export/inspection works when DuckDB and the Substrait extension are installed.
3. The PyTorch Q1 executor matches a DuckDB baseline on a small fixture.
4. CLI-level validation can run against a generated small DuckDB database.

## References

- DuckDB TPC-H extension can generate SF1 with `CALL dbgen(sf = 1)` and exposes TPC-H queries/answers.
- DuckDB Substrait community extension exposes `get_substrait_json`, which converts SQL to Substrait JSON.
