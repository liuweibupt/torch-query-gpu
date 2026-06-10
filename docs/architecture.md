# Current Architecture: DuckDB Frontend -> TQP IR -> PyTorch Backend

This repository is a correctness-first TQP-style prototype for analytical query
execution on PyTorch tensors. The default path uses DuckDB for SQL admission and
planning metadata, then executes with PyTorch operators on CPU or CUDA.

Substrait is preserved as an explicit strict experimental frontend. There is no
automatic fallback between frontends.

## End-to-end flow

```text
SQL text / --query N
  -> runner.load_sql()
  -> Sirius-like DuckDB frontend
       DuckDB parses, binds, plans, and optimizes the original SQL via EXPLAIN
  -> TQPPlan
       immutable frontend/backend boundary object
  -> PyTorchBackend
       dispatches to TPC-H tensor executors q01.py ... q22.py
       or to the generic SQL subset executor
  -> optional DuckDB validation baseline
```

The validation baseline runs the same original SQL in DuckDB and compares rows.
It is not used as a fallback result for the PyTorch path.

Frontend admission and backend execution are intentionally separate. The
Sirius-like frontend can admit any SQL that DuckDB can parse and plan. The
PyTorch backend executes all TPC-H Q1-Q22 templates plus an explicit generic SQL
subset: single-table `SELECT`, simple `WHERE`, arithmetic projections,
`COUNT(*)`, `SUM(col)`, simple `GROUP BY`, `ORDER BY`, and `LIMIT`. Unsupported
generic operators raise `UnsupportedPlanError`.

## Runtime entrypoints

The generic CLI commands are:

```bash
# Default clean path.
tpch-torch-run \
  --db data/tpch_sf1.duckdb \
  --query 21 \
  --device cuda \
  --frontend sirius

# Validate all TPC-H queries on the same path.
tpch-torch-validate \
  --db data/tpch_sf1.duckdb \
  --queries all \
  --device cuda \
  --frontend sirius \
  --keep-going

# Strict experimental Substrait path for DuckDB-exportable queries.
tpch-torch-validate \
  --db data/tpch_sf1.duckdb \
  --query 6 \
  --device cuda \
  --frontend substrait
```

Supported frontends:

- `sirius`: default DuckDB parser/planner admission path.
- `substrait`: strict native DuckDB Substrait export path.

## Module map

| Layer | Files | Responsibility |
| --- | --- | --- |
| CLI | `scripts/run_query.py`, `scripts/validate_query.py` | Parse command-line arguments and pass explicit frontend/device/query source. |
| Runner | `tpch_torch/runner.py` | Thin orchestration: load SQL, compile frontend plan, call backend, validate output. |
| Frontend | `tpch_torch/frontend/sirius.py`, `tpch_torch/frontend/substrait.py` | Compile original SQL into `TQPPlan`. |
| IR | `tpch_torch/ir/plan.py` | Immutable internal plan object passed from frontend to backend. |
| Backend | `tpch_torch/backend/pytorch.py`, `tpch_torch/backend/generic.py` | Execute `TQPPlan` with PyTorch tensor operators. |
| Query catalog | `tpch_torch/query_catalog.py` | Identify supported TPC-H query shapes from original SQL text. |
| DuckDB planner admission | `tpch_torch/planner.py` | Ask DuckDB to parse/plan original SQL via `EXPLAIN`. |
| Tensor kernels | `tpch_torch/queries/q01.py` ... `q22.py` | Correctness-first PyTorch implementations of TPC-H query templates. |
| Strict Substrait bridge | `tpch_torch/duckdb_bridge.py`, `tpch_torch/substrait.py` | Export and compile real DuckDB Substrait plans for the experimental frontend. |

## Runner orchestration

`tpch_torch/runner.py` is intentionally small. It does not contain frontend or
backend implementation details; it only wires the selected frontend to the
PyTorch backend.

```python
def run_sql_with_frontend(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    device: str = "cpu",
    frontend: FrontendName = "sirius",
) -> QueryResult:
    _validate_device(device)
    plan = compile_tqp_plan(con, sql, frontend)
    rows = PyTorchBackend().execute(con, plan, device=device)
    return QueryResult(query_id=plan.query_id, rows=rows)
```

Frontend selection is explicit:

```python
def compile_tqp_plan(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    frontend: FrontendName = "sirius",
) -> TQPPlan:
    if frontend == "sirius":
        return compile_sirius_plan(con, sql)
    if frontend == "substrait":
        return compile_substrait_plan(con, sql)
    raise ValueError(f"unknown frontend: {frontend}")
```

Validation calls the same execution path, then compares against DuckDB:

```python
def validate_sql_with_frontend(...):
    result = run_sql_with_frontend(con, sql, device=device, frontend=frontend)
    duckdb_rows = run_duckdb_sql(con, sql)
    max_abs_error = compare_rows(duckdb_rows, result.rows)
    return SQLValidationResult(...)
```

## TQP IR boundary

The internal IR is currently a compact immutable plan object. It is deliberately
small: the repository first guarantees all TPC-H queries can run through a clean
frontend/backend boundary, then can evolve `TQPPlan` into a richer operator graph.

```python
FrontendName = Literal["sirius", "substrait"]


@dataclass(frozen=True)
class DuckDBPlanMetadata:
    logical_plan: str = ""
    logical_opt: str = ""
    physical_plan: str = ""


@dataclass(frozen=True)
class TQPPlan:
    query_id: int | None
    source_sql: str
    frontend: FrontendName
    duckdb_metadata: DuckDBPlanMetadata | None = None
    plan_json: dict[str, Any] | None = None
    generic_plan: Any | None = None
    generic_error: str | None = None
```

Important fields:

- `query_id`: selected TPC-H executor template, or `None` for generic SQL.
- `source_sql`: the unchanged original SQL.
- `frontend`: `sirius` or `substrait`.
- `duckdb_metadata`: textual DuckDB plans captured by the Sirius-like frontend.
- `plan_json`: real DuckDB Substrait JSON when the strict Substrait frontend is
  selected.
- `generic_plan`: executable generic SQL subset plan for non-TPC-H SQL.
- `generic_error`: parser/executor-subset reason when the frontend admitted SQL
  but the PyTorch backend does not yet support that generic shape.

## Sirius-like frontend

The default frontend uses DuckDB as the SQL parser/binder/planner/optimizer. It
captures DuckDB's logical, optimized logical, and physical plan text through
`EXPLAIN` and stores that metadata in `TQPPlan`.

```python
def compile_sirius_plan(con: duckdb.DuckDBPyConnection, sql: str) -> TQPPlan:
    duckdb_plan = export_duckdb_logical_plan(con, sql)
    generic_plan = None
    generic_error = None
    try:
        query_id = identify_tpch_query(sql)
    except UnsupportedPlanError:
        query_id = None
        try:
            generic_plan = parse_generic_sql(sql)
        except UnsupportedPlanError as exc:
            generic_error = str(exc)
    return TQPPlan(
        query_id=query_id,
        source_sql=sql,
        frontend="sirius",
        duckdb_metadata=DuckDBPlanMetadata(...),
        generic_plan=generic_plan,
        generic_error=generic_error,
    )
```

The DuckDB admission function is:

```python
def export_duckdb_logical_plan(con: object, sql: str) -> DuckDBLogicalPlan:
    try:
        con.execute("PRAGMA explain_output='all'")
        rows = con.execute(f"EXPLAIN {sql}").fetchall()
    except duckdb.Error as exc:
        raise DuckDBPlannerError(f"DuckDB EXPLAIN failed: {exc}") from exc
    sections = {str(name): str(plan) for name, plan in rows}
    return DuckDBLogicalPlan(
        logical_plan=sections.get("logical_plan", ""),
        logical_opt=sections.get("logical_opt", ""),
        physical_plan=sections.get("physical_plan", ""),
    )
```

This mirrors the Sirius architectural choice: rely on DuckDB's parser/planner
rather than requiring every query to be exportable through DuckDB's Substrait
extension.

## Strict Substrait frontend

The strict Substrait frontend remains available for experiments and for checking
DuckDB Substrait exporter coverage. It does not synthesize plans and does not
fallback to another frontend.

```python
def compile_substrait_plan(con: duckdb.DuckDBPyConnection, sql: str) -> TQPPlan:
    return TQPPlan(
        query_id=identify_tpch_query(sql),
        source_sql=sql,
        frontend="substrait",
        plan_json=export_substrait_json(con, sql),
    )
```

If DuckDB cannot export the original SQL, this path raises
`DuckDBSubstraitError`.

## PyTorch backend

The backend receives a `TQPPlan`; it does not parse SQL or call DuckDB's
planner. It dispatches to tensor kernels or to the generic SQL subset executor.

```python
class PyTorchBackend:
    def execute(self, con, plan: TQPPlan, device: str = "cpu") -> list[dict[str, Any]]:
        if plan.query_id is None:
            if plan.generic_plan is None:
                detail = plan.generic_error or "generic SQL plan is missing executable operator plan"
                raise UnsupportedPlanError(f"generic SQL is not executable by PyTorch backend: {detail}")
            return execute_generic_sql_plan(con, plan.generic_plan, device=device)
        if plan.query_id == 1:
            q1_plan = _compile_q1_plan(plan.plan_json)
            from tpch_torch.duckdb_bridge import fetch_lineitem_tensor_table

            return execute_q1(fetch_lineitem_tensor_table(con, device=device), q1_plan)
        module_name = _EXECUTOR_BY_QUERY.get(plan.query_id)
        if module_name is None:
            raise UnsupportedPlanError(...)
        module = __import__(f"tpch_torch.queries.{module_name}", fromlist=[...])
        return getattr(module, f"execute_q{plan.query_id}")(con, device=device)
```

The Q1 branch can consume real Substrait JSON when `frontend="substrait"`; for
`frontend="sirius"`, it uses a canonical internal Q1 plan.

## Query identification and generic SQL

TPC-H templates still use query-id dispatch. `tpch_torch/query_catalog.py`
identifies those queries by checking stable SQL markers from DuckDB's TPC-H
query text. Non-TPC-H SQL is admitted with `query_id=None`. If the SQL falls in
the current generic backend subset, it is parsed into `GenericSQLPlan`;
otherwise the plan keeps `generic_plan=None` and records `generic_error` so
backend failure is explicit and attributable to unsupported execution, not
frontend admission.

```python
def identify_tpch_query(sql: str) -> int:
    normalized = _normalize_sql(sql)
    for query_id, markers in QUERY_MARKERS:
        if all(_normalize_sql(marker) in normalized for marker in markers):
            return query_id
    raise UnsupportedPlanError("SQL text does not match a supported TPC-H query shape")
```

This is intentionally a correctness-first bridge. The current generic plan is a
small operator subset, and the planned next architectural step is to replace more
template dispatch with a richer operator graph inside `TQPPlan`:

```text
Scan -> Filter -> Join -> Aggregate -> Sort -> Limit
```

## Current SQL support

Generic SQL subset supported by the PyTorch backend:

```text
single-table SELECT
WHERE with simple comparisons combined by AND
column projection and column * constant projection
COUNT(*) and SUM(column)
simple GROUP BY
ORDER BY output columns
LIMIT
```

Unsupported generic SQL, including joins, subqueries, windows, set operations,
and HAVING, fails explicitly.

## Current TPC-H support

| Query set | Default Sirius-like frontend | Strict DuckDB Substrait frontend | PyTorch backend |
| --- | --- | --- | --- |
| Q1, Q3, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, Q13, Q14, Q15, Q18, Q19 | yes | yes | yes |
| Q2, Q4, Q16, Q17, Q20, Q21, Q22 | yes | blocked in DuckDB 1.2.x Substrait export | yes |

The strict Substrait failures are explicit exporter limitations, not PyTorch
backend fallbacks.

## Operator roadmap

The paper-derived operator and optimization backlog is tracked in
[`docs/operator-roadmap.md`](operator-roadmap.md). It separates verified full-text
items from abstract-derived TQEx/TQP++/CoddSpeed items and identifies the current
implementation batches.

## Verification commands

Use these commands after architecture changes:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q
timeout 60 /work/torch-query-gpu/.venv/bin/python -m compileall -q tpch_torch scripts
timeout 300 /work/torch-query-gpu/.venv/bin/tpch-torch-validate \
  --db /work/torch-query-gpu/data/tpch_sf1.duckdb \
  --queries all \
  --device cuda \
  --frontend sirius \
  --keep-going
```
