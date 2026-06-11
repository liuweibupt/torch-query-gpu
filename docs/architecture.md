# Current Architecture: DuckDB Frontend -> TQP IR -> PyTorch Backend

中文版本见 [`docs/architecture.zh.md`](architecture.zh.md).

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
`COUNT(*)`, `COUNT(col)`, `SUM`, `MIN`, `MAX`, `AVG`, simple `GROUP BY`,
`ORDER BY` with `ASC`/`DESC`, `IN`, `LIKE`, `AND`, `OR`, `NOT`, and `LIMIT`. Unsupported
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
    def execute(
        self,
        con,
        plan: TQPPlan,
        device: str = "cpu",
        use_compressed_masks: bool = False,
    ) -> list[dict[str, Any]]:
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
        executor = getattr(module, f"execute_q{plan.query_id}")
        if plan.query_id == 6:
            return executor(con, device=device, use_compressed_masks=use_compressed_masks)
        return executor(con, device=device)
```

The Q1 branch can consume real Substrait JSON when `frontend="substrait"`; for
`frontend="sirius"`, it uses a canonical internal Q1 plan. Q6 additionally has
an explicit correctness-first compressed-mask option exposed by CLI
`--compressed-masks`. This flag changes only PyTorch predicate-mask execution;
it does not alter SQL, frontend admission, validation baseline, or storage
format.


## Operator fast paths added below the backend

The backend now avoids several correctness-first but slow Python materialization
paths while keeping the same frontend/backend boundary. Typed TPC-H fetch and
generic single-table fetch use DuckDB `fetchnumpy()` and encode NumPy arrays
directly:

```python
def table_from_columnar_typed(columnar, device="cpu") -> TensorTable:
    columns = {}
    dictionaries = {}
    for column_name, values_iterable in columnar.items():
        tensor, vocabulary = encode_column(column_name, values_iterable, device)
        columns[column_name] = tensor
        if vocabulary is not None:
            dictionaries[column_name] = vocabulary
    return TensorTable(columns=columns, dictionaries=dictionaries)

def _encode_numpy_typed_column(column_name, values, device):
    if column_name in STRING_COLUMNS_EXTENDED:
        vocabulary, inverse = np.unique(values.astype(str), return_inverse=True)
        return torch.as_tensor(inverse, dtype=torch.int64, device=device), tuple(vocabulary.tolist())
    if column_name in INT_COLUMNS:
        return torch.as_tensor(values, dtype=torch.int64, device=device), None
    return torch.as_tensor(values, dtype=torch.float64, device=device), None
```

Reusable grouping and lookup helpers live in `tpch_torch/operators.py` and
`tpch_torch/relational.py`:

```python
def low_cardinality_group_ids(key_columns, cardinalities):
    group_ids = torch.zeros(key_columns[0].shape, dtype=torch.int64, device=key_columns[0].device)
    multiplier = 1
    for key, cardinality in reversed(tuple(zip(key_columns, cardinalities))):
        group_ids = group_ids + key.to(dtype=torch.int64) * multiplier
        multiplier *= cardinality
    return group_ids, multiplier

@dataclass(frozen=True)
class LookupIndex:
    sorted_keys: torch.Tensor
    sorted_values: torch.Tensor
```

The generic grouped aggregate executor now groups masked rows with tensor
`composite_group_ids()` and computes aggregate columns with grouped reductions.
It decodes only final group keys and aggregate scalars, instead of building a
Python dictionary of row-index lists per group.

## Encoded mask execution

`tpch_torch/compressed.py` now has an explicit mask abstraction for the first
compressed execution experiments:

```python
@dataclass(frozen=True)
class PlainMask:
    values: torch.Tensor

@dataclass(frozen=True)
class RLEMask:
    ranges: RLERanges
    row_count: int

@dataclass(frozen=True)
class IndexMask:
    positions: torch.Tensor
    row_count: int

def mask_and(left, right):
    if isinstance(left, RLEMask) and isinstance(right, RLEMask):
        return RLEMask(range_intersect(left.ranges, right.ranges), left.row_count)
    if isinstance(left, IndexMask) and isinstance(right, IndexMask):
        return IndexMask(idx_in_idx(left.positions, right.positions), left.row_count)
    return _mask_and_mixed(left, right)
```

Q6 uses this path only when requested:

```python
def execute_q6(con, device="cpu", use_compressed_masks=False):
    table = fetch_tensor_table(con, "lineitem", LINEITEM_COLUMNS, device=device)
    if use_compressed_masks:
        mask = _q6_compressed_mask(table)
        positions = mask_to_index(mask)
        revenue = (extendedprice.index_select(0, positions) * discount.index_select(0, positions)).sum()
        return [{"revenue": float(revenue.cpu().item())}]
    return [_execute_q6_plain_mask_row(table)]
```

Mixed mask cases use documented explicit conversion to Plain or Index for
correctness. That exposes the encoded-mask boundary without pretending that the
repository already has full compressed column storage, compressed joins, or
cost-based encoding selection.

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
WHERE with comparisons, IN, LIKE, AND, OR, and NOT
column projection and column * constant projection
COUNT(*), COUNT(column), SUM, MIN, MAX, and AVG
simple GROUP BY
ORDER BY output columns with ASC/DESC
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

## Cold/hot benchmark timing

`tpch_torch/benchmark.py` provides the repository's reproducible timing path. It
intentionally times the same execution chain as `tpch-torch-run` rather than a
handwritten query shortcut:

```text
SQL text already resolved from --query/--sql/--sql-file
  -> run_sql_with_frontend()
  -> compile_tqp_plan()
  -> PyTorchBackend.execute()
  -> query/generic tensor executor
  -> materialized result rows
```

The CLI entrypoint is `tpch-torch-benchmark`:

```bash
tpch-torch-benchmark \
  --db data/tpch_sf1.duckdb \
  --query 6 \
  --device cuda \
  --frontend sirius \
  --cold-runs 3 \
  --warmup-runs 5 \
  --hot-runs 20 \
  --json
```

Key implementation snippet:

```python
def benchmark_sql(config, *, connect=connect_database, runner=run_sql_with_frontend):
    sync = _synchronizer(config.device, synchronizer)
    cold_samples = _measure_cold(config, connect, runner, clock_ns, sync)
    hot_samples = _measure_hot(config, connect, runner, clock_ns, sync)
    return BenchmarkReport(
        config=config,
        cold=summarize_samples(cold_samples),
        hot=summarize_samples(hot_samples),
        samples=tuple(cold_samples + hot_samples),
    )

def _measure_one(mode, iteration, con, config, runner, clock_ns, sync):
    sync()
    start_ns = clock_ns()
    result = runner(con, config.sql, device=config.device, frontend=config.frontend)
    sync()
    elapsed_ms = (clock_ns() - start_ns) / 1_000_000.0
    return TimingSample(mode, iteration, elapsed_ms, result.query_id, len(result.rows))
```

Cold samples open and close a new DuckDB connection for each measured run. Hot
samples reuse one connection and run unrecorded warmups before measurement. CUDA
samples synchronize before and after every measured run so asynchronous kernels
are included. The command reports `min`, `median`, `mean`, nearest-rank `p95`,
`max`, and sample standard deviation for cold and hot groups. It does not run
DuckDB validation; validation remains a separate correctness step.

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
