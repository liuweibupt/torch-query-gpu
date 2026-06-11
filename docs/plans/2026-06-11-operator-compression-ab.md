# Operator Fast Paths and Compressed Mask Execution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement AB batch: speed up common tensor/operator paths and add the first end-to-end compressed-mask execution path for TPC-H Q6 without changing SQL or using DuckDB result fallbacks.

**Architecture:** Keep the clean `SQL -> DuckDB/Sirius-like frontend -> TQPPlan -> PyTorch backend` chain. Add reusable tensor primitives below the query templates, make generic/relational table fetch use columnar NumPy inputs instead of row/list materialization, and introduce explicit compressed mask objects (`PlainMask`, `RLEMask`, `IndexMask`) with logical dispatch. Q6 will still read SQL normally; only its PyTorch backend predicate combination can choose an explicit compressed mask path.

**Tech Stack:** Python 3.12, DuckDB, NumPy, PyTorch CPU/CUDA tensors, pytest, existing `tpch_torch` package.

---

## Constraints

- No SQL rewriting and no manually exported JSON.
- No DuckDB result fallback for PyTorch output.
- Fail unsupported behavior explicitly; do not add silent fallbacks.
- Keep code testable; write failing tests before production changes.
- Backend unit tests must be run with `timeout 60`.
- Commit and push the final result; if push fails, report the blocker.

## Task 1: Relational and generic columnar fetch fast path

**Files:**
- Modify: `tpch_torch/relational.py`
- Modify: `tpch_torch/backend/generic.py`
- Test: `tests/test_relational.py`
- Test: `tests/test_generic_backend.py`

**Step 1: Write failing tests**

Add tests proving:

```python
def test_table_from_columnar_typed_encodes_numpy_without_python_list_path():
    columnar = {
        "l_returnflag": np.array(["R", "A", "R"]),
        "l_shipdate": np.array([19940101, 19940102, 19940103], dtype=np.int32),
        "l_orderkey": np.array([3, 4, 5], dtype=np.int64),
        "l_extendedprice": np.array([10.5, 20.25, 30.0], dtype=np.float64),
    }
    table = table_from_columnar_typed(columnar, device="cpu")
    assert table.columns["l_returnflag"].tolist() == [1, 0, 1]
    assert table.dictionaries["l_returnflag"] == ("A", "R")
    assert table.columns["l_shipdate"].dtype == torch.int32
    assert table.columns["l_orderkey"].dtype == torch.int64
```

Extend generic backend tests to ensure `_fetch_generic_tensor_table()` can execute existing generic SQL through `fetchnumpy()` data, including string/date/numeric columns.

**Step 2: Run tests to verify failure**

Run:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_relational.py tests/test_generic_backend.py -q
```

Expected: new relational test fails before the NumPy typed encoder exists or generic fetch still uses row materialization.

**Step 3: Implement minimal code**

- Teach `table_from_columnar_typed()` to pass NumPy arrays directly to `encode_column()`.
- Make `encode_column()` accept `Iterable[Any] | np.ndarray` and dispatch to a NumPy path.
- Use `np.unique(..., return_inverse=True)` for string columns.
- Preserve date/int/float dtype decisions for typed TPC-H columns.
- Replace generic backend `fetchall()` with `fetchnumpy()` and add a NumPy-aware generic encoder.

**Step 4: Run tests to verify pass**

Run the same test command. Expected: PASS.

**Step 5: Commit**

```bash
git add tpch_torch/relational.py tpch_torch/backend/generic.py tests/test_relational.py tests/test_generic_backend.py
git commit -m "perf: add columnar numpy fetch fast paths"
```

## Task 2: Low-cardinality grouping and lookup index helpers

**Files:**
- Modify: `tpch_torch/operators.py`
- Modify: `tpch_torch/relational.py`
- Test: `tests/test_operators.py`
- Test: `tests/test_relational.py`

**Step 1: Write failing tests**

Add tests proving:

```python
def test_low_cardinality_group_ids_encode_dense_composite_keys():
    group_ids, group_count = low_cardinality_group_ids(
        (torch.tensor([0, 0, 1, 1]), torch.tensor([1, 2, 1, 2])),
        (2, 3),
    )
    assert group_count == 6
    assert group_ids.tolist() == [1, 2, 4, 5]
```

```python
def test_grouped_sum_and_count_bincount_reduce_dense_ids():
    group_ids = torch.tensor([1, 1, 2, 5])
    values = torch.tensor([1.5, 2.5, 10.0, 3.0])
    assert grouped_sum_bincount(values, group_ids, 6).tolist() == [0.0, 4.0, 10.0, 0.0, 0.0, 3.0]
    assert grouped_count_bincount(group_ids, 6).tolist() == [0, 2, 1, 0, 0, 1]
```

```python
def test_lookup_index_reuses_sorted_dimension_keys():
    index = build_lookup_index(torch.tensor([30, 10, 20]), torch.tensor([3, 1, 2]))
    assert lookup_values_from_index(index, torch.tensor([20, 40, 10])).tolist() == [2, -1, 1]
```

**Step 2: Run tests to verify failure**

Run:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_operators.py tests/test_relational.py -q
```

Expected: FAIL because helpers do not exist.

**Step 3: Implement minimal code**

- Add `low_cardinality_group_ids()`, `grouped_sum_bincount()`, and `grouped_count_bincount()` to `operators.py`.
- Add frozen `LookupIndex`, `build_lookup_index()`, and `lookup_values_from_index()` to `relational.py`.
- Keep existing `lookup_values()` behavior by delegating through a built index.
- Add optional `aggregate_sum_by_low_cardinality_keys()` and `aggregate_count_by_low_cardinality_keys()` wrappers for query reuse.

**Step 4: Run tests to verify pass**

Run the same command. Expected: PASS.

**Step 5: Commit**

```bash
git add tpch_torch/operators.py tpch_torch/relational.py tests/test_operators.py tests/test_relational.py
git commit -m "feat: add reusable grouped aggregate fast paths"
```

## Task 3: Tensorize generic grouped aggregate output

**Files:**
- Modify: `tpch_torch/backend/generic.py`
- Test: `tests/test_generic_backend.py`

**Step 1: Write failing/guard tests**

Add a grouped aggregate test with multiple aggregate kinds and filtered rows; the result must match the existing Python-row semantics:

```python
def test_generic_backend_tensorizes_grouped_aggregates_with_filter():
    con = _make_table()
    sql = "select a, sum(b) as total, count(*) as n, avg(b) as mean_b from t where b >= 2 group by a order by a"
    rows = execute_generic_sql_plan(con, parse_generic_sql(sql), device="cpu")
    assert rows == [{"a": 1, "total": 2.5, "n": 1, "mean_b": 2.5}, {"a": 2, "total": 3.0, "n": 1, "mean_b": 3.0}]
```

**Step 2: Run test to verify current behavior**

If it already passes, keep it as a regression guard. Also inspect implementation to ensure production still has row-loop grouping.

**Step 3: Implement minimal code**

- In `_execute_grouped()`, apply mask once and use `composite_group_ids()` over encoded group columns.
- Use tensor reductions (`grouped_sum`, `grouped_count`, `grouped_min`, `grouped_max`, `grouped_mean`) to compute aggregate projections by group id.
- Decode only one key row per output group.
- Preserve explicit errors for unsupported grouped projections.

**Step 4: Run tests**

Run:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_generic_backend.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tpch_torch/backend/generic.py tests/test_generic_backend.py
git commit -m "perf: tensorize generic grouped aggregates"
```

## Task 4: Compressed MaskColumn abstraction and logical dispatch

**Files:**
- Modify: `tpch_torch/compressed.py`
- Test: `tests/test_compressed.py`

**Step 1: Write failing tests**

Add tests proving:

```python
def test_mask_column_dispatch_matches_plain_boolean_logic():
    left = RLEMask(plain_to_rle(torch.tensor([False, True, True, False, True])), row_count=5)
    right = IndexMask(torch.tensor([2, 3, 4]), row_count=5)
    assert mask_to_plain(mask_and(left, right)).tolist() == [False, False, True, False, True]
    assert mask_to_plain(mask_or(left, right)).tolist() == [False, True, True, True, True]
    assert mask_to_plain(mask_not(right)).tolist() == [True, True, False, True, False]
```

Add validation tests for mismatched row counts/devices.

**Step 2: Run tests to verify failure**

Run:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_compressed.py -q
```

Expected: FAIL because `PlainMask`/`RLEMask`/`IndexMask` and dispatch functions do not exist.

**Step 3: Implement minimal code**

- Add frozen mask dataclasses: `PlainMask`, `RLEMask`, `IndexMask`.
- Add `mask_to_plain()` and `mask_to_index()`.
- Add `mask_and()`, `mask_or()`, and `mask_not()`.
- Prefer encoded primitives for RLE/RLE, Index/Index, RLE/Index; mixed paths may explicitly convert through plain/index for correctness, with documented conversion points.

**Step 4: Run tests**

Run the same command. Expected: PASS.

**Step 5: Commit**

```bash
git add tpch_torch/compressed.py tests/test_compressed.py
git commit -m "feat: add encoded mask dispatch"
```

## Task 5: Q6 compressed mask path and benchmark guard

**Files:**
- Modify: `tpch_torch/queries/q06.py`
- Test: `tests/test_q06.py`

**Step 1: Write failing tests**

Add test proving the compressed path returns the same revenue:

```python
def test_execute_q6_compressed_mask_path_matches_plain_path():
    con = duckdb.connect()
    create_lineitem_fixture(con, FIXTURE_ROWS)
    assert execute_q6(con, device="cpu", use_compressed_masks=True) == execute_q6(con, device="cpu")
```

**Step 2: Run test to verify failure**

Run:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_q06.py -q
```

Expected: FAIL because the keyword/path does not exist.

**Step 3: Implement minimal code**

- Add explicit optional `use_compressed_masks: bool = False` parameter to `execute_q6()`.
- Build predicate masks from tensors, wrap them as `PlainMask`, convert date masks to RLE via `plain_to_rle()`, and combine through `mask_and()`.
- Use `mask_to_index()` to select revenue rows for the compressed path.
- Keep default path unchanged for existing callers.

**Step 4: Run tests**

Run the same command. Expected: PASS.

**Step 5: Commit**

```bash
git add tpch_torch/queries/q06.py tests/test_q06.py
git commit -m "feat: route q6 through compressed mask execution"
```

## Task 6: Documentation and verification

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/operator-roadmap.md`

**Step 1: Update docs**

- Mark AB batch items in README TODO.
- Document columnar fetch fast path, grouped aggregate fast path, lookup index helper, `MaskColumn`, and Q6 compressed execution.
- State that compressed Q6 path is explicit and correctness-first; it does not imply full compressed storage yet.

**Step 2: Run full verification**

Run:

```bash
for f in tests/test_*.py; do timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest "$f" -q || exit $?; done
timeout 60 /work/torch-query-gpu/.venv/bin/python -m compileall -q tpch_torch scripts
PYTHONPATH=$PWD timeout 120 /work/torch-query-gpu/.venv/bin/python -m scripts.validate_query --db /work/torch-query-gpu/data/tpch_sf1.duckdb --query 6 --device cuda --frontend sirius
PYTHONPATH=$PWD timeout 120 /work/torch-query-gpu/.venv/bin/python -m scripts.validate_query --db /work/torch-query-gpu/data/tpch_sf1.duckdb --query 1 --device cuda --frontend sirius
```

If CUDA is unavailable, run the validation commands with `--device cpu` and record the device limitation.

**Step 3: Commit docs**

```bash
git add README.md docs/architecture.md docs/operator-roadmap.md
git commit -m "docs: document operator fast paths and compressed masks"
```
