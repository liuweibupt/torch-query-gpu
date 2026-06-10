# Sirius-like Logical Plan Path Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run all 22 TPC-H queries from original SQL on the PyTorch/GPU path by adding an explicit DuckDB logical-plan admission path inspired by Sirius.

**Architecture:** Keep strict `substrait` mode unchanged. Add `duckdb-logical` and `auto` modes: `duckdb-logical` validates that DuckDB can plan the original SQL via `EXPLAIN`, then dispatches to PyTorch executors; `auto` tries DuckDB Substrait export first and switches to logical-plan admission only on native Substrait export failures.

**Tech Stack:** Python, DuckDB Python API, PyTorch tensor helpers, pytest, existing CLI scripts.

---

### Task 1: Add plan-source model and planner admission tests

**Files:**
- Create: `tpch_torch/planner.py`
- Create: `tests/test_planner.py`
- Modify later: `tpch_torch/runner.py`

**Step 1: Write failing tests**

Test that `DuckDBLogicalPlan` carries logical/optimized/physical strings, and that `export_duckdb_logical_plan` calls `PRAGMA explain_output='all'` then `EXPLAIN <sql>`.

**Step 2: Run RED**

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q tests/test_planner.py
```

Expected: import error for missing `tpch_torch.planner`.

**Step 3: Implement minimal code**

Create `tpch_torch/planner.py` with:

- `DuckDBLogicalPlan` dataclass
- `DuckDBPlannerError`
- `export_duckdb_logical_plan(con, sql)`

**Step 4: Run GREEN**

Run the same pytest command.

**Step 5: Commit**

```bash
git add tpch_torch/planner.py tests/test_planner.py
git commit -m "feat: add duckdb logical plan admission"
```

---

### Task 2: Add plan-source runner dispatch

**Files:**
- Modify: `tpch_torch/runner.py`
- Modify: `scripts/validate_query.py`
- Modify: `scripts/run_query.py`
- Modify: `tests/test_runner.py`
- Modify: `tests/test_runner_cli.py`

**Step 1: Write failing tests**

Add tests for:

- `run_sql_with_plan_source(..., plan_source='duckdb-logical')` calls planner admission and does not call Substrait export.
- `plan_source='substrait'` preserves existing export behavior.
- `plan_source='auto'` tries Substrait first and then planner admission only when export raises `DuckDBSubstraitError`.
- CLI parser accepts `--plan-source auto|substrait|duckdb-logical`.

Use dependency injection helpers where necessary; do not add runtime mocks or success fallbacks.

**Step 2: Run RED**

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q tests/test_runner.py tests/test_runner_cli.py
```

Expected: missing plan-source APIs/arguments.

**Step 3: Implement minimal code**

- Add `PlanSource = Literal['substrait', 'duckdb-logical', 'auto']`.
- Add `run_sql_with_plan_source` and `validate_sql_with_plan_source`.
- Update CLI to pass `args.plan_source`.
- Keep existing `run_sql` and `validate_sql` as strict Substrait wrappers.

**Step 4: Run GREEN**

Run the same pytest command.

**Step 5: Commit**

```bash
git add tpch_torch/runner.py scripts/validate_query.py scripts/run_query.py tests/test_runner.py tests/test_runner_cli.py
git commit -m "feat: add sirius-like plan source selection"
```

---

### Task 3: Add blocked query markers and dispatch stubs

**Files:**
- Modify: `tpch_torch/runner.py`
- Create: `tpch_torch/queries/q02.py`, `q04.py`, `q16.py`, `q17.py`, `q20.py`, `q21.py`, `q22.py`
- Modify: `tests/test_runner.py`

**Step 1: Write failing tests**

Add tests that original DuckDB TPC-H SQL for Q2/Q4/Q16/Q17/Q20/Q21/Q22 identifies to the correct query id.

**Step 2: Run RED**

Run targeted runner tests. Expected: unsupported query shape.

**Step 3: Implement markers and dispatcher entries**

Add robust marker tuples for the 7 blocked queries and add module dispatch entries. Stub executors should raise `UnsupportedPlanError` until their TDD task implements real logic.

**Step 4: Run GREEN for identification only**

The identification tests pass; executor tests can still fail until implemented.

**Step 5: Commit**

```bash
git add tpch_torch/runner.py tpch_torch/queries/q02.py tpch_torch/queries/q04.py tpch_torch/queries/q16.py tpch_torch/queries/q17.py tpch_torch/queries/q20.py tpch_torch/queries/q21.py tpch_torch/queries/q22.py tests/test_runner.py
git commit -m "feat: identify remaining tpch queries"
```

---

### Task 4: Implement Q4 and Q17 executors first

**Files:**
- Modify: `tpch_torch/queries/q04.py`
- Modify: `tpch_torch/queries/q17.py`
- Create/modify tests: `tests/test_remaining_tpch.py`

**Step 1: Write failing validation tests**

Parametrize Q4 and Q17 through `validate_sql_with_plan_source(..., plan_source='duckdb-logical')` on SF0.01.

**Step 2: Run RED**

Expected: executor stub raises.

**Step 3: Implement Q4/Q17**

- Q4: filter orders by date, compute orderkeys with at least one lineitem commitdate < receiptdate, group count by orderpriority.
- Q17: filter part by brand/container, compute avg lineitem quantity per part, sum extendedprice for qualifying lineitems and divide by 7.

**Step 4: Run GREEN**

Run targeted tests.

**Step 5: Commit**

```bash
git add tpch_torch/queries/q04.py tpch_torch/queries/q17.py tests/test_remaining_tpch.py
git commit -m "feat: add logical-plan executors for q4 q17"
```

---

### Task 5: Implement Q16 and Q22 executors

**Files:**
- Modify: `tpch_torch/queries/q16.py`
- Modify: `tpch_torch/queries/q22.py`
- Modify: `tests/test_remaining_tpch.py`

**Step 1: Write failing validation tests**

Add Q16 and Q22 to the SF0.01 logical-plan validation parametrization.

**Step 2: Run RED**

Expected: executor stubs raise or mismatches.

**Step 3: Implement Q16/Q22**

- Q16: filter parts and suppliers, count distinct valid suppliers by part brand/type/size.
- Q22: country-code/customer account balance filter and anti-join against orders.

**Step 4: Run GREEN**

Run targeted tests.

**Step 5: Commit**

```bash
git add tpch_torch/queries/q16.py tpch_torch/queries/q22.py tests/test_remaining_tpch.py
git commit -m "feat: add logical-plan executors for q16 q22"
```

---

### Task 6: Implement Q2, Q20, and Q21 executors

**Files:**
- Modify: `tpch_torch/queries/q02.py`
- Modify: `tpch_torch/queries/q20.py`
- Modify: `tpch_torch/queries/q21.py`
- Modify: `tests/test_remaining_tpch.py`

**Step 1: Write failing validation tests**

Add Q2, Q20, and Q21 to the SF0.01 logical-plan validation parametrization.

**Step 2: Run RED**

Expected: executor stubs raise or mismatches.

**Step 3: Implement Q2/Q20/Q21**

Correctness-first implementations may combine GPU tensor filters/grouping with small Python row assembly after reductions.

**Step 4: Run GREEN**

Run targeted tests.

**Step 5: Commit**

```bash
git add tpch_torch/queries/q02.py tpch_torch/queries/q20.py tpch_torch/queries/q21.py tests/test_remaining_tpch.py
git commit -m "feat: add logical-plan executors for q2 q20 q21"
```

---

### Task 7: Update README and full support tests

**Files:**
- Modify: `README.md`
- Modify: `tests/test_supported_tpch.py`

**Step 1: Write failing test**

Add an all-22 logical-plan validation test at SF0.01 with `plan_source='auto'`.

**Step 2: Run RED/GREEN as needed**

Run targeted and full tests.

**Step 3: README update**

Document:

- strict Substrait support matrix
- Sirius-like logical-plan support for all 22 queries
- example CLI commands with `--plan-source auto`
- explicit statement that this path is not DuckDB result fallback

**Step 4: Commit**

```bash
git add README.md tests/test_supported_tpch.py
git commit -m "docs: document sirius-like tpch path"
```

---

### Task 8: Verification and push

**Step 1: Unit tests**

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q
```

**Step 2: Compile**

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m compileall -q tpch_torch scripts
```

**Step 3: Real SF1 CUDA full TPC-H**

```bash
timeout 180 /work/torch-query-gpu/.venv/bin/tpch-torch-validate \
  --db /work/torch-query-gpu/data/tpch_sf1.duckdb \
  --queries 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 \
  --device cuda \
  --plan-source auto \
  --keep-going
```

If timeout is too low for correctness-first implementations, rerun with a higher explicit timeout and report exact runtime.

**Step 4: Push**

```bash
git status --short
git push
```
