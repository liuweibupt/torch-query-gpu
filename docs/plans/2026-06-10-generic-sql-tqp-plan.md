# Generic SQL TQP Plan Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let the Sirius-like frontend accept any DuckDB-plannable SQL into a `TQPPlan`, and add a real generic PyTorch backend path for an initial SQL subset without using DuckDB result rows as fallback.

**Architecture:** Keep TPC-H Q1-Q22 template executors intact. Add `query_id: int | None` and `generic_plan` to `TQPPlan`; frontend emits `query_id=None` for non-TPC-H SQL instead of failing. Add `GenericSQLPlan` and `execute_generic_sql_plan` for a constrained but real subset: single-table `SELECT`, `WHERE`, arithmetic projections, `COUNT(*)`, `SUM(col)`, simple `GROUP BY`, `ORDER BY`, and `LIMIT`. Unsupported SQL must raise an explicit `UnsupportedPlanError`.

**Tech Stack:** Python 3.12, DuckDB for planning/admission and schema inspection, PyTorch tensor operators for execution, pytest.

---

### Task 1: Extend TQPPlan for generic SQL

**Files:**
- Modify: `tpch_torch/ir/plan.py`
- Modify: `tpch_torch/frontend/sirius.py`
- Modify: `tpch_torch/frontend/substrait.py`
- Test: `tests/test_frontend.py`

**Step 1:** Write failing tests proving `compile_sirius_plan(con, "select count(*) as n from t")` returns `TQPPlan(query_id=None)` after DuckDB EXPLAIN succeeds.

**Step 2:** Run `timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q tests/test_frontend.py` and confirm failure from `identify_tpch_query`.

**Step 3:** Add `query_id: int | None` and `generic_plan: Any | None` to `TQPPlan`. In Sirius frontend catch `UnsupportedPlanError` from TPC-H identification and keep `query_id=None`. In strict Substrait frontend preserve strict behavior for now by still requiring TPC-H identification.

**Step 4:** Re-run focused tests and commit.

### Task 2: Add generic SQL parser for a first real operator subset

**Files:**
- Create: `tpch_torch/generic_sql.py`
- Test: `tests/test_generic_sql.py`

**Step 1:** Write failing tests for parsing:
- `select count(*) as n from t`
- `select a, sum(b) as total from t where b >= 2 group by a order by a limit 10`
- unsupported join raises `UnsupportedPlanError`.

**Step 2:** Implement a small parser based on DuckDB-planned SQL text and constrained regex, not DuckDB result fallback. It should emit immutable plan dataclasses with table, projections, filters, group_by, order_by, limit.

**Step 3:** Run parser tests and commit.

### Task 3: Add generic PyTorch executor

**Files:**
- Create: `tpch_torch/backend/generic.py`
- Modify: `tpch_torch/backend/pytorch.py`
- Test: `tests/test_generic_backend.py`

**Step 1:** Write failing execution tests against small DuckDB tables for:
- `select count(*) as n from t`
- `select a, sum(b) as total from t group by a order by a`
- `select a, b * 2 as twice from t where b >= 2 order by a`

**Step 2:** Implement real PyTorch execution by fetching only source table columns into tensors and evaluating filters/projections/aggregates/sort/limit with torch/Python materialization. Do not execute the full SQL in DuckDB except validation outside backend.

**Step 3:** Wire `PyTorchBackend.execute` to use generic executor when `plan.query_id is None` and `plan.generic_plan is not None`.

**Step 4:** Run focused tests and commit.

### Task 4: Wire generic plans into Sirius frontend and validation

**Files:**
- Modify: `tpch_torch/frontend/sirius.py`
- Modify: `tpch_torch/runner.py`
- Test: `tests/test_runner_generic_sql.py`

**Step 1:** Write failing validation tests for generic SQL through `validate_sql(con, sql, device="cpu")`.

**Step 2:** In Sirius frontend call the generic SQL parser for non-TPC-H SQL and store `generic_plan` in `TQPPlan`.

**Step 3:** Let `QueryResult.query_id` support `None` or use `0` for generic display if changing dataclass broadly is too invasive. Prefer `int | None` with tests.

**Step 4:** Run focused tests and commit.

### Task 5: Update documentation and run final verification

**Files:**
- Modify: `docs/architecture.md`
- Modify: `README.md`

**Step 1:** Document that the frontend accepts any DuckDB-plannable SQL, while backend generic support is an explicit subset and unsupported operators fail clearly.

**Step 2:** Run:
```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q
timeout 60 /work/torch-query-gpu/.venv/bin/python -m compileall -q tpch_torch scripts
timeout 300 /work/torch-query-gpu/.venv/bin/tpch-torch-validate --db /work/torch-query-gpu/data/tpch_sf1.duckdb --queries all --device cuda --frontend sirius --keep-going
```

**Step 3:** Commit docs, push branch.
