# Sirius Frontend TQP IR Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the default execution path cleanly use a Sirius-like DuckDB frontend that emits a TQP IR, then execute that IR with the PyTorch backend; keep Substrait as a strict experimental frontend.

**Architecture:** Introduce three small layers: `frontend` creates `TQPPlan`, `backend` executes `TQPPlan`, and `runner` orchestrates validation/timing. The first IR version remains query-template based for TPC-H correctness, but all execution goes through `SQL -> frontend -> TQPPlan -> PyTorchBackend`. Substrait stays available as `--frontend substrait`, not as the default path.

**Tech Stack:** Python 3.12, DuckDB, PyTorch, pytest, existing TPC-H executors in `tpch_torch/queries`.

---

### Task 1: Add TQP IR dataclasses

**Files:**
- Create: `tpch_torch/ir/__init__.py`
- Create: `tpch_torch/ir/plan.py`
- Test: `tests/test_tqp_ir.py`

**Step 1: Write the failing test**

```python
from tpch_torch.ir import FrontendName, TQPPlan


def test_tqp_plan_records_frontend_and_query_id():
    plan = TQPPlan(query_id=1, source_sql="select 1", frontend="sirius")

    assert plan.query_id == 1
    assert plan.frontend == "sirius"
    assert plan.plan_json is None
```

**Step 2: Run test to verify it fails**

Run: `timeout 60 .venv/bin/python -m pytest -q tests/test_tqp_ir.py`
Expected: FAIL with missing module/import.

**Step 3: Write minimal implementation**

`tpch_torch/ir/plan.py` should define immutable dataclasses:

```python
from dataclasses import dataclass
from typing import Any, Literal

FrontendName = Literal["sirius", "substrait", "auto"]

@dataclass(frozen=True)
class DuckDBPlanMetadata:
    logical_plan: str = ""
    logical_opt: str = ""
    physical_plan: str = ""

@dataclass(frozen=True)
class TQPPlan:
    query_id: int
    source_sql: str
    frontend: FrontendName
    duckdb_metadata: DuckDBPlanMetadata | None = None
    plan_json: dict[str, Any] | None = None
```

Export these from `tpch_torch/ir/__init__.py`.

**Step 4: Run test to verify it passes**

Run: `timeout 60 .venv/bin/python -m pytest -q tests/test_tqp_ir.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add tpch_torch/ir tests/test_tqp_ir.py
git commit -m "feat: add tqp plan ir"
```

### Task 2: Add frontend package with Sirius-like and Substrait frontends

**Files:**
- Create: `tpch_torch/frontend/__init__.py`
- Create: `tpch_torch/frontend/sirius.py`
- Create: `tpch_torch/frontend/substrait.py`
- Create: `tpch_torch/frontend/auto.py`
- Test: `tests/test_frontend.py`

**Step 1: Write failing tests**

Tests should assert:
- `compile_sirius_plan(con, sql)` calls DuckDB logical planner and returns `TQPPlan(frontend="sirius")`.
- `compile_substrait_plan(con, sql)` calls Substrait exporter and returns `TQPPlan(frontend="substrait", plan_json=...)`.
- `compile_auto_plan` tries Substrait and falls back to Sirius only on `DuckDBSubstraitError`.

**Step 2: Run tests to verify failure**

Run: `timeout 60 .venv/bin/python -m pytest -q tests/test_frontend.py`
Expected: FAIL with missing modules.

**Step 3: Implement minimal frontend functions**

- Move orchestration currently in `runner._admit_plan` into frontend modules.
- Use `identify_tpch_query(sql)` from runner initially to avoid duplicating markers.
- Sirius frontend wraps `export_duckdb_logical_plan` into `DuckDBPlanMetadata`.
- Substrait frontend stores real JSON in `plan_json`.

**Step 4: Run tests**

Run: `timeout 60 .venv/bin/python -m pytest -q tests/test_frontend.py tests/test_runner.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add tpch_torch/frontend tests/test_frontend.py
git commit -m "feat: add tqp frontends"
```

### Task 3: Add PyTorch backend layer

**Files:**
- Create: `tpch_torch/backend/__init__.py`
- Create: `tpch_torch/backend/pytorch.py`
- Modify: `tpch_torch/runner.py`
- Test: `tests/test_backend.py`

**Step 1: Write failing tests**

Test that `PyTorchBackend.execute(plan, con, device="cpu")` dispatches Q1 through the existing executor and raises for unsupported query ids.

**Step 2: Run failing tests**

Run: `timeout 60 .venv/bin/python -m pytest -q tests/test_backend.py`
Expected: FAIL with missing backend module.

**Step 3: Implement backend by moving `_execute_supported_query` and `_compile_q1_plan` from runner**

- Keep function sizes under 100 lines.
- No behavior change.
- Q1 uses `plan.plan_json` when present, otherwise canonical Q1 plan.

**Step 4: Update runner to use frontend + backend**

- `run_sql_with_frontend(..., frontend="sirius")` is the new primary function.
- Keep `run_sql_with_plan_source` as a compatibility wrapper mapping old names to new frontend names.
- `run_sql` should default to `frontend="sirius"`.
- `validate_sql` should default to `frontend="sirius"`.

**Step 5: Run tests**

Run: `timeout 60 .venv/bin/python -m pytest -q tests/test_backend.py tests/test_runner.py tests/test_remaining_tpch.py tests/test_supported_tpch.py`
Expected: PASS.

**Step 6: Commit**

```bash
git add tpch_torch/backend tpch_torch/runner.py tests/test_backend.py
git commit -m "feat: execute tqp plans with pytorch backend"
```

### Task 4: Update CLI from plan-source to frontend while preserving compatibility

**Files:**
- Modify: `scripts/run_query.py`
- Modify: `scripts/validate_query.py`
- Modify: `tests/test_runner_cli.py`
- Modify: `tests/test_validate_query_batch.py`

**Step 1: Write/update failing tests**

Assert parsers accept:

```bash
--frontend sirius|substrait|auto
```

and still accept legacy:

```bash
--plan-source substrait|duckdb-logical|auto
```

Compatibility mapping:
- `duckdb-logical` -> `sirius`
- `substrait` -> `substrait`
- `auto` -> `auto`

**Step 2: Run focused tests**

Run: `timeout 60 .venv/bin/python -m pytest -q tests/test_runner_cli.py tests/test_validate_query_batch.py`
Expected: FAIL until parser/validator is updated.

**Step 3: Implement CLI**

- Add `--frontend` optional argument.
- Keep `--plan-source` hidden or documented as legacy alias.
- Reject conflicting values if both are supplied and map differently.
- Pass frontend to runner.

**Step 4: Run focused tests**

Run: `timeout 60 .venv/bin/python -m pytest -q tests/test_runner_cli.py tests/test_validate_query_batch.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/run_query.py scripts/validate_query.py tests/test_runner_cli.py tests/test_validate_query_batch.py
git commit -m "feat: expose sirius frontend cli"
```

### Task 5: Update support tests and docs to make Sirius frontend default

**Files:**
- Modify: `tests/test_supported_tpch.py`
- Modify: `README.md`

**Step 1: Update tests**

- All Q1-Q22 validation should call the new frontend API or CLI with `frontend="sirius"` by default.
- Keep a strict Substrait test only for DuckDB-exportable query set.

**Step 2: Run tests**

Run: `timeout 60 .venv/bin/python -m pytest -q tests/test_supported_tpch.py`
Expected: PASS.

**Step 3: Update README**

Document default path:

```text
SQL -> Sirius-like DuckDB frontend -> TQP IR -> PyTorch/GPU backend
```

Document experimental path:

```text
SQL -> DuckDB Substrait export -> TQP IR -> PyTorch/GPU backend
```

Use examples with `--frontend sirius` and `--frontend substrait`.

**Step 4: Commit**

```bash
git add README.md tests/test_supported_tpch.py
git commit -m "docs: make sirius frontend the default path"
```

### Task 6: Final verification and push

**Files:**
- No direct code edits expected.

**Step 1: Run all tests**

Run:

```bash
timeout 60 .venv/bin/python -m pytest -q
timeout 60 .venv/bin/python -m compileall -q tpch_torch scripts
```

Expected: PASS.

**Step 2: Run SF1 GPU all-query validation**

Run:

```bash
timeout 300 .venv/bin/tpch-torch-validate --db /work/torch-query-gpu/data/tpch_sf1.duckdb --queries all --device cuda --frontend sirius --keep-going
```

Expected: Q1-Q22 all validated, exit 0.

**Step 3: Push**

```bash
git status --short --branch
git push
```

Expected: branch pushed to `origin/feat/operator-expansion-papers`.
