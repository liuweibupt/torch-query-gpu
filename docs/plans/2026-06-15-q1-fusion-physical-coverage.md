# Q1 Fusion and Physical Coverage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a graph-lowered fused Q1 primitive and a TPC-H physical-interpreter coverage probe without reintroducing query-id template execution.

**Architecture:** `execute_physical_plan()` first asks `physical_fusion.try_execute_fused_physical_plan()` whether the lowered graph has a supported fusion. The initial fusion recognizes canonical TPC-H Q1 from the DuckDB physical graph and executes the heavy tensor work in a single fused helper. Separately, a physical coverage probe invokes `execute_physical_plan()` directly for each TPC-H query to track automatic operator coverage.

**Tech Stack:** Python, DuckDB JSON EXPLAIN, PyTorch tensor ops, pytest, existing benchmark scripts.

---

### Task 1: Add Q1 fusion entry regression test

**Files:**
- Modify: `tests/test_physical_plan.py`

**Step 1: Write the failing test**

Add `test_q1_physical_plan_uses_graph_lowered_fusion` that builds a tiny `lineitem` fixture, compiles Q1 through `compile_sirius_plan()`, monkeypatches `tpch_torch.backend.physical_fusion.try_execute_fused_physical_plan` to record query id and return a sentinel row, then calls `execute_physical_plan()` and asserts the sentinel row is returned.

**Step 2: Run RED**

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q tests/test_physical_plan.py::test_q1_physical_plan_uses_graph_lowered_fusion
```

Expected: FAIL because `physical_fusion` does not exist or is not called.

### Task 2: Add fusion module and call site

**Files:**
- Create: `tpch_torch/backend/physical_fusion.py`
- Modify: `tpch_torch/backend/physical.py`

**Step 1: Implement minimal fusion hook**

Create `try_execute_fused_physical_plan(con, graph, device)` returning `None` by default. In `PhysicalPlanExecutor.execute()`, call it before interpreting nodes and return fused rows when not `None`.

**Step 2: Run GREEN for hook test**

Run the RED test again; expected PASS.

### Task 3: Implement canonical Q1 fused primitive

**Files:**
- Modify: `tpch_torch/backend/physical_fusion.py`
- Test: `tests/test_physical_plan.py`

**Step 1: Write correctness test**

Add `test_q1_fused_physical_plan_matches_duckdb_fixture` that validates Q1 fixture output through `validate_sql_with_frontend(..., frontend="sirius")` while monkeypatching the normal interpreter path to fail after fusion recognition if needed.

**Step 2: Implement fused Q1**

Use existing tensor fetch/encoding utilities. Execute filter, grouped reductions, decode, and sort. Keep function bodies under 100 lines; split helpers.

**Step 3: Run targeted tests**

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q tests/test_physical_plan.py::test_q1_fused_physical_plan_matches_duckdb_fixture tests/test_backend.py::test_q1_graph_execution_uses_physical_plan_interpreter
```

### Task 4: Add physical-only TPC-H coverage probe

**Files:**
- Create: `tpch_torch/physical_coverage.py`
- Create or modify: `tests/test_physical_coverage.py`
- Update docs: `README.md`, `README.zh.md`, `docs/architecture*.md`, `docs/operator-roadmap*.md`

**Step 1: Write test**

Test that coverage over SF=0.01 reports Q1/Q6/Q12/Q14/Q19 as supported and includes explicit unsupported reasons for at least one remaining recipe-backed query.

**Step 2: Implement probe**

Compile each query with `compile_sirius_plan()`, call `execute_physical_plan()` directly, catch `UnsupportedPlanError`, and return structured records. Do not use backend graph recipe dispatch.

### Task 5: Q1 benchmark and documentation

**Files:**
- Modify docs only after benchmark.

**Step 1: Run Q1 smoke benchmark**

Use existing `scripts.benchmark_query` with `--query 1 --frontend sirius --device cpu --cold-runs 1 --warmup-runs 1 --hot-runs 3`.

**Step 2: Update docs**

Record that Q1 now has graph-lowered fusion; include benchmark command and observed smoke value.

### Task 6: Full verification, commit, merge, push

Run compileall and tests in 60s batches, validate Q1/Q6/Q12/Q14/Q19, commit, merge to `main`, push `origin main`, and clean the worktree/branch if merged.
