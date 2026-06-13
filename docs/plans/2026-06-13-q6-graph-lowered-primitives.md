# Q6 Graph-Lowered Primitives Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Route default TPC-H Q6 through SQL-lowered `TQPOperatorGraph` and the physical interpreter, while keeping `--compressed-masks` as an explicit experimental primitive path.

**Architecture:** The Sirius-like frontend already emits DuckDB JSON physical graphs for Q6. `PyTorchGraphExecutor` should dispatch Q6 to `execute_physical_plan()` unless `use_compressed_masks=True`; the compressed path remains explicit until compressed storage metadata is represented in the graph. Tests assert the default path uses the physical interpreter and not the old direct primitive.

**Tech Stack:** Python, DuckDB JSON EXPLAIN, PyTorch tensors, pytest.

---

### Task 1: Add Q6 default physical interpreter regression test

**Files:**
- Modify: `tests/test_backend.py`

**Step 1: Write the failing test**

Add a test that compiles canonical TPC-H Q6 SQL with `compile_sirius_plan()`, monkeypatches `tpch_torch.backend.graph.execute_physical_plan` to record calls, monkeypatches `_execute_q6_graph` to raise when called, executes `PyTorchBackend().execute(..., use_compressed_masks=False)`, and asserts physical execution was called with query id 6.

**Step 2: Run test to verify it fails**

Run:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q tests/test_backend.py::test_q6_default_graph_execution_uses_physical_plan_interpreter
```

Expected: FAIL with `q6 direct primitive path used`.

### Task 2: Route default Q6 to physical interpreter

**Files:**
- Modify: `tpch_torch/backend/graph.py`

**Step 1: Implement minimal dispatch change**

Change Q6 dispatch so `_execute_q6_graph()` is called only when `use_compressed_masks=True`. Add Q6 to the physical interpreter query-id set.

**Step 2: Run targeted tests**

Run:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q \
  tests/test_backend.py::test_q6_default_graph_execution_uses_physical_plan_interpreter \
  tests/test_backend.py::test_pytorch_backend_passes_compressed_mask_option_to_q6
```

Expected: PASS.

### Task 3: Verify physical interpreter supports Q6 SQL

**Files:**
- Modify only if tests expose a missing DuckDB physical expression/operator.

**Step 1: Run Q6 validation**

Run:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q 'tests/test_supported_tpch.py::test_all_tpch_queries_validate_through_sirius_frontend[6]'
```

Expected: PASS. If it fails, add the smallest expression/operator support with a failing unit test first.

### Task 4: Update docs

**Files:**
- Modify: `README.md`
- Modify: `README.zh.md`
- Modify: `docs/architecture.md`
- Modify: `docs/architecture.zh.md`
- Modify: `docs/operator-roadmap.md`
- Modify: `docs/operator-roadmap.zh.md`

**Step 1: Update status matrix and flow diagrams**

State that Q1/Q6/Q12/Q14/Q19 default to physical interpreter. State that Q6 compressed masks are an explicit experimental option.

**Step 2: Verify docs contain no stale Q6 default-direct wording**

Run:

```bash
grep -R "Q6.*direct graph primitive\|Q1/Q12/Q14/Q19" -n README.md README.zh.md docs/architecture.md docs/architecture.zh.md docs/operator-roadmap.md docs/operator-roadmap.zh.md
```

Expected: no stale claim that default Q6 is direct primitive.

### Task 5: Full verification, commit, merge, push

**Files:** all changed files.

**Step 1: Run verification in 60s batches**

Run compileall, backend/physical/operator tests, Q6/Q1 supported TPC-H tests, and Q6 CLI validation.

**Step 2: Commit, merge to main, push**

Commit the branch, fast-forward merge to `main`, push `origin main`, and delete the local feature branch.
