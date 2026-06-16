# Q1 Resident Fast Path Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring TPC-H Q1 timing closer to the TQP paper's execution-time setup by separating tensor residency from query execution and reducing Q1 fused intermediate materialization.

**Architecture:** Keep the SQL → DuckDB JSON physical plan → TQPOperatorGraph → PyTorch physical fusion route unchanged. Add an explicit per-connection tensor table cache so hot runs reuse converted `lineitem` tensors, and rewrite Q1 fused aggregation to use masked `torch.bincount` directly instead of `nonzero` + `index_select` for every payload column.

**Tech Stack:** Python 3.12, DuckDB, PyTorch CPU/CUDA tensors, pytest with 60s timeout.

---

### Task 1: Resident tensor cache

**Files:**
- Modify: `tests/test_duckdb_bridge.py`
- Modify: `tpch_torch/duckdb_bridge.py`

**Step 1:** Write failing test that calls `fetch_lineitem_tensor_table()` twice on the same connection/device and monkeypatches the second `con.execute` to fail.

**Step 2:** Run targeted test and observe failure.

**Step 3:** Implement a small explicit per-connection/device cache keyed by DuckDB connection object id and device. Do not change SQL semantics; cold connections still fetch once.

**Step 4:** Run targeted test and bridge tests.

### Task 2: Q1 masked bincount aggregation

**Files:**
- Modify: `tests/test_physical_plan.py`
- Modify: `tpch_torch/backend/physical_fusion.py`

**Step 1:** Write failing test that monkeypatches `torch.nonzero` / gather helper for payload materialization and asserts canonical Q1 fused rows still match.

**Step 2:** Run targeted test and observe failure.

**Step 3:** Rewrite Q1 fused reductions to compute group ids for all rows, then use float weights multiplied by the date mask for sums and date mask as count weights. Avoid selected-row payload gathers.

**Step 4:** Run Q1/fusion tests and full physical plan tests.

### Task 3: Benchmark docs and verification

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture*.md`

**Step 1:** Document that hot benchmark now reuses resident tensors and is closer to TQP's execution-only measurement, while cold still includes DuckDB→tensor conversion.

**Step 2:** Run compileall, pytest, and Q1 CPU/CUDA benchmark smoke.

**Step 3:** Commit, merge to main, push.
