# P0 Correctness and dtype Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete README P0 correctness TODOs for join key dtype handling, integer min/max reduction initialization, and `PhysicalValue.valid`-based NULL semantics.

**Architecture:** Keep the current `PhysicalValue` physical interpreter model. Add small dtype/validity helpers in `physical_join.py`, `physical_membership.py`, `physical_aggregate.py`, and `physical_expr.py`; write focused regression tests before production changes.

**Tech Stack:** Python 3.12, PyTorch tensor operators, DuckDB physical JSON interpreter, pytest.

---

### Task 1: Join key dtype correctness

**Files:**
- Modify: `tpch_torch/backend/physical_join.py`
- Modify: `tpch_torch/backend/physical_membership.py`
- Test: `tests/test_p0_correctness.py`

**Steps:**
1. Add failing tests for `inner_join_indices()` with float keys where truncation would over-match.
2. Add failing test for metadata-backed sorted-unique float build keys.
3. Add failing test for SEMI membership with float keys.
4. Replace `.to(dtype=torch.int64)` casts on single join keys with a shared comparable dtype helper.
5. Replace composite-key packed `int64` path with first-condition join plus tensor filtering for remaining conditions.
6. Run focused tests.

### Task 2: Integer MIN/MAX scatter reduce

**Files:**
- Modify: `tpch_torch/backend/physical_aggregate.py`
- Test: `tests/test_p0_correctness.py`

**Steps:**
1. Add failing grouped aggregate test for integer `min`/`max`.
2. Add dtype-aware min/max fill helper using `torch.iinfo()` for integer tensors.
3. Run focused tests.

### Task 3: Basic NULL-aware boolean and aggregate semantics

**Files:**
- Modify: `tpch_torch/backend/physical_expr.py`
- Modify: `tpch_torch/backend/physical_aggregate.py`
- Test: `tests/test_p0_correctness.py`

**Steps:**
1. Add failing tests for `NULL OR TRUE`, `NULL AND FALSE`, `NOT NULL`, and `CASE WHEN NULL` using `PhysicalValue.valid`.
2. Add failing tests for scalar/grouped `SUM/MIN/MAX/AVG/COUNT(DISTINCT)` ignoring invalid rows and returning invalid/null for all-invalid input.
3. Implement SQL three-valued boolean helpers over tensor+valid pairs.
4. Implement aggregate validity filtering and invalid-result masks.
5. Run focused tests.

### Task 4: Docs, full verification, integration

**Files:**
- Modify: `README.md`
- Modify: `docs/operator-roadmap.zh.md`
- Modify: `docs/operator-roadmap.md`

**Steps:**
1. Mark P0 items as complete in README TODO and roadmap notes.
2. Run `git diff --check`.
3. Run `timeout 60 /work/torch-query-gpu/.venv/bin/python -m compileall -q tpch_torch scripts`.
4. Run `timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q`.
5. Commit, merge to main, rerun tests, push `main`.
