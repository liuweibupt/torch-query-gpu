# Paper-Driven Optimizations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a small verified batch of optimizations from the TQP line and the compressed SQL analytics paper without introducing query-specific fallback scripts.

**Architecture:** Keep SQL admission unchanged: SQL is still planned by DuckDB/Sirius-like JSON physical plans and executed by the physical interpreter. Add reusable tensor primitives under the physical and compressed layers: membership-only semi/anti join probes, sorted/unique-consecutive grouping, and RLE aggregate/compaction primitives. These primitives remain generic and callable by any lowered plan.

**Tech Stack:** Python 3.12, PyTorch tensors on CPU/CUDA, DuckDB JSON physical plans, pytest with 60-second timeout.

---

### Task 1: Semi/anti join membership probe

**Files:**
- Modify: `tests/test_physical_plan.py`
- Create: `tpch_torch/backend/physical_membership.py`
- Modify: `tpch_torch/backend/physical_join.py`

**Step 1: Write failing tests**

Add tests that monkeypatch `tpch_torch.backend.physical_join.join_indices_for_conditions` to raise. Call `semi_join_indices()` and `anti_join_indices()` on duplicate right-side keys and assert the preserved left rows are correct. Before implementation these fail because semi/anti joins expand inner join pairs.

**Step 2: Run red tests**

Run:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q tests/test_physical_plan.py::test_physical_semi_and_anti_join_use_membership_probe_without_pair_expansion
```

Expected: FAIL with the monkeypatch assertion.

**Step 3: Implement membership probe**

Create `physical_membership.py` with tensor `searchsorted` membership over sorted right keys. Use composite key encoding through `torch.unique(..., dim=0, return_inverse=True)` for multi-column conditions. Update `semi_join_indices()` and `anti_join_indices()` to call it.

**Step 4: Run green tests**

Run the same targeted test and then `tests/test_physical_plan.py tests/test_physical_coverage.py`.

---

### Task 2: Sorted group-by fast path

**Files:**
- Modify: `tests/test_physical_plan.py`
- Modify: `tpch_torch/backend/physical_aggregate.py`

**Step 1: Write failing test**

Add a direct `execute_grouped_aggregate()` test whose grouping keys are already lexicographically sorted. Monkeypatch `torch.unique` to raise and assert grouped sums/counts are produced. Before implementation this fails because grouped aggregate always calls `torch.unique`.

**Step 2: Run red test**

Run the single new test with `timeout 60 ... pytest`.

**Step 3: Implement fast path**

Add a grouping helper that detects non-decreasing stacked keys and uses `torch.unique_consecutive(..., dim=0, return_inverse=True)` instead of `torch.unique`. Keep the unsorted path unchanged.

**Step 4: Run green tests**

Run targeted test and `tests/test_physical_plan.py`.

---

### Task 3: Compressed RLE primitives and aggregates

**Files:**
- Modify: `tests/test_compressed.py`
- Modify: `tpch_torch/compressed.py`
- Create: `tpch_torch/compressed_aggregates.py`

**Step 1: Write failing tests**

Add tests for public `range_arange`, `compact_rle`, and RLE aggregate primitives (`rle_count`, `rle_sum`, `rle_min`, `rle_max`, `rle_mean`).

**Step 2: Run red tests**

Run:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q tests/test_compressed.py
```

Expected: FAIL on missing imports/functions.

**Step 3: Implement primitives**

Expose `range_arange`, implement `compact_rle`, and add RLE aggregate helpers that operate on run values and lengths without row expansion.

**Step 4: Run green tests**

Run `tests/test_compressed.py`.

---

### Task 4: Documentation and verification

**Files:**
- Modify: `README.md`
- Modify: `README.zh.md`
- Modify: `docs/operator-roadmap.md`
- Modify: `docs/operator-roadmap.zh.md`
- Modify if needed: `docs/architecture.md`, `docs/architecture.zh.md`

**Step 1: Update docs**

Record that this batch adds membership-only semi/anti probes, sorted group-by fast path, public range_arange/compact_rle, and encoded RLE aggregate primitives.

**Step 2: Verify**

Run:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m compileall -q tpch_torch scripts
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q
```

Expected: all tests pass.

**Step 3: Commit, merge, push**

Commit on the feature branch, fast-forward merge into `main`, rerun verification on `main`, and push `origin main`.
