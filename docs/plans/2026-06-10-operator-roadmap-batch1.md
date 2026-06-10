# Operator Roadmap Batch 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Document the paper-grounded TODO and implement the first reusable PyTorch/GPU operator primitives.

**Architecture:** The Sirius-like DuckDB frontend and `TQPPlan` boundary remain unchanged. New operators are lower-layer tensor helpers: plain relational primitives in `tpch_torch/operators.py` and compressed RLE mask primitives in `tpch_torch/compressed.py`.

**Tech Stack:** Python 3.12, PyTorch tensors, pytest, DuckDB-backed existing tests.

---

### Task 1: Paper-grounded roadmap docs

**Files:**
- Create: `docs/operator-roadmap.md`
- Modify: `docs/papers/README.md`
- Modify: `README.md`
- Modify: `docs/architecture.md`

**Steps:**
1. Record source status for TQP, TQEx, TQP++, CoddSpeed, and compressed SQL analytics.
2. List verified TQP operators and tensor operations.
3. List verified compressed-data primitives, logical operations, alignment, group-by, join, and appendix optimization rules.
4. Mark abstract-only papers as pending full-text/appendix extraction.
5. Add links from README and architecture docs.

### Task 2: Plain tensor operator tests

**Files:**
- Create: `tests/test_operators.py`

**Steps:**
1. Write tests for logical mask composition and masked gather.
2. Write tests for grouped min/max/mean and missing-group errors.
3. Write tests for top-k index validation.
4. Run `timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_operators.py -q` and verify it fails because functions do not exist yet.

### Task 3: Plain tensor operator implementation

**Files:**
- Modify: `tpch_torch/operators.py`

**Steps:**
1. Add explicit validation helpers.
2. Implement logical mask composition and masked gather.
3. Implement grouped min/max/mean using PyTorch reductions.
4. Implement top-k index selection.
5. Run `timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_operators.py -q` and verify it passes.

### Task 4: Compressed RLE primitive tests

**Files:**
- Create: `tests/test_compressed.py`

**Steps:**
1. Write tests for `plain_to_rle` and `rle_to_index`.
2. Write tests for `range_intersect` and `range_union`.
3. Write tests for `complement_rle` and malformed ranges.
4. Run `timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_compressed.py -q` and verify it fails because module/functions do not exist yet.

### Task 5: Compressed RLE primitive implementation

**Files:**
- Create: `tpch_torch/compressed.py`

**Steps:**
1. Add immutable `RLERanges` with validation and length helper.
2. Implement plain/RLE conversion helpers.
3. Implement range intersection, union, and complement.
4. Run `timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_compressed.py -q` and verify it passes.

### Task 6: Final verification and git

**Steps:**
1. Run full tests: `timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q`.
2. Run compile check: `timeout 60 /work/torch-query-gpu/.venv/bin/python -m compileall -q tpch_torch scripts`.
3. Review `git diff`.
4. Commit and push branch.
