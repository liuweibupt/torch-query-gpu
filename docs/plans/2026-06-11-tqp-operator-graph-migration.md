# TQP Operator Graph Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate TPC-H execution from backend query-id template dispatch to an explicit SQL/DuckDB-plan-lowered TQP operator graph executed by a PyTorch graph executor.

**Architecture:** Sirius-like frontend exports DuckDB `EXPLAIN (FORMAT JSON)`, lowers the JSON into `TQPOperatorGraph`, stores it on `TQPPlan`, and `PyTorchBackend` delegates graph execution to `PyTorchGraphExecutor`. Q1/Q6 and generic single-table SQL use real reusable graph primitives first; complex TPC-H queries are represented as explicit graph nodes until their joins/subqueries are incrementally lowered.

**Tech Stack:** Python dataclasses, DuckDB JSON explain, PyTorch eager tensor ops, pytest.

---

### Task 1: Add operator graph IR tests

**Files:**
- Create: `tests/test_operator_graph.py`
- Create: `tpch_torch/operator_graph.py`

**Steps:**
1. Write failing tests for immutable graph/node construction and root lookup.
2. Implement minimal dataclasses.
3. Verify targeted tests pass.
4. Commit.

### Task 2: Add DuckDB JSON plan export and frontend lowering tests

**Files:**
- Create: `tpch_torch/duckdb_plan_json.py`
- Modify: `tpch_torch/frontend/sirius.py`
- Modify: `tpch_torch/ir/plan.py`
- Test: `tests/test_frontend.py`, `tests/test_operator_graph.py`

**Steps:**
1. Test that Q1-Q22 Sirius plans all carry `operator_graph`.
2. Test root/source fields and DuckDB node names are present.
3. Implement JSON export and plan lowering.
4. Verify targeted tests pass.
5. Commit.

### Task 3: Add graph executor and route backend through graph

**Files:**
- Create: `tpch_torch/backend/graph.py`
- Modify: `tpch_torch/backend/pytorch.py`
- Test: `tests/test_backend.py`, `tests/test_supported_tpch.py`, `tests/test_runner.py`

**Steps:**
1. Write failing test that monkeypatching qXX module import does not affect backend dispatch when graph exists.
2. Implement `PyTorchGraphExecutor` and make backend require graph for TPC-H.
3. Move existing qXX calls behind explicit graph node execution for compatibility.
4. Verify Q1-Q22 validation tests still pass.
5. Commit.

### Task 4: Implement real single-table graph primitives for Q1/Q6/generic subset

**Files:**
- Modify: `tpch_torch/backend/graph.py`
- Modify: `tpch_torch/frontend/sirius.py`
- Test: `tests/test_operator_graph.py`, `tests/test_q01.py`, `tests/test_q06.py`, `tests/test_runner_generic_sql.py`

**Steps:**
1. Write tests that Q1/Q6 graph has real scan/filter/aggregate/order nodes, not only compiled template nodes.
2. Implement single-table graph execution by delegating through generic graph primitives or existing reusable operators.
3. Verify Q1/Q6 validation passes and generic SQL subset still passes.
4. Commit.

### Task 5: Update docs and final verification

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.zh.md`
- Modify: `docs/operator-roadmap.zh.md`
- Modify: `docs/architecture.md`
- Modify: `docs/operator-roadmap.md`

**Steps:**
1. Document new operator graph path and remaining compiled complex node migration work.
2. Run full test suite and compileall.
3. Merge main, push.
