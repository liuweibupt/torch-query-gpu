# Full TPC-H Graph Node Execution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the Q2-Q22 complex compatibility executor and make all TPC-H Q1-Q22 execute through generic lowered graph nodes.

**Architecture:** Extend `tpch_torch/backend/graph.py` from Q1/Q6 special graph primitives into a real DuckDB JSON physical-plan interpreter over tensor tables. Keep explicit failures for unsupported nodes; do not use DuckDB rows or qXX templates as fallback.

**Tech Stack:** Python dataclasses, DuckDB JSON explain, PyTorch tensor ops, pytest.

---

### Task 1: Add no-compatibility regression tests

**Files:**
- Modify: `tests/test_operator_graph.py`
- Modify: `tests/test_supported_tpch.py`

**Steps:**
1. Add tests asserting graph executor has no `_execute_complex_tpch_graph` and no `_EXECUTOR_BY_QUERY`.
2. Add batch validation test that monkeypatches qXX execute functions to fail for Q2-Q22.
3. Run tests and verify RED.
4. Commit tests if desired after GREEN with implementation.

### Task 2: Add graph execution table model

**Files:**
- Modify: `tpch_torch/backend/graph.py`
- Create if needed: `tests/test_graph_executor.py`

**Steps:**
1. Add failing tests for `GraphTable` projection/filter/join basics.
2. Implement immutable `GraphTable` with columns, aliases, dictionaries, row count.
3. Add expression evaluation for columns, literals, arithmetic, comparisons, boolean ops, `prefix`, `contains`, `CASE`, `substring` subset.
4. Verify targeted tests.

### Task 3: Implement physical nodes batch A

**Files:**
- Modify: `tpch_torch/backend/graph.py`
- Tests: `tests/test_graph_executor.py`, `tests/test_supported_tpch.py`

**Steps:**
1. Implement `SEQ_SCAN` with DuckDB filters/projections.
2. Implement `FILTER`, `PROJECTION`, `HASH_JOIN` inner joins.
3. Implement grouped/ungrouped aggregate, `ORDER_BY`, `TOP_N`.
4. Validate Q3/Q5/Q10/Q12/Q14/Q19 first.
5. Commit.

### Task 4: Implement physical nodes batch B

**Files:**
- Modify: `tpch_torch/backend/graph.py`
- Tests: `tests/test_supported_tpch.py`

**Steps:**
1. Implement semi/anti joins and delimiter-style nodes.
2. Implement CTE/CTE_SCAN and COLUMN_DATA_SCAN as needed.
3. Implement scalar/nested subquery patterns needed by Q2/Q4/Q11/Q15/Q16/Q17/Q18/Q20/Q21/Q22.
4. Validate Q1-Q22.
5. Commit.

### Task 5: Remove compatibility code and update docs

**Files:**
- Modify: `tpch_torch/backend/graph.py`
- Modify: `README.md`, `docs/architecture.zh.md`, `docs/architecture.md`, `docs/operator-roadmap.zh.md`, `docs/operator-roadmap.md`

**Steps:**
1. Delete `_execute_complex_tpch_graph`, `_execute_compiled_tpch_node`, `_EXECUTOR_BY_QUERY`.
2. Update docs to state all Q1-Q22 use generic graph nodes.
3. Run full test suite and compileall.
4. Merge main, push.
