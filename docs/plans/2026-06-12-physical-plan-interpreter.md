# DuckDB Physical Plan Interpreter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a correctness-first DuckDB JSON physical-plan interpreter so arbitrary SQL joins/aggregates/order/limit can execute through PyTorch tensors, and migrate the first TPC-H Q2-Q22 queries away from query-id recipes.

**Architecture:** Keep Sirius as the DuckDB-planned frontend. Add `tpch_torch.backend.physical` that interprets `TQPOperatorGraph` nodes (`SEQ_SCAN`, `FILTER`, `PROJECTION`, `HASH_JOIN`, aggregate, `ORDER_BY`, `TOP_N`) with PyTorch tensor tables. Route non-TPC-H SQL and the first migrated TPC-H queries (Q12/Q19) through this interpreter; keep Q1/Q6 optimized paths and unsupported complex TPC-H recipes explicit.

**Tech Stack:** DuckDB `EXPLAIN (FORMAT JSON)`, PyTorch tensor operations, existing `TensorTable` column encoding, pytest.

---

### Task 1: RED tests for generic physical joins and migrated TPC-H queries

**Files:**
- Create: `tests/test_physical_plan.py`

**Steps:**
1. Add a test that runs `select a, name from t join u on t.id = u.id order by a` through `run_sql` and monkeypatches the old generic executor path to fail.
2. Add a test that validates `select name, sum(amount) as total ... group by name order by total desc` through `validate_sql`.
3. Add tests that monkeypatch `tpch_graph_q12.execute_q12_graph` and `tpch_graph_q19.execute_q19_graph` to fail, then validate Q12/Q19 through the Sirius frontend on SF=0.01.
4. Run `timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q tests/test_physical_plan.py` and confirm the tests fail because the physical interpreter is missing/not routed.

### Task 2: Implement physical table and expression interpreter

**Files:**
- Create: `tpch_torch/backend/physical.py`

**Steps:**
1. Add immutable `PhysicalTable` and `PhysicalValue` helpers carrying tensor columns, visible column order, dictionaries, and date columns.
2. Implement expression evaluation for column refs, `#N`, literals, arithmetic, comparisons, `AND`/`OR`/`NOT`, `IN`, `prefix`/`contains`/`suffix`, `CASE WHEN`, and DuckDB internal compress/decompress wrappers.
3. Implement projection/filter helpers over tensors.
4. Run the RED tests and keep expected failures limited to missing node execution/routing.

### Task 3: Implement physical node execution and routing

**Files:**
- Modify: `tpch_torch/backend/physical.py`
- Modify: `tpch_torch/backend/graph.py`

**Steps:**
1. Interpret scan, filter, projection, hash join, grouped/ungrouped aggregate, order by, and top-n nodes.
2. Convert final tensor output to row dictionaries using `DESCRIBE <sql>` aliases, never DuckDB result rows.
3. Route query-id `None` and migrated TPC-H `{12, 19}` through the physical interpreter.
4. Run `timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q tests/test_physical_plan.py` and confirm green.

### Task 4: Update docs and verification

**Files:**
- Modify: `README.md`
- Modify: `README.zh.md`
- Modify: `docs/architecture.md`
- Modify: `docs/architecture.zh.md`
- Modify: `docs/operator-roadmap.md`
- Modify: `docs/operator-roadmap.zh.md`

**Steps:**
1. Document Q1 benchmark comparison and the physical interpreter route.
2. Mark Q12/Q19 as physical-interpreter migrated and keep remaining TPC-H recipe boundary explicit.
3. Run `timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q` and `timeout 60 /work/torch-query-gpu/.venv/bin/python -m compileall -q tpch_torch scripts`.
4. Commit and push the branch.
