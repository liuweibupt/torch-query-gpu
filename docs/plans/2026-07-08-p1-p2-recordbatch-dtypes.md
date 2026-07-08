# P1/P2 TensorRecordBatch and Multi-Precision Operators Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the P1 type-system substrate and the P2 first multi-precision join/aggregate support, including DECIMAL as scaled int64.

**Architecture:** Introduce `tpch_torch/record_batch.py` and `tpch_torch/backend/type_mapping.py`; add optional `ColumnMeta` to `PhysicalValue`; use metadata in physical scan, expressions, joins, and aggregates while preserving existing APIs.

**Tech Stack:** Python 3.12, PyTorch tensors, DuckDB type strings, pytest.

---

### Task 1: Record batch and metadata substrate

**Files:**
- Create: `tpch_torch/record_batch.py`
- Test: `tests/test_record_batch.py`

**Steps:**
1. Write failing tests for `ColumnMeta`, decimal meta construction, and `TensorRecordBatch.filter/gather/project`.
2. Implement `LogicalDType`, `ColumnMeta`, `TensorRecordBatch`.
3. Verify focused tests.
4. Commit.

### Task 2: DuckDB dtype mapping and decimal encoding

**Files:**
- Create: `tpch_torch/backend/type_mapping.py`
- Modify: `tpch_torch/backend/physical_scan.py`
- Test: `tests/test_type_mapping.py`

**Steps:**
1. Write failing tests for DuckDB type strings: BIGINT, INTEGER, FLOAT, DOUBLE, BOOLEAN, DATE, VARCHAR, DECIMAL(p,s).
2. Write failing tests for encoding Python `Decimal` / numeric values into scaled int64.
3. Implement mapping and decimal helpers.
4. Attach `ColumnMeta` to `PhysicalValue` during physical scan.
5. Verify focused tests.
6. Commit.

### Task 3: Decimal projection and metadata propagation

**Files:**
- Modify: `tpch_torch/backend/physical_types.py`
- Modify: `tpch_torch/backend/physical_expr.py`
- Test: `tests/test_decimal_physical.py`

**Steps:**
1. Write failing tests for decimal `cell()` decoding and filter/gather metadata preservation.
2. Write failing tests for decimal +, -, *, / metadata and values.
3. Implement optional `meta` on `PhysicalValue` and decimal arithmetic helpers.
4. Verify focused tests.
5. Commit.

### Task 4: P2 multi-precision join and aggregate

**Files:**
- Modify: `tpch_torch/backend/physical_key_ops.py`
- Modify: `tpch_torch/backend/physical_aggregate.py`
- Create: `tpch_torch/backend/physical_hash_join.py`
- Test: `tests/test_p2_multi_precision.py`

**Steps:**
1. Write failing tests for INT64/FP32/FP64/DECIMAL joins.
2. Write failing tests for grouped SUM over INT64/FP32/FP64/DECIMAL.
3. Write failing tests for first hash-style join probe helper on CPU/CUDA-capable tensors.
4. Implement decimal scale alignment in key comparisons.
5. Preserve decimal metadata in SUM; AVG remains fp64.
6. Implement `hash_join_indices_for_values()` as explicit first tensor dictionary-encoding prototype.
7. Verify focused tests.
8. Commit.

### Task 5: Docs, full verification, integration

**Files:**
- Modify: `README.md`
- Modify: `docs/operator-roadmap.zh.md`
- Modify: `docs/operator-roadmap.md`

**Steps:**
1. Mark P1 completed and P2 first implementation completed/partially scoped.
2. Document DECIMAL int64+scale and TensorRecordBatch shape.
3. Run `git diff --check`.
4. Run `timeout 60 /work/torch-query-gpu/.venv/bin/python -m compileall -q tpch_torch scripts`.
5. Run `timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q`.
6. Merge to `main`, rerun tests, push.
