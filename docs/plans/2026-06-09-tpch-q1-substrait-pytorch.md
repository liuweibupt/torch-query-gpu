# TPC-H Q1 Substrait PyTorch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a repository initialization that runs TPC-H Q1 from SQL through DuckDB Substrait JSON into PyTorch tensor operators, with DuckDB validation and SF1 scripts.

**Architecture:** DuckDB owns SQL parsing and Substrait JSON export. A narrow compiler validates the Q1 plan shape and constructs a `Q1Plan`, then PyTorch executes the supported filter/project/groupby/aggregate/order pipeline on a columnar tensor table. No fallback execution path bypasses Substrait compilation.

**Tech Stack:** Python 3.10+, PyTorch, DuckDB, pytest, optional CUDA.

---

### Task 1: Project skeleton and dependency metadata

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `.gitignore`
- Create: `tpch_torch/__init__.py`
- Create: `tests/__init__.py`

**Steps:**
1. Add Python package metadata with runtime dependencies `duckdb` and `torch`, dev dependency `pytest`, and console scripts.
2. Add README describing SQL → DuckDB/Substrait → PyTorch flow and SF1 usage.
3. Add `.gitignore` for virtualenvs, caches, DuckDB databases, and generated data.
4. Run `python3 -m compileall tpch_torch`.
5. Commit skeleton.

### Task 2: Canonical Q1 SQL and Substrait compiler tests

**Files:**
- Create: `tpch_torch/sql.py`
- Create: `tpch_torch/substrait.py`
- Create: `tests/test_substrait.py`

**Steps:**
1. Write failing tests for canonical Q1 SQL and `compile_q1_substrait_plan` requiring read/filter/aggregate/sort nodes.
2. Run targeted pytest and confirm failure.
3. Implement `TPC_H_Q1_SQL`, `UnsupportedPlanError`, `Q1Plan`, and plan inspection helpers.
4. Run targeted pytest and confirm pass.
5. Commit.

### Task 3: Tensor storage and Q1 operator tests

**Files:**
- Create: `tpch_torch/storage.py`
- Create: `tpch_torch/operators.py`
- Create: `tpch_torch/queries/__init__.py`
- Create: `tpch_torch/queries/q01.py`
- Create: `tests/test_q01.py`

**Steps:**
1. Write failing test comparing Q1 PyTorch results on a small fixture with expected DuckDB-style aggregates.
2. Run targeted pytest and confirm failure.
3. Implement columnar tensor conversion, dictionary encoding, groupby reductions, average derivation, and result sorting.
4. Run targeted pytest and confirm pass.
5. Commit.

### Task 4: DuckDB bridge and validation runner

**Files:**
- Create: `tpch_torch/duckdb_bridge.py`
- Create: `tpch_torch/validate.py`
- Create: `tests/test_duckdb_bridge.py`

**Steps:**
1. Write failing tests for small DuckDB lineitem creation, Substrait export, DuckDB baseline Q1, and PyTorch validation.
2. Run targeted pytest and confirm failure.
3. Implement bridge functions and validation comparison with numeric tolerance.
4. Run targeted pytest and confirm pass.
5. Commit.

### Task 5: Scripts and documentation for SF1

**Files:**
- Create: `scripts/gen_sf1.py`
- Create: `scripts/export_q1_substrait.py`
- Create: `scripts/run_q1.py`
- Create: `scripts/validate_q1.py`
- Modify: `README.md`

**Steps:**
1. Add scripts for generating SF1, exporting Q1 Substrait JSON, running Q1, and validating Q1.
2. Document exact commands and CUDA behavior.
3. Run script help commands.
4. Commit.

### Task 6: Final verification and push

**Steps:**
1. Install package in a virtualenv.
2. Run `timeout 60 python3 -m pytest -q`.
3. Run `python3 -m compileall tpch_torch scripts`.
4. Run a small end-to-end validation database command.
5. Check `git status`.
6. Push commits to `origin main`.
