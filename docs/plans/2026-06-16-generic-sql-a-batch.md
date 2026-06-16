# Generic SQL A Batch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Roadmap A batch: Basic `HAVING`, generic searched `CASE`, and tensor top-k for `ORDER BY ... LIMIT` in the DuckDB physical-plan interpreter.

**Architecture:** Extend `tpch_torch/backend/physical_expr.py` and `tpch_torch/backend/physical.py` so DuckDB physical nodes continue to drive execution. Use tests comparing PyTorch backend output against DuckDB baseline; unsupported shapes must keep raising explicit errors rather than falling back to DuckDB query results.

**Tech Stack:** Python 3.10+, DuckDB physical JSON plans, PyTorch tensor operators, pytest.

---

### Task 1: HAVING via aggregate-output filter expressions

**Files:**
- Modify: `tpch_torch/backend/physical_expr.py`
- Modify: `tpch_torch/backend/physical_join.py` only if alias helper needs reuse extraction
- Test: `tests/test_physical_plan.py` or nearest existing physical interpreter test file

**Steps:**
1. Add a failing test for `select l_returnflag, sum(l_quantity) as total_qty from lineitem group by l_returnflag having sum(l_quantity) > 40 order by l_returnflag` on a small fixture.
2. Run that single test and confirm it fails because aggregate expression lookup is unsupported.
3. Implement minimal aggregate alias lookup in expression evaluation for aggregate output tables.
4. Re-run targeted test and then relevant physical tests.

### Task 2: Generic searched CASE expression

**Files:**
- Modify: `tpch_torch/backend/physical_expr_parse.py`
- Modify: `tpch_torch/backend/physical_expr.py`
- Test: `tests/test_physical_plan.py` or nearest existing physical interpreter test file

**Steps:**
1. Add a failing test for `select case when l_quantity < 10 then 1 when l_quantity < 20 then 2 else 3 end as bucket, count(*) as n from lineitem group by bucket order by bucket`.
2. Confirm the test fails because multi-branch CASE is unsupported.
3. Extend CASE parser/evaluator for searched CASE branches using nested `torch.where` from last branch to first branch.
4. Re-run targeted tests.

### Task 3: ORDER BY LIMIT tensor top-k path

**Files:**
- Modify: `tpch_torch/backend/physical.py`
- Test: `tests/test_physical_plan.py` or nearest existing physical interpreter test file

**Steps:**
1. Add a failing unit test that monkeypatches/targets `_execute_limit` or observes `torch.topk` use for single-key `ORDER BY l_quantity DESC LIMIT 3` while comparing rows to DuckDB.
2. Confirm it fails because current path uses full sort only.
3. Implement `_try_topk_limit()` for single-key order items and positive limit smaller than row count.
4. Re-run targeted tests and full suite.

### Task 4: Documentation and Roadmap update

**Files:**
- Modify: `README.md`
- Modify: `docs/operator-roadmap.md`
- Modify: `docs/operator-roadmap.zh.md`

**Steps:**
1. Mark Basic `HAVING`, Generic `CASE`, and Tensor top-k integration as complete for current supported subset.
2. Add a short README bullet under current status / optimization notes.
3. Run `git diff --check`, `compileall`, and full pytest.
