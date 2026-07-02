# Join Metadata Optimizations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add explicit sorted/unique metadata propagation and use it to optimize physical joins/sorts/group-by in the DuckDB physical-plan → PyTorch path.

**Architecture:** `PhysicalValue` carries immutable sorted/unique flags shared by aliases. Scan/filter/sort/group-by update those flags with conservative rules. Single-key joins consume the metadata to skip redundant build-side sort/duplicate checks when the right/build key is known sorted and unique.

**Tech Stack:** Python 3.12, DuckDB Python API, PyTorch tensor operators, pytest.

---

### Task 0: Stabilize DuckDB helper baseline under container PID pressure

**Files:**
- Modify: `tpch_torch/duckdb_bridge.py`
- Modify: `tests/test_duckdb_bridge.py`
- Modify: `README.md`

**Step 1: Write failing tests**

- Keep `test_load_substrait_extension_uses_default_install_when_env_unset` on DuckDB's Python extension API and expect:
  - `install_extension("substrait", repository="community")`
  - `load_extension("substrait")`
- Add a test where `load_extension("substrait")` raises `RuntimeError("Resource temporarily unavailable")` and assert `_load_substrait_extension()` raises `DuckDBSubstraitError` with the visible resource message.
- Add a test where `execute("load substrait")` raises `RuntimeError("Resource temporarily unavailable")` and assert `_load_substrait_extension()` raises `DuckDBSubstraitError` with the visible resource message.
- Add a test where `generate_tpch()` records `pragma threads=1` before `call dbgen` when `TQG_DUCKDB_THREADS` is unset.

**Step 2: Verify RED**

Run:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_duckdb_bridge.py::test_load_substrait_extension_uses_default_install_when_env_unset tests/test_duckdb_bridge.py::test_load_substrait_extension_wraps_runtime_error tests/test_duckdb_bridge.py::test_generate_tpch_sets_default_helper_threads -q
```

Expected: fail because default loader still calls Python extension methods and `generate_tpch()` does not set helper threads.

**Step 3: Implement minimal code**

- Keep default Substrait loading on `install_extension()` / `load_extension()` to avoid SQL `LOAD` aborts seen with a `duckdb`-then-`torch` import order.
- Catch `(duckdb.Error, RuntimeError)` in `_load_substrait_extension()`.
- Add `_duckdb_helper_threads()` reading `TQG_DUCKDB_THREADS`, validating positive integer, defaulting to `1`.
- Set `pragma threads=<value>` before `dbgen`.

**Step 4: Verify GREEN**

Run the focused command again, then rerun the previously failing baseline tests:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_duckdb_bridge.py::test_export_substrait_json_uses_duckdb_extension_or_reports_unavailable tests/test_physical_coverage.py::test_tpch_physical_coverage_reports_all_tpch_queries_supported -q
```

**Step 5: Commit**

```bash
git add tpch_torch/duckdb_bridge.py tests/test_duckdb_bridge.py README.md docs/plans/2026-07-02-join-metadata-optimizations-design.md docs/plans/2026-07-02-join-metadata-optimizations.md
git commit -m "test: stabilize duckdb helper resources"
```

### Task 1: Add PhysicalValue metadata and propagation tests

**Files:**
- Modify: `tpch_torch/backend/physical_types.py`
- Modify: `tpch_torch/backend/physical_scan.py`
- Modify: `tpch_torch/backend/physical.py`
- Modify: `tpch_torch/backend/physical_aggregate.py`
- Test: `tests/test_physical_types.py`
- Test: `tests/test_physical_plan.py`

**Step 1: Write failing tests**

- Assert `PhysicalValue(..., sorted_non_decreasing=True, unique=True).filter(mask)` preserves both flags.
- Assert `.gather(indices)` drops both flags.
- Assert `fetch_physical_table()` marks `region.r_regionkey` and `rowid` as sorted/unique on generated TPC-H data.
- Assert `_sort_table()` marks a single ascending sorted key as `sorted_non_decreasing=True` and `unique` when the source key was unique.
- Assert grouped aggregate output group keys are `unique=True`.

**Step 2: Verify RED**

Run focused pytest for the new tests.

**Step 3: Implement minimal code**

- Add fields to `PhysicalValue` with default `False`.
- Preserve/drop metadata in `filter`, `gather`, `gather_optional`.
- Add scan metadata helpers with conservative validation checks.
- Mark rowid metadata.
- After sort, refresh the sorted key value metadata for single ascending key sorts.
- Mark aggregate group key outputs unique; sorted if unique keys came from sorted-input path.

**Step 4: Verify GREEN**

Run focused tests plus `tests/test_physical_plan.py -q`.

**Step 5: Commit**

```bash
git add tpch_torch/backend tests
git commit -m "feat: propagate physical column metadata"
```

### Task 2: Use metadata in PK/FK join fast path

**Files:**
- Modify: `tpch_torch/backend/physical_join.py`
- Modify: `tests/test_physical_plan.py` or create `tests/test_physical_join_metadata.py`

**Step 1: Write failing tests**

- Build left probe tensor with duplicate FK keys and right build tensor tagged `sorted_non_decreasing=True, unique=True`.
- Monkeypatch `torch.argsort` inside the test to raise if called.
- Call `join_indices_for_conditions()` and assert correct row pairs.
- Add missing-key coverage: unmatched probe keys are ignored.

**Step 2: Verify RED**

Expected: current value-blind path calls `torch.argsort` or performs dynamic checks, so the monkeypatch test fails.

**Step 3: Implement minimal code**

- Add `inner_join_indices_for_values(left_value, right_value)`.
- If `right_value.sorted_non_decreasing and right_value.unique`, search the right tensor directly and call `_unique_build_join_indices()` with `right_order=None`.
- Keep the existing generic path unchanged for absent metadata and composite keys.

**Step 4: Verify GREEN**

Run focused join tests and representative physical plan tests.

**Step 5: Commit**

```bash
git add tpch_torch/backend/physical_join.py tests
git commit -m "feat: use sorted unique metadata in joins"
```

### Task 3: Docs, roadmap, benchmark notes

**Files:**
- Modify: `README.md`
- Modify: `docs/operator-roadmap.md`
- Modify: `docs/operator-roadmap.zh.md`
- Create or modify: `docs/performance-notes.zh.md` if needed

**Step 1: Update docs**

- Document `TQG_DUCKDB_THREADS` as a helper-data-generation resource knob.
- Add a short architecture note: physical values carry sorted/unique metadata and joins consume it.
- Update roadmap checkboxes for this batch.

**Step 2: Run benchmarks if database exists**

```bash
if [ -f /work/torch-query-gpu/data/tpch_sf1.duckdb ]; then
  timeout 60 /work/torch-query-gpu/.venv/bin/python -m scripts.benchmark_query --db /work/torch-query-gpu/data/tpch_sf1.duckdb --query 14 --device cpu --cold-runs 0 --warmup-runs 1 --hot-runs 3
  timeout 60 /work/torch-query-gpu/.venv/bin/python -m scripts.benchmark_query --db /work/torch-query-gpu/data/tpch_sf1.duckdb --query 19 --device cpu --cold-runs 0 --warmup-runs 1 --hot-runs 3
fi
```

**Step 3: Commit**

```bash
git add README.md docs
git commit -m "docs: describe join metadata optimizations"
```

### Task 4: Final verification and integration

**Step 1: Verify**

```bash
git diff --check
timeout 60 /work/torch-query-gpu/.venv/bin/python -m compileall -q tpch_torch scripts
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q
```

**Step 2: Merge to main and push**

```bash
cd /work/torch-query-gpu
git pull --ff-only origin main
git merge --no-ff feat/join-metadata-optimizations -m "merge: join metadata optimizations"
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q
git push origin main
git worktree remove /work/torch-query-gpu/.worktrees/join-metadata-optimizations
git branch -d feat/join-metadata-optimizations
```
