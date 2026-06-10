# Native DuckDB Substrait Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the user-selected B方案 by keeping original SQL untouched, making native DuckDB Substrait support probeable, and preparing PyTorch operator work only after original SQL exports real Substrait.

**Architecture:** Add a diagnostic capability-probe layer and an explicit local-extension load hook. Existing runners still call `get_substrait_json(original_sql)` and fail clearly for unsupported native exports; no SQL rewrite or DuckDB-result fallback is introduced.

**Tech Stack:** Python 3.12, DuckDB Python package, DuckDB Substrait extension, PyTorch, pytest, CLI scripts.

---

### Task 1: Add native Substrait capability probe data model

**Files:**
- Create: `tpch_torch/capabilities.py`
- Test: `tests/test_capabilities.py`

**Step 1: Write the failing test**

```python
from tpch_torch.capabilities import QueryExportStatus


def test_query_export_status_serializes_failure():
    status = QueryExportStatus(
        query_id=16,
        export_ok=False,
        error_type="DuckDBSubstraitError",
        error_message="Unsupported join type MARK",
        executor_supported=False,
    )

    assert status.to_dict() == {
        "query_id": 16,
        "export_ok": False,
        "error_type": "DuckDBSubstraitError",
        "error_message": "Unsupported join type MARK",
        "executor_supported": False,
    }
```

**Step 2: Run test to verify it fails**

Run: `timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_capabilities.py -q`

Expected: FAIL because `tpch_torch.capabilities` does not exist.

**Step 3: Write minimal implementation**

Implement frozen dataclasses:

```python
@dataclass(frozen=True)
class QueryExportStatus:
    query_id: int
    export_ok: bool
    error_type: str | None
    error_message: str | None
    executor_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "export_ok": self.export_ok,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "executor_supported": self.executor_supported,
        }
```

Keep functions under 100 lines.

**Step 4: Run test to verify it passes**

Run: `timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_capabilities.py -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add tpch_torch/capabilities.py tests/test_capabilities.py
git commit -m "feat: add native substrait capability status model"
```

### Task 2: Add explicit local DuckDB Substrait extension hook

**Files:**
- Modify: `tpch_torch/duckdb_bridge.py`
- Test: `tests/test_duckdb_bridge.py`

**Step 1: Write the failing tests**

Add tests that set `TQG_SUBSTRAIT_EXTENSION` to a non-existent path and assert `export_substrait_json` raises `DuckDBSubstraitError` whose message mentions the path. Add a second test using monkeypatch to clear the env var and assert the existing install/load code path is still invoked.

**Step 2: Run test to verify it fails**

Run: `timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_duckdb_bridge.py -q`

Expected: FAIL because the env var is ignored.

**Step 3: Implement minimal code**

In `_load_substrait_extension`, check:

```python
extension_path = os.environ.get("TQG_SUBSTRAIT_EXTENSION")
if extension_path:
    path = Path(extension_path)
    if not path.exists():
        raise DuckDBSubstraitError(f"TQG_SUBSTRAIT_EXTENSION does not exist: {path}")
    try:
        con.load_extension(str(path))
    except duckdb.Error as exc:
        raise DuckDBSubstraitError(f"failed to load TQG_SUBSTRAIT_EXTENSION {path}: {exc}") from exc
    return
```

Then keep the current install/load behavior unchanged.

**Step 4: Run test to verify it passes**

Run: `timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_duckdb_bridge.py -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add tpch_torch/duckdb_bridge.py tests/test_duckdb_bridge.py
git commit -m "feat: allow explicit duckdb substrait extension loading"
```

### Task 3: Implement native original-SQL export probe

**Files:**
- Modify: `tpch_torch/capabilities.py`
- Modify: `tpch_torch/runner.py`
- Test: `tests/test_capabilities.py`

**Step 1: Write the failing test**

Create a small DuckDB TPC-H fixture with `generate_tpch(sf=0.01)`. Test that `probe_tpch_substrait_exports(con, (2, 4, 16))` returns statuses for original SQL where `export_ok` is `False` under DuckDB 1.2.x, and that Q16's message contains `MARK` while Q2/Q4 contain `DELIM_JOIN`.

**Step 2: Run test to verify it fails**

Run: `timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_capabilities.py -q`

Expected: FAIL because probe function does not exist.

**Step 3: Implement minimal code**

Implement:

```python
def probe_tpch_substrait_exports(con, query_ids: Sequence[int]) -> list[QueryExportStatus]:
    statuses = []
    for query_id in query_ids:
        sql = get_tpch_query(con, query_id)
        try:
            export_substrait_json(con, sql)
            export_ok = True
            error_type = None
            error_message = None
        except DuckDBSubstraitError as exc:
            export_ok = False
            error_type = type(exc).__name__
            error_message = str(exc).splitlines()[0]
        statuses.append(QueryExportStatus(query_id, export_ok, error_type, error_message, _executor_supported(query_id)))
    return statuses
```

Expose supported executor IDs from `runner.py` without importing query modules unnecessarily.

**Step 4: Run test to verify it passes**

Run: `timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_capabilities.py -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add tpch_torch/capabilities.py tpch_torch/runner.py tests/test_capabilities.py
git commit -m "feat: probe native duckdb substrait exports"
```

### Task 4: Add CLI for native capability probing

**Files:**
- Create: `scripts/probe_substrait.py`
- Modify: `pyproject.toml`
- Test: `tests/test_runner_cli.py`

**Step 1: Write the failing CLI parser test**

Test that `scripts.probe_substrait.build_parser()` accepts:

```bash
--db data/tpch_sf1.duckdb --queries all --json
--db data/tpch_sf1.duckdb --queries 2,4,16
```

**Step 2: Run test to verify it fails**

Run: `timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_runner_cli.py -q`

Expected: FAIL because script does not exist.

**Step 3: Implement CLI**

Implement parser and `main()` that:

- Opens database via `connect_database`.
- Parses `all` as `range(1, 23)`.
- Calls `probe_tpch_substrait_exports`.
- Prints JSON when `--json`; otherwise prints table-like text.
- Does not treat export failures as process failures unless the database/extension itself cannot be used.

Add console script:

```toml
tpch-torch-probe-substrait = "scripts.probe_substrait:main"
```

**Step 4: Run test to verify it passes**

Run: `timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_runner_cli.py -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/probe_substrait.py pyproject.toml tests/test_runner_cli.py
git commit -m "feat: add native substrait probe cli"
```

### Task 5: Document B方案 and verification evidence

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-06-09-native-duckdb-substrait-design.md`

**Step 1: Write/update documentation**

Add a README section:

```text
Native DuckDB Substrait policy (B方案)
```

Include:

- No SQL rewrites.
- Current blocked query list.
- Probe command example.
- `TQG_SUBSTRAIT_EXTENSION` local extension hook.

**Step 2: Verify docs and tests**

Run:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q
timeout 60 /work/torch-query-gpu/.venv/bin/python -m compileall -q tpch_torch scripts
```

Expected: all tests pass and compileall is quiet.

**Step 3: Commit**

```bash
git add README.md docs/plans/2026-06-09-native-duckdb-substrait-design.md
git commit -m "docs: document native duckdb substrait expansion path"
```

### Task 6: Future native-export executor gate

**Files:**
- Modify later only when `tpch-torch-probe-substrait` reports `export_ok=true` for a previously blocked original query.

**Step 1: Write failing end-to-end test**

For the newly exported query, add a test that calls `validate_sql(con, get_tpch_query(con, query_id), device="cpu")` and expects DuckDB/PyTorch agreement.

**Step 2: Implement missing PyTorch operators**

Add only operators proven necessary by the exported Substrait plan, likely semi/anti/mark/single joins, count-distinct, scalar aggregate broadcast, or composite-key joins.

**Step 3: Verify and commit**

Run targeted tests, full tests, compileall, then commit and push.
