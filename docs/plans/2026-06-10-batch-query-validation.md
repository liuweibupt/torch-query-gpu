# Batch Query Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `tpch-torch-validate --queries` so several TPC-H queries run through original DuckDB SQL, DuckDB Substrait export, PyTorch execution, and DuckDB baseline validation in one command.

**Architecture:** Keep all execution inside `scripts.validate_query` orchestration and the existing `tpch_torch.runner.validate_sql` path. Batch mode loads original SQL for each query id with `get_tpch_query`, then invokes the same validator used by single-query mode, preserving the mandatory `export_substrait_json(con, sql)` chain.

**Tech Stack:** Python argparse CLI, DuckDB Python connection, existing `tpch_torch.runner`, pytest, git.

---

### Task 1: Add parser and query-id parsing tests

**Files:**
- Modify: `tests/test_runner_cli.py`
- Modify later: `scripts/validate_query.py`

**Step 1: Write the failing tests**

Add tests like:

```python
from scripts.validate_query import build_parser as validate_parser
from scripts.validate_query import parse_query_ids


def test_validate_parser_accepts_batch_queries(tmp_path):
    db_path = tmp_path / "tpch.duckdb"

    args = validate_parser().parse_args([
        "--db", str(db_path), "--queries", "1,3,5,6", "--keep-going"
    ])

    assert args.db == db_path
    assert args.queries == "1,3,5,6"
    assert args.keep_going is True


def test_parse_validate_query_ids_list():
    assert parse_query_ids("1,3,5,6") == (1, 3, 5, 6)
```

**Step 2: Run test to verify it fails**

Run:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q tests/test_runner_cli.py::test_validate_parser_accepts_batch_queries tests/test_runner_cli.py::test_parse_validate_query_ids_list
```

Expected: FAIL because `--queries` and `parse_query_ids` do not exist yet.

**Step 3: Write minimal implementation**

In `scripts/validate_query.py`:

- Add `source.add_argument("--queries", help="TPC-H query ids as comma-separated numbers")` to the existing mutually exclusive source group.
- Add `parser.add_argument("--keep-going", action="store_true", help="Continue batch validation after a query fails")`.
- Add:

```python
def parse_query_ids(raw: str) -> tuple[int, ...]:
    query_ids = tuple(int(item) for item in raw.split(",") if item)
    if not query_ids:
        raise ValueError("at least one query id is required")
    return query_ids
```

**Step 4: Run test to verify it passes**

Run the same pytest command. Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_runner_cli.py scripts/validate_query.py
git commit -m "test: cover batch tpch validate cli parsing"
```

---

### Task 2: Add batch validation orchestration tests

**Files:**
- Create: `tests/test_validate_query_batch.py`
- Modify later: `scripts/validate_query.py`

**Step 1: Write the failing tests**

Create tests with a fake connection and injected SQL loader / validator. The injected validator must record SQL strings and return real `SQLValidationResult` objects; it must not fake a production success path in runtime code.

```python
import pytest

from scripts.validate_query import validate_queries
from tpch_torch.relational import SQLValidationResult


class FakeConnection:
    pass


def _result(query_id: int) -> SQLValidationResult:
    return SQLValidationResult(
        query_id=query_id,
        row_count=query_id,
        max_abs_error=0.0,
        duckdb_rows=[],
        pytorch_rows=[],
    )


def test_validate_queries_loads_original_sql_and_validates_each_query():
    calls = []

    def load_query(con, query_id):
        assert isinstance(con, FakeConnection)
        return f"select -- q{query_id}"

    def validator(con, sql, device):
        calls.append((sql, device))
        return _result(int(sql.rsplit("q", 1)[1]))

    results = validate_queries(
        FakeConnection(),
        (1, 3, 6),
        device="cuda",
        tolerance=1e-2,
        keep_going=False,
        load_query=load_query,
        validator=validator,
    )

    assert [result.query_id for result in results] == [1, 3, 6]
    assert calls == [("select -- q1", "cuda"), ("select -- q3", "cuda"), ("select -- q6", "cuda")]


def test_validate_queries_keep_going_records_failures():
    def load_query(con, query_id):
        return f"select -- q{query_id}"

    def validator(con, sql, device):
        query_id = int(sql.rsplit("q", 1)[1])
        if query_id == 3:
            raise RuntimeError("substrait export failed")
        return _result(query_id)

    results = validate_queries(
        FakeConnection(),
        (1, 3, 6),
        device="cpu",
        tolerance=1e-2,
        keep_going=True,
        load_query=load_query,
        validator=validator,
    )

    assert [result.query_id for result in results] == [1, 3, 6]
    assert results[1].ok is False
    assert "substrait export failed" in results[1].message


def test_validate_queries_without_keep_going_raises_first_failure():
    def load_query(con, query_id):
        return f"select -- q{query_id}"

    def validator(con, sql, device):
        raise RuntimeError("substrait export failed")

    with pytest.raises(RuntimeError, match="substrait export failed"):
        validate_queries(
            FakeConnection(),
            (3,),
            device="cpu",
            tolerance=1e-2,
            keep_going=False,
            load_query=load_query,
            validator=validator,
        )
```

**Step 2: Run test to verify it fails**

Run:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q tests/test_validate_query_batch.py
```

Expected: FAIL because `validate_queries` and batch result model do not exist yet.

**Step 3: Write minimal implementation**

In `scripts/validate_query.py`:

- Add a frozen `BatchValidationRecord` dataclass with `query_id`, `ok`, `message`, `row_count`, `max_abs_error`.
- Add `_validate_one_query(...)` that calls `load_query(con, query_id)` then `validator(con, sql, device)`.
- Add `validate_queries(...)` accepting optional injected `load_query` and `validator`, defaulting to `get_tpch_query` and `validate_sql`.
- Enforce tolerance by raising `AssertionError` in non-keep-going mode or returning an `ok=False` record in keep-going mode.
- Do not catch and suppress exceptions unless `keep_going=True`; in keep-going mode record failure and continue.

**Step 4: Run test to verify it passes**

Run the same pytest command. Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_validate_query_batch.py scripts/validate_query.py
git commit -m "feat: add batch tpch validation orchestration"
```

---

### Task 3: Wire batch mode into CLI main

**Files:**
- Modify: `scripts/validate_query.py`
- Modify: `tests/test_runner_cli.py` if parser assertions need updates

**Step 1: Write the failing test**

Add a CLI-oriented test if needed for invalid combinations:

```python
import pytest


def test_validate_parser_rejects_query_and_queries_together(tmp_path):
    parser = validate_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--db", str(tmp_path / "tpch.duckdb"),
            "--query", "1",
            "--queries", "1,3",
        ])
```

**Step 2: Run test to verify it fails if not already covered**

Run:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q tests/test_runner_cli.py::test_validate_parser_rejects_query_and_queries_together
```

Expected: PASS after Task 1 if `--queries` is in the same mutually exclusive group; if absent, FAIL and fix parser.

**Step 3: Implement CLI main batch branch**

In `main()`:

- Open the DB connection once.
- If `args.queries is not None`, call `validate_queries(con, parse_query_ids(args.queries), ...)`.
- Print one line per record:

```text
validated query=1 rows=4 max_abs_error=0
failed query=3 substrait export failed
```

- If any batch record has `ok=False`, raise `AssertionError("batch validation failed: Q...")` after printing all records.
- Single-query / SQL / SQL-file path keeps the current behavior.

**Step 4: Run targeted tests**

Run:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q tests/test_runner_cli.py tests/test_validate_query_batch.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_runner_cli.py scripts/validate_query.py
git commit -m "feat: wire batch validation cli"
```

---

### Task 4: Verify full suite and real end-to-end flow

**Files:**
- No code changes expected.

**Step 1: Run unit test suite**

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q
```

Expected: all tests pass.

**Step 2: Run compile check**

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m compileall -q tpch_torch scripts
```

Expected: exit 0.

**Step 3: Run real batch validation through the CLI**

Use the existing SF1 database from the primary worktree:

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/tpch-torch-validate \
  --db /work/torch-query-gpu/data/tpch_sf1.duckdb \
  --queries 1,3,5,6 \
  --device cuda \
  --keep-going
```

If CUDA is unavailable, re-run with `--device cpu` and report the CUDA blocker explicitly. The command must not use JSON or manual plan export.

**Step 4: Commit any verification-doc updates if made**

If README or docs changed, commit them. Otherwise no commit needed.

---

### Task 5: Push branch

**Files:**
- Git metadata only.

**Step 1: Check status**

```bash
git status --short
```

Expected: clean.

**Step 2: Push**

```bash
git push
```

Expected: branch pushes to `origin feat/operator-expansion-papers`.

If push fails because of credentials or remote configuration, stop and report the exact blocker.
