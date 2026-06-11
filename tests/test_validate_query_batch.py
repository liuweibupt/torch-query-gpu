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

    def validator(con, sql, device, frontend, use_compressed_masks):
        calls.append((sql, device, frontend, use_compressed_masks))
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
    assert [result.ok for result in results] == [True, True, True]
    assert calls == [
        ("select -- q1", "cuda", "sirius", False),
        ("select -- q3", "cuda", "sirius", False),
        ("select -- q6", "cuda", "sirius", False),
    ]


def test_validate_queries_keep_going_records_failures():
    def load_query(con, query_id):
        return f"select -- q{query_id}"

    def validator(con, sql, device, frontend, use_compressed_masks):
        query_id = int(sql.rsplit("q", 1)[1])
        if query_id == 3:
            raise RuntimeError("backend unsupported")
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
    assert "backend unsupported" in results[1].message


def test_validate_queries_without_keep_going_raises_first_failure():
    def load_query(con, query_id):
        return f"select -- q{query_id}"

    def validator(con, sql, device, frontend, use_compressed_masks):
        raise RuntimeError("backend unsupported")

    with pytest.raises(RuntimeError, match="backend unsupported"):
        validate_queries(
            FakeConnection(),
            (3,),
            device="cpu",
            tolerance=1e-2,
            keep_going=False,
            load_query=load_query,
            validator=validator,
        )


def test_main_runs_batch_validation_branch(monkeypatch, tmp_path, capsys):
    from scripts import validate_query

    class FakeConnection:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    con = FakeConnection()
    calls = []

    def connect_database(path):
        assert path == tmp_path / "tpch.duckdb"
        return con

    def validate_batch(connection, query_ids, *, device, tolerance, keep_going, frontend, use_compressed_masks):
        calls.append((connection, query_ids, device, tolerance, keep_going, frontend, use_compressed_masks))
        return [
            validate_query.BatchValidationRecord(
                query_id=1,
                ok=True,
                message="validated",
                row_count=4,
                max_abs_error=0.0,
            ),
            validate_query.BatchValidationRecord(
                query_id=3,
                ok=True,
                message="validated",
                row_count=10,
                max_abs_error=0.001,
            ),
        ]

    monkeypatch.setattr(validate_query, "connect_database", connect_database)
    monkeypatch.setattr(validate_query, "validate_queries", validate_batch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "tpch-torch-validate",
            "--db",
            str(tmp_path / "tpch.duckdb"),
            "--queries",
            "1,3",
            "--device",
            "cuda",
            "--keep-going",
        ],
    )

    validate_query.main()

    assert calls == [(con, (1, 3), "cuda", 1e-2, True, "sirius", False)]
    assert con.closed is True
    assert capsys.readouterr().out.splitlines() == [
        "validated query=1 rows=4 max_abs_error=0",
        "validated query=3 rows=10 max_abs_error=0.001",
    ]
