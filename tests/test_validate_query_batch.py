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
    assert [result.ok for result in results] == [True, True, True]
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
