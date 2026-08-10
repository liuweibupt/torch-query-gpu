import duckdb

from tpch_torch.runner import validate_sql


def _set_con():
    con = duckdb.connect()
    con.execute("create table t(id integer, x integer, grp integer)")
    con.execute("insert into t values (1, 10, 1), (2, 20, 1), (3, 30, 2), (4, 40, 2)")
    con.execute("create table u(id integer, x integer, grp integer)")
    con.execute("insert into u values (3, 30, 2), (5, 50, 3)")
    return con


def test_generic_union_all_runs_through_physical_interpreter():
    con = _set_con()
    result = validate_sql(
        con,
        "select id from t where id < 3 union all select id from u where id > 3",
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"id": 1}, {"id": 2}, {"id": 5}]


def test_generic_union_distinct_uses_union_then_group_by():
    con = _set_con()
    result = validate_sql(
        con,
        "select id from (select id from t union select id from u) s order by id",
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}]


def test_generic_window_row_number_and_rank_validate():
    con = _set_con()
    con.execute("insert into t values (5, 40, 2)")

    row_number = validate_sql(
        con,
        """
        select id, row_number() over (partition by grp order by x desc) as rn
        from t
        order by grp, rn
        """,
        device="cpu",
    )
    rank = validate_sql(
        con,
        """
        select id, rank() over (partition by grp order by x desc) as r
        from t
        order by grp, r, id
        """,
        device="cpu",
    )
    dense_rank = validate_sql(
        con,
        """
        select id, dense_rank() over (partition by grp order by x desc) as r
        from t
        order by grp, r, id
        """,
        device="cpu",
    )

    assert row_number.max_abs_error == 0.0
    assert rank.max_abs_error == 0.0
    assert dense_rank.max_abs_error == 0.0
    assert dense_rank.pytorch_rows[-1] == {"id": 3, "r": 2}


def test_generic_window_partition_aggregate_validate():
    con = _set_con()
    result = validate_sql(
        con,
        """
        select id, sum(x) over (partition by grp) as sx, count(*) over () as n
        from t
        order by id
        """,
        device="cpu",
    )

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [
        {"id": 1, "sx": 30, "n": 4},
        {"id": 2, "sx": 30, "n": 4},
        {"id": 3, "sx": 70, "n": 4},
        {"id": 4, "sx": 70, "n": 4},
    ]
