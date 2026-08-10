from decimal import Decimal

import duckdb
import pytest

from tpch_torch.backend.universal import execute_universal_sql, iter_universal_record_batches
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.record_batch import TensorRecordBatch
from tpch_torch.record_batch_types import LogicalDType
from tpch_torch.runner import validate_sql


def _window_con():
    con = duckdb.connect()
    con.execute("create table t(id integer, x integer, grp integer)")
    con.execute("insert into t values (1, 10, 1), (2, 20, 1), (3, 30, 2)")
    return con


def test_strict_mode_still_rejects_unsupported_window_frame():
    con = _window_con()
    sql = "select id, sum(x) over (partition by grp order by id) as running from t order by id"

    with pytest.raises(UnsupportedPlanError, match="aggregate WINDOW with ORDER BY frame"):
        validate_sql(con, sql, device="cpu")


def test_universal_mode_executes_nested_sql_without_template_match():
    con = _window_con()
    sql = """
    select id, running
    from (
        select id, sum(x) over (partition by grp order by id) as running
        from t
    ) s
    where running >= 10
    order by id
    """

    result = validate_sql(con, sql, device="cpu", execution_mode="universal")

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [
        {"id": 1, "running": 10},
        {"id": 2, "running": 30},
        {"id": 3, "running": 30},
    ]


def test_universal_mode_executes_cte_and_correlated_subquery():
    con = _window_con()
    con.execute("create table limits(grp integer, min_running integer)")
    con.execute("insert into limits values (1, 25), (2, 25)")
    sql = """
    with running_totals as (
        select id, grp, sum(x) over (partition by grp order by id) as running
        from t
    )
    select id, running
    from running_totals r
    where exists (
        select 1
        from limits l
        where l.grp = r.grp and r.running >= l.min_running
    )
    order by id
    """

    result = validate_sql(con, sql, device="cpu", execution_mode="universal")

    assert result.max_abs_error == 0.0
    assert result.pytorch_rows == [{"id": 2, "running": 30}, {"id": 3, "running": 30}]


def test_invalid_execution_mode_is_explicit_error():
    con = duckdb.connect()

    with pytest.raises(ValueError, match="unknown execution mode"):
        validate_sql(
            con,
            "select 1 as x",
            device="cpu",
            execution_mode="duckdb",  # type: ignore[arg-type]
        )


def test_universal_mode_materializes_duckdb_result_as_tensor_record_batch():
    con = _window_con()
    sql = """
    select
      id,
      case when id = 1 then 'a' else 'b' end as label,
      cast(id * 10 as decimal(10,2)) as amount,
      date '1998-09-02' as d
    from t
    order by id
    """

    batches = tuple(iter_universal_record_batches(con, sql, device="cpu", chunk_size=2))
    rows = execute_universal_sql(con, sql, device="cpu", chunk_size=2)

    assert all(isinstance(batch, TensorRecordBatch) for batch in batches)
    assert [batch.row_count for batch in batches] == [2, 1]
    assert batches[0].types["id"].logical_dtype == LogicalDType.INT64
    assert batches[0].types["label"].logical_dtype == LogicalDType.STRING_DICT
    assert batches[0].types["amount"].logical_dtype == LogicalDType.DECIMAL
    assert batches[0].types["d"].logical_dtype == LogicalDType.DATE
    assert rows[0] == {"id": 1, "label": "a", "amount": Decimal("10.00"), "d": "1998-09-02"}
