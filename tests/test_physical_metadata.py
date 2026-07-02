import duckdb
import torch


def test_physical_join_uses_value_metadata_without_dynamic_sortedness_checks(monkeypatch):
    import tpch_torch.backend.physical_join as physical_join
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue

    def fail_sorted_check(*_args, **_kwargs):
        raise AssertionError("metadata-backed join should not rediscover sortedness")

    monkeypatch.setattr(physical_join, "_is_sorted_non_decreasing", fail_sorted_check)
    monkeypatch.setattr(physical_join, "_is_strictly_increasing", fail_sorted_check)

    left = PhysicalTable(
        "orders",
        {"o_custkey": PhysicalValue(torch.tensor([3, 1, 2, 5, 2], dtype=torch.int64))},
        ("o_custkey",),
        5,
    )
    right_key = PhysicalValue(
        torch.tensor([1, 2, 3, 4], dtype=torch.int64),
        sorted_non_decreasing=True,
        unique=True,
    )
    right = PhysicalTable("customer", {"c_custkey": right_key}, ("c_custkey",), 4)

    left_rows, right_rows = physical_join.join_indices_for_conditions(
        left,
        right,
        (("o_custkey", "c_custkey"),),
    )

    assert torch.equal(left_rows, torch.tensor([0, 1, 2, 4], dtype=torch.int64))
    assert torch.equal(right_rows, torch.tensor([2, 0, 1, 1], dtype=torch.int64))


def test_physical_single_key_grouped_aggregate_marks_group_key_metadata():
    from tpch_torch.backend.physical_aggregate import AggregateSpec, execute_grouped_aggregate
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue

    child = PhysicalTable(
        "sorted_input",
        {
            "k": PhysicalValue(
                torch.tensor([1, 1, 2, 2, 3], dtype=torch.int64),
                sorted_non_decreasing=True,
            ),
            "v": PhysicalValue(torch.tensor([5.0, 7.0, 11.0, 13.0, 17.0])),
        },
        ("k", "v"),
        5,
    )

    result = execute_grouped_aggregate(
        child,
        ("k",),
        (AggregateSpec("sum", "v", ("sum_v",)),),
    )

    key = result.value_named("k")
    assert key.require_tensor().tolist() == [1, 2, 3]
    assert key.sorted_non_decreasing is True
    assert key.unique is True


def test_physical_scan_marks_validated_tpch_primary_key_metadata():
    from tpch_torch.backend.physical_scan import fetch_physical_table

    con = duckdb.connect()
    con.execute("create table region(r_regionkey integer, r_name varchar)")
    con.execute("insert into region values (0, 'AFRICA'), (1, 'AMERICA'), (2, 'ASIA')")

    table = fetch_physical_table(con, "region", ("r_regionkey",), ("r_regionkey",), "cpu")

    key = table.value_named("r_regionkey")
    rowid = table.value_named("rowid")
    assert key.sorted_non_decreasing is True
    assert key.unique is True
    assert rowid.sorted_non_decreasing is True
    assert rowid.unique is True


def test_physical_sort_marks_single_ascending_key_metadata():
    from tpch_torch.backend.physical import PhysicalPlanExecutor
    from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue

    executor = PhysicalPlanExecutor.__new__(PhysicalPlanExecutor)
    executor._select_aliases = {}
    table = PhysicalTable(
        "t",
        {
            "k": PhysicalValue(torch.tensor([3, 1, 2], dtype=torch.int64), unique=True),
            "v": PhysicalValue(torch.tensor([30.0, 10.0, 20.0])),
        },
        ("k", "v"),
        3,
    )

    result = PhysicalPlanExecutor._sort_table(executor, table, ("k ASC",))

    key = result.value_named("k")
    assert key.require_tensor().tolist() == [1, 2, 3]
    assert key.sorted_non_decreasing is True
    assert key.unique is True
