import pytest

from tpch_torch.generic_sql import parse_generic_sql
from tpch_torch.errors import UnsupportedPlanError


def test_parse_count_star_query():
    plan = parse_generic_sql("select count(*) as n from t")

    assert plan.table == "t"
    assert plan.projections[0].kind == "count_star"
    assert plan.projections[0].alias == "n"
    assert plan.required_columns == ()


def test_parse_grouped_sum_with_filter_order_and_limit():
    plan = parse_generic_sql(
        "select a, sum(b) as total from t where b >= 2 group by a order by a limit 10"
    )

    assert plan.table == "t"
    assert [projection.alias for projection in plan.projections] == ["a", "total"]
    assert plan.filters[0].column == "b"
    assert plan.filters[0].operator == ">="
    assert plan.filters[0].value == 2
    assert plan.group_by == ("a",)
    assert plan.order_by == ("a",)
    assert plan.limit == 10
    assert plan.required_columns == ("a", "b")


def test_parse_rejects_join_query():
    with pytest.raises(UnsupportedPlanError, match="joins are not supported"):
        parse_generic_sql("select * from t join u on t.id = u.id")


def test_parse_extended_aggregates_and_count_column():
    plan = parse_generic_sql("select min(b) as lo, max(b) as hi, avg(b) as mean_b, count(c) as c_count from t")

    assert [projection.kind for projection in plan.projections] == ["min", "max", "avg", "count"]
    assert [projection.column for projection in plan.projections] == ["b", "b", "b", "c"]
    assert plan.required_columns == ("b", "c")


def test_parse_filter_in_like_or_and_not():
    plan = parse_generic_sql("select c from t where not a = 1 or c in ('x', 'z') and c like 'z%'")

    assert plan.filters.kind == "or"
    assert plan.required_columns == ("c", "a")


def test_parse_order_by_direction():
    plan = parse_generic_sql("select a from t order by a desc, b asc limit 2")

    assert [(item.column, item.descending) for item in plan.order_by] == [("a", True), ("b", False)]
    assert plan.required_columns == ("a", "b")
