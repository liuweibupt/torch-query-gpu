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
