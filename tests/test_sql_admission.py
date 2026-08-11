import duckdb

from tpch_torch.operator_graph import OperatorKind, TQPOperatorGraph, TQPOperatorNode
from tpch_torch.sql_admission import analyze_strict_coverage, admit_sql


def test_admit_sql_lowers_simple_query_to_strict_admissible_graph():
    con = duckdb.connect()
    con.execute("create table t(id integer, x integer)")

    admission = admit_sql(con, "select id, x + 1 as y from t where x > 0")

    assert admission.graph.output_names == ("id", "y")
    assert admission.graph.output_types == ("INTEGER", "INTEGER")
    assert admission.strict_coverage.strict_admissible is True
    assert admission.strict_coverage.gaps == ()
    assert any(node.kind == OperatorKind.SCAN for node in admission.graph.nodes)


def test_admit_sql_parses_nested_query_and_reports_static_window_gap():
    con = duckdb.connect()
    con.execute("create table t(id integer, x integer, grp integer)")
    sql = """
    select id, running
    from (
        select id, sum(x) over (partition by grp order by id) as running
        from t
    ) s
    where running >= 10
    order by id
    """

    admission = admit_sql(con, sql)

    assert admission.graph.output_names == ("id", "running")
    assert admission.strict_coverage.strict_admissible is False
    assert [gap.node_name for gap in admission.strict_coverage.gaps] == ["WINDOW"]
    assert "aggregate WINDOW with ORDER BY frame" in admission.strict_coverage.gaps[0].reason


def test_static_coverage_reports_unknown_physical_node():
    unknown = TQPOperatorNode(node_id="n1", kind=OperatorKind.UNKNOWN, name="RECURSIVE_CTE")
    graph = TQPOperatorGraph(
        source_sql="with recursive r(x) as (select 1) select * from r",
        query_id=None,
        root_id="n1",
        nodes=(unknown,),
    )

    coverage = analyze_strict_coverage(graph)

    assert coverage.strict_admissible is False
    assert coverage.gaps[0].to_dict() == {
        "node_id": "n1",
        "node_name": "RECURSIVE_CTE",
        "node_kind": "unknown",
        "reason": "unsupported DuckDB physical node: RECURSIVE_CTE",
    }
