import duckdb

from tpch_torch.frontend.duckdb_ast import render_expression, select_expressions_by_alias, serialize_sql_ast


def test_duckdb_parser_ast_extracts_alias_expressions_without_regex():
    con = duckdb.connect()
    sql = "select a as x, b + 1 as y, sum(c * (1 - d)) as total from t"

    aliases = select_expressions_by_alias(con, sql)

    assert aliases == {
        "x": "a",
        "y": "(b + 1)",
        "total": "sum((c * (1 - d)))",
    }


def test_duckdb_parser_ast_renders_common_expression_nodes():
    con = duckdb.connect()
    sql = "select case when a > 1 then cast(b as bigint) else c end as z from t"
    select_item = serialize_sql_ast(con, sql)["statements"][0]["node"]["select_list"][0]

    assert render_expression(select_item) == "CASE WHEN (a > 1) THEN CAST(b AS BIGINT) ELSE c END"


def test_duckdb_parser_ast_extracts_nested_select_aliases():
    con = duckdb.connect()
    sql = "select x as outer_x from (select a + 1 as x from t) s"

    aliases = select_expressions_by_alias(con, sql)

    assert aliases["outer_x"] == "x"
    assert aliases["x"] == "(a + 1)"
