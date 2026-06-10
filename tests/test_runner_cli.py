from pathlib import Path

from scripts.run_query import build_parser as run_parser
from scripts.validate_query import build_parser as validate_parser
from scripts.validate_query import parse_query_ids as parse_validate_query_ids


def test_run_parser_accepts_query_number(tmp_path):
    db_path = tmp_path / "tpch.duckdb"

    args = run_parser().parse_args(["--db", str(db_path), "--query", "6", "--device", "cuda"])

    assert args.db == db_path
    assert args.query == 6
    assert args.device == "cuda"


def test_run_parser_accepts_inline_sql(tmp_path):
    args = run_parser().parse_args(["--db", str(tmp_path / "tpch.duckdb"), "--sql", "select 1"])

    assert args.sql == "select 1"


def test_validate_parser_accepts_sql_file(tmp_path):
    sql_path = tmp_path / "q.sql"

    args = validate_parser().parse_args(["--db", str(tmp_path / "tpch.duckdb"), "--sql-file", str(sql_path)])

    assert args.sql_file == sql_path


def test_validate_parser_accepts_batch_queries(tmp_path):
    db_path = tmp_path / "tpch.duckdb"

    args = validate_parser().parse_args(
        ["--db", str(db_path), "--queries", "1,3,5,6", "--keep-going"]
    )

    assert args.db == db_path
    assert args.queries == "1,3,5,6"
    assert args.keep_going is True


def test_parse_validate_query_ids_list():
    assert parse_validate_query_ids("1,3,5,6") == (1, 3, 5, 6)

from scripts.probe_substrait import build_parser as probe_parser


def test_probe_parser_accepts_all_queries_json(tmp_path):
    db_path = tmp_path / "tpch.duckdb"

    args = probe_parser().parse_args(["--db", str(db_path), "--queries", "all", "--json"])

    assert args.db == db_path
    assert args.queries == "all"
    assert args.json is True


def test_probe_parser_accepts_query_list(tmp_path):
    db_path = tmp_path / "tpch.duckdb"

    args = probe_parser().parse_args(["--db", str(db_path), "--queries", "2,4,16"])

    assert args.db == db_path
    assert args.queries == "2,4,16"
    assert args.json is False

from scripts.probe_substrait import parse_query_ids


def test_parse_probe_query_ids_all_and_list():
    assert parse_query_ids("all") == tuple(range(1, 23))
    assert parse_query_ids("2,4,16") == (2, 4, 16)
