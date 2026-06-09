from pathlib import Path

from scripts.run_query import build_parser as run_parser
from scripts.validate_query import build_parser as validate_parser


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
