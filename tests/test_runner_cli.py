from pathlib import Path
import pytest

from scripts.explain_query import build_parser as explain_parser
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


def test_explain_parser_accepts_inline_sql_and_json(tmp_path):
    args = explain_parser().parse_args(
        ["--db", str(tmp_path / "tpch.duckdb"), "--sql", "select 1", "--json"]
    )

    assert args.sql == "select 1"
    assert args.json is True


def test_validate_parser_accepts_batch_queries_with_frontend(tmp_path):
    db_path = tmp_path / "tpch.duckdb"

    args = validate_parser().parse_args(
        ["--db", str(db_path), "--queries", "1,3,5,6", "--keep-going", "--frontend", "substrait"]
    )

    assert args.db == db_path
    assert args.queries == "1,3,5,6"
    assert args.keep_going is True
    assert args.frontend == "substrait"


def test_validate_parser_rejects_auto_frontend(tmp_path):
    parser = validate_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--db", str(tmp_path / "tpch.duckdb"), "--query", "1", "--frontend", "auto"])


def test_validate_parser_rejects_removed_plan_source(tmp_path):
    parser = validate_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--db", str(tmp_path / "tpch.duckdb"), "--query", "1", "--plan-source", "duckdb-logical"])


def test_run_parser_accepts_partition_options(tmp_path):
    args = run_parser().parse_args(
        [
            "--db",
            str(tmp_path / "tpch.duckdb"),
            "--query",
            "6",
            "--partition-table",
            "lineitem",
            "--partition-chunk-size",
            "100",
        ]
    )

    assert args.partition_table == "lineitem"
    assert args.partition_chunk_size == 100


def test_validate_parser_accepts_partition_options(tmp_path):
    args = validate_parser().parse_args(
        [
            "--db",
            str(tmp_path / "tpch.duckdb"),
            "--query",
            "6",
            "--partition-table",
            "lineitem",
            "--partition-chunk-size",
            "100",
        ]
    )

    assert args.partition_table == "lineitem"
    assert args.partition_chunk_size == 100


def test_run_parser_accepts_frontend(tmp_path):
    args = run_parser().parse_args(
        ["--db", str(tmp_path / "tpch.duckdb"), "--query", "6", "--frontend", "sirius"]
    )

    assert args.frontend == "sirius"


def test_run_parser_rejects_auto_frontend(tmp_path):
    parser = run_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--db", str(tmp_path / "tpch.duckdb"), "--query", "1", "--frontend", "auto"])


def test_run_parser_rejects_removed_plan_source(tmp_path):
    parser = run_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--db", str(tmp_path / "tpch.duckdb"), "--query", "1", "--plan-source", "duckdb-logical"])


def test_parse_validate_query_ids_list():
    assert parse_validate_query_ids("1,3,5,6") == (1, 3, 5, 6)


def test_parse_validate_query_ids_all():
    assert parse_validate_query_ids("all") == tuple(range(1, 23))


def test_validate_parser_rejects_query_and_queries_together(tmp_path):
    parser = validate_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--db",
                str(tmp_path / "tpch.duckdb"),
                "--query",
                "1",
                "--queries",
                "1,3",
            ]
        )

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


def test_run_query_main_prints_generic_query_label(monkeypatch, tmp_path, capsys):
    from scripts import run_query
    from tpch_torch.relational import QueryResult

    class FakeConnection:
        def close(self):
            pass

    def connect_database(path):
        return FakeConnection()

    def timed_run_sql(con, sql, *, device, frontend, use_compressed_masks, partition_config, execution_mode):
        assert sql == "select count(*) as n from t"
        assert frontend == "sirius"
        assert use_compressed_masks is False
        assert partition_config is None
        assert execution_mode == "strict"
        return QueryResult(query_id=None, rows=[{"n": 2}]), 1.25

    monkeypatch.setattr(run_query, "connect_database", connect_database)
    monkeypatch.setattr(run_query, "timed_run_sql", timed_run_sql)
    monkeypatch.setattr(
        "sys.argv",
        [
            "tpch-torch-run",
            "--db",
            str(tmp_path / "generic.duckdb"),
            "--sql",
            "select count(*) as n from t",
        ],
    )

    run_query.main()

    assert capsys.readouterr().out.splitlines() == ["{'n': 2}", "generic_pytorch_ms=1.250"]


def test_run_and_validate_parsers_accept_compressed_masks(tmp_path):
    db_path = tmp_path / "tpch.duckdb"

    run_args = run_parser().parse_args(["--db", str(db_path), "--query", "6", "--compressed-masks"])
    validate_args = validate_parser().parse_args(["--db", str(db_path), "--query", "6", "--compressed-masks"])

    assert run_args.compressed_masks is True
    assert validate_args.compressed_masks is True

from scripts.benchmark_query import build_parser as benchmark_parser


def test_benchmark_parser_accepts_cold_hot_timing_options(tmp_path):
    db_path = tmp_path / "tpch.duckdb"

    args = benchmark_parser().parse_args(
        [
            "--db",
            str(db_path),
            "--query",
            "6",
            "--device",
            "cuda",
            "--frontend",
            "sirius",
            "--cold-runs",
            "2",
            "--warmup-runs",
            "3",
            "--hot-runs",
            "5",
            "--compressed-masks",
            "--json",
        ]
    )

    assert args.db == db_path
    assert args.query == 6
    assert args.cold_runs == 2
    assert args.warmup_runs == 3
    assert args.hot_runs == 5
    assert args.compressed_masks is True
    assert args.json is True
