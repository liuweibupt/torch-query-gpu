from scripts.export_q1_substrait import build_parser as export_parser
from scripts.gen_sf1 import build_parser as gen_parser
from scripts.run_q1 import build_parser as run_parser
from scripts.validate_q1 import build_parser as validate_parser


def test_script_parsers_accept_required_arguments(tmp_path):
    db_path = tmp_path / "tpch.duckdb"
    json_path = tmp_path / "q1.json"

    assert gen_parser().parse_args(["--db", str(db_path)]).sf == 1.0
    assert export_parser().parse_args(["--db", str(db_path), "--out", str(json_path)]).out == json_path
    assert run_parser().parse_args(["--db", str(db_path), "--device", "cpu"]).device == "cpu"
    assert validate_parser().parse_args(["--db", str(db_path), "--tolerance", "0.1"]).tolerance == 0.1
