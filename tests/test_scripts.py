from scripts.gen_sf1 import build_parser as gen_parser


def test_gen_parser_accepts_required_arguments(tmp_path):
    db_path = tmp_path / "tpch.duckdb"

    assert gen_parser().parse_args(["--db", str(db_path)]).sf == 1.0
