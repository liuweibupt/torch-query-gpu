import tomllib
from pathlib import Path


def test_public_entrypoints_exclude_removed_q1_json_commands():
    scripts = tomllib.loads(Path("pyproject.toml").read_text())["project"]["scripts"]

    assert "tpch-torch-export-q1-substrait" not in scripts
    assert "tpch-torch-run-q1" not in scripts
    assert "tpch-torch-validate-q1" not in scripts
    assert scripts["tpch-torch-run"] == "scripts.run_query:main"
    assert scripts["tpch-torch-validate"] == "scripts.validate_query:main"
    assert scripts["tpch-torch-benchmark"] == "scripts.benchmark_query:main"
