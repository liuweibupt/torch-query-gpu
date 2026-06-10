import duckdb
import pytest

from tpch_torch.planner import DuckDBPlannerError, export_duckdb_logical_plan


class RecordingConnection:
    def __init__(self):
        self.calls = []

    def execute(self, sql):
        self.calls.append(sql)
        if sql.startswith("EXPLAIN"):
            return self
        return self

    def fetchall(self):
        return [
            ("logical_plan", "LOGICAL_GET"),
            ("logical_opt", "LOGICAL_FILTER"),
            ("physical_plan", "SEQ_SCAN"),
        ]


def test_export_duckdb_logical_plan_uses_explain_all():
    con = RecordingConnection()

    plan = export_duckdb_logical_plan(con, "select 1")

    assert con.calls == ["PRAGMA explain_output='all'", "EXPLAIN select 1"]
    assert plan.logical_plan == "LOGICAL_GET"
    assert plan.logical_opt == "LOGICAL_FILTER"
    assert plan.physical_plan == "SEQ_SCAN"


def test_export_duckdb_logical_plan_reports_planner_errors():
    con = duckdb.connect()

    with pytest.raises(DuckDBPlannerError, match="DuckDB EXPLAIN failed"):
        export_duckdb_logical_plan(con, "select * from missing_table")
