from tpch_torch.capabilities import QueryExportStatus


def test_query_export_status_serializes_failure():
    status = QueryExportStatus(
        query_id=16,
        export_ok=False,
        error_type="DuckDBSubstraitError",
        error_message="Unsupported join type MARK",
        executor_supported=False,
    )

    assert status.to_dict() == {
        "query_id": 16,
        "export_ok": False,
        "error_type": "DuckDBSubstraitError",
        "error_message": "Unsupported join type MARK",
        "executor_supported": False,
    }

import duckdb
import pytest

from tpch_torch.capabilities import probe_tpch_substrait_exports
from tpch_torch.duckdb_bridge import DuckDBSubstraitError, generate_tpch


@pytest.fixture(scope="module")
def native_probe_con():
    con = duckdb.connect()
    generate_tpch(con, scale_factor=0.01)
    try:
        yield con
    finally:
        con.close()


def test_probe_tpch_substrait_exports_reports_native_failures(native_probe_con):
    try:
        statuses = probe_tpch_substrait_exports(native_probe_con, (2, 4, 16))
    except DuckDBSubstraitError as exc:
        pytest.skip(f"DuckDB Substrait extension unavailable: {exc}")

    by_query = {status.query_id: status for status in statuses}

    assert by_query[2].export_ok is False
    assert "DELIM_JOIN" in by_query[2].error_message
    assert by_query[4].export_ok is False
    assert "DELIM_JOIN" in by_query[4].error_message
    assert by_query[16].export_ok is False
    assert "MARK" in by_query[16].error_message
