import duckdb

from tpch_torch.duckdb_bridge import generate_tpch


def test_tpch_physical_coverage_reports_supported_and_blocked_queries():
    from tpch_torch.physical_coverage import probe_tpch_physical_coverage

    con = duckdb.connect()
    generate_tpch(con, scale_factor=0.01)

    records = probe_tpch_physical_coverage(con, range(1, 23), device="cpu")
    by_query = {record.query_id: record for record in records}

    assert {query_id for query_id, record in by_query.items() if record.supported} >= {1, 6, 12, 14, 19}
    assert any(not record.supported and record.reason for record in records)
    assert by_query[2].supported is False
