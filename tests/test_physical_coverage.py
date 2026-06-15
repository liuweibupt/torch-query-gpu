import duckdb

from tpch_torch.duckdb_bridge import generate_tpch


def test_tpch_physical_coverage_reports_all_tpch_queries_supported():
    from tpch_torch.physical_coverage import probe_tpch_physical_coverage

    con = duckdb.connect()
    generate_tpch(con, scale_factor=0.01)

    records = probe_tpch_physical_coverage(con, range(1, 23), device="cpu")
    by_query = {record.query_id: record for record in records}

    assert set(by_query) == set(range(1, 23))
    assert all(record.supported for record in records)
    assert all(record.reason == "" for record in records)
