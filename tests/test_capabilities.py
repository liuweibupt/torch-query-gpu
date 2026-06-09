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
