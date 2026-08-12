import numpy as np
from types import SimpleNamespace

from scripts.profile_hbf_q6_time_gate import (
    _candidate_groups,
    _column_pages_for_ids,
    _full_required_column_bytes,
    _module_sweep,
    _selective_return_bytes,
    BaselineBytes,
)


def test_q6_selective_return_uses_smaller_metadata():
    assert _selective_return_bytes(rows=1024, selected_rows=8, rowid_bytes=8) == (8 * 16) + 64
    assert _selective_return_bytes(rows=1024, selected_rows=100, rowid_bytes=8) == (100 * 16) + 128


def test_column_pages_for_ids_counts_unique_physical_pages():
    ids = np.array([0, 1, 511, 512, 1023, 1024], dtype=np.int64)
    assert _column_pages_for_ids(rows=2048, width=8, ids=ids, page_bytes=4096) == 3 * 4096


def test_candidate_groups_uses_scan_local_q6_predicates():
    columns = {
        "l_shipdate": np.array([19940101, 19940102, 19960101, 19960102], dtype=np.int32),
        "l_discount": np.array([0.06, 0.08, 0.06, 0.06], dtype=np.float64),
        "l_quantity": np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64),
    }
    assert _candidate_groups(columns, group_rows=2) == [(0, 2)]


def test_module_sweep_binds_gb_to_target_time():
    args = SimpleNamespace(
        stack_count=3,
        grade_floor_gb_s=400.0,
        payload_efficiency=0.85,
    )
    baseline = BaselineBytes("page", 0, 168_000_000, 16.8, 65.0, "")
    rows = _module_sweep([baseline], (10.0, 50.0), args)

    assert rows[0].required_gb_s_per_stack == 560.0
    assert rows[0].x64_modules_per_stack == 2
    assert rows[1].required_gb_s_per_stack == 112.0
    assert rows[1].x64_modules_per_stack == 1


def test_full_required_column_bytes_uses_physical_page_rounding():
    values = _full_required_column_bytes(rows=513, page_bytes=4096)

    assert values["l_discount"] == 2 * 4096
    assert values["l_shipdate"] == 1 * 4096
