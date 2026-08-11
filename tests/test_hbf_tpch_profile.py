from types import SimpleNamespace

from scripts.profile_hbf_tpch import (
    ModuleConfig,
    _candidate_status,
    _column_width,
    _minimum_modules,
    _selective_bytes,
)


def test_profile_column_width_matches_tqp_encoding():
    assert _column_width("DATE") == 4
    assert _column_width("DECIMAL(15,2)") == 8
    assert _column_width("VARCHAR") == 8


def test_candidate_status_requires_amplification_and_crossing():
    page = ModuleConfig(64, 64.0, 2, 870.4)
    selective = ModuleConfig(64, 64.0, 1, 435.2)

    assert _candidate_status(20.0, page, selective) == "candidate_trace_followup"
    assert _candidate_status(4.0, page, selective) == "reject_low_amplification"
    assert _candidate_status(20.0, selective, selective) == "reject_no_module_crossing"


def test_minimum_modules_respects_grade_floor():
    args = SimpleNamespace(grade_floor_gb_s=400.0, payload_efficiency=0.85)
    config = _minimum_modules(40.0, 16, args)

    assert config.modules_per_stack == 4
    assert config.data_rate_gt_s == 64.0


def test_effective_amp_helper_handles_metadata():
    selective, metadata = _selective_bytes(
        selected_rows=50,
        table_rows=1000,
        projected_row_bytes=64,
        rowid_bytes=8,
    )

    assert metadata == 125
    assert selective == 3325
