from tpch_torch.errors import UnsupportedPlanError


def test_unsupported_plan_error_is_shared_runtime_error():
    error = UnsupportedPlanError("not supported")

    assert isinstance(error, ValueError)
    assert str(error) == "not supported"
