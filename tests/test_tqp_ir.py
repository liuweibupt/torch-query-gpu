from tpch_torch.ir import TQPPlan


def test_tqp_plan_records_frontend_and_query_id():
    plan = TQPPlan(query_id=1, source_sql="select 1", frontend="sirius")

    assert plan.query_id == 1
    assert plan.frontend == "sirius"
    assert plan.plan_json is None
