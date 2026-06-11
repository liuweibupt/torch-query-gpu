import duckdb
import torch

from tpch_torch.backend.graph_nodes import (
    AggregateNode,
    AntiJoinNode,
    LookupJoinNode,
    ScalarSubqueryNode,
    ScanNode,
    SemiJoinNode,
)


def test_scan_lookup_and_grouped_aggregate_nodes_execute_tensor_flow():
    con = duckdb.connect()
    con.execute("create table fact(k integer, amount double)")
    con.execute("create table dim(k integer, label varchar)")
    con.execute("insert into fact values (1, 10.0), (2, 20.0), (1, 5.0), (3, 7.0)")
    con.execute("insert into dim values (1, 'a'), (2, 'b')")

    fact = ScanNode("fact", ("k", "amount")).execute(con, device="cpu")
    dim = ScanNode("dim", ("k", "label")).execute(con, device="cpu")
    label = LookupJoinNode(dim, "k", "label", fact.columns["k"]).execute(missing_value=-1)
    matched = label >= 0
    aggregate = AggregateNode.grouped_sum((label[matched],), fact.columns["amount"][matched])

    assert dim.dictionaries["label"] == ("a", "b")
    assert aggregate.keys.tolist() == [[0], [1]]
    assert aggregate.values.tolist() == [15.0, 20.0]


def test_semi_anti_and_scalar_subquery_nodes_are_explicit():
    probe = torch.tensor([1, 2, 3, 4])
    build = torch.tensor([2, 4])

    assert SemiJoinNode(probe, build).execute().tolist() == [False, True, False, True]
    assert AntiJoinNode(probe, build).execute().tolist() == [True, False, True, False]
    assert ScalarSubqueryNode.max(torch.tensor([3.0, 7.0, 2.0])).execute().item() == 7.0


def test_grouped_scalar_subquery_node_aggregates_and_maps_probe_keys():
    from tpch_torch.backend.graph_nodes import GroupedScalarSubqueryNode

    keys = torch.tensor([1, 1, 2, 3, 3, 3])
    values = torch.tensor([10.0, 14.0, 9.0, 1.0, 2.0, 3.0])
    probe = torch.tensor([3, 1, 4])

    means = GroupedScalarSubqueryNode.mean((keys,), values).lookup((probe,), missing_value=-1.0)
    sums = GroupedScalarSubqueryNode.sum((keys,), values).lookup((probe,), missing_value=0.0)
    mins = GroupedScalarSubqueryNode.min((keys,), values).lookup((probe,), missing_value=-1.0)

    assert means.tolist() == [2.0, 12.0, -1.0]
    assert sums.tolist() == [6.0, 24.0, 0.0]
    assert mins.tolist() == [1.0, 10.0, -1.0]


def test_grouped_scalar_subquery_node_uses_shared_packing_for_multi_key_lookup():
    from tpch_torch.backend.graph_nodes import GroupedScalarSubqueryNode

    left = torch.tensor([10, 10, 20, 20])
    right = torch.tensor([100, 101, 100, 101])
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    probe_left = torch.tensor([20, 10, 30])
    probe_right = torch.tensor([101, 101, 101])

    result = GroupedScalarSubqueryNode.sum((left, right), values).lookup(
        (probe_left, probe_right),
        missing_value=-1.0,
    )

    assert result.tolist() == [4.0, 2.0, -1.0]
