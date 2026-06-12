import torch

from tpch_torch.backend.physical_types import PhysicalTable, PhysicalValue


def test_physical_table_filter_transforms_shared_alias_value_once(monkeypatch):
    value = PhysicalValue(torch.tensor([1, 2, 3], dtype=torch.int64))
    table = PhysicalTable("t", {"a": value, "t.a": value}, ("a",), 3)
    calls = []
    original_filter = PhysicalValue.filter

    def recording_filter(self, mask):
        calls.append(id(self))
        return original_filter(self, mask)

    monkeypatch.setattr(PhysicalValue, "filter", recording_filter)

    filtered = table.filter(torch.tensor([True, False, True]))

    assert len(calls) == 1
    assert filtered.columns["a"] is filtered.columns["t.a"]


def test_physical_table_gather_transforms_shared_alias_value_once(monkeypatch):
    value = PhysicalValue(torch.tensor([10, 20, 30], dtype=torch.int64))
    table = PhysicalTable("t", {"a": value, "t.a": value}, ("a",), 3)
    calls = []
    original_gather = PhysicalValue.gather

    def recording_gather(self, indices):
        calls.append(id(self))
        return original_gather(self, indices)

    monkeypatch.setattr(PhysicalValue, "gather", recording_gather)

    gathered = table.gather(torch.tensor([2, 0], dtype=torch.int64))

    assert len(calls) == 1
    assert gathered.columns["a"] is gathered.columns["t.a"]
