import numpy as np
from decimal import Decimal

import tpch_torch.storage as storage_module
from tpch_torch.storage import table_from_columnar


def test_table_from_columnar_uses_array_conversion_for_numpy_numeric(monkeypatch):
    values = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    calls = []
    original_as_tensor = storage_module.torch.as_tensor

    def recording_as_tensor(data, *args, **kwargs):
        calls.append(data)
        return original_as_tensor(data, *args, **kwargs)

    monkeypatch.setattr(storage_module.torch, "as_tensor", recording_as_tensor)

    table = table_from_columnar({"l_quantity": values}, device="cpu")

    assert any(isinstance(data, np.ndarray) for data in calls)
    assert table.columns["l_quantity"].tolist() == [10.0, 20.0, 30.0]


def test_table_from_columnar_uses_numpy_unique_for_numpy_string_columns(monkeypatch):
    values = np.array(["N", "A", "N", "R"], dtype=object)
    calls = []
    if not hasattr(storage_module, "np"):
        monkeypatch.setattr(storage_module, "np", np, raising=False)
    original_unique = storage_module.np.unique

    def recording_unique(data, *args, **kwargs):
        calls.append((data, kwargs.get("return_inverse")))
        return original_unique(data, *args, **kwargs)

    monkeypatch.setattr(storage_module.np, "unique", recording_unique)

    table = table_from_columnar({"l_returnflag": values}, device="cpu")

    assert calls and calls[0][1] is True
    assert table.dictionaries["l_returnflag"] == ("A", "N", "R")
    assert table.columns["l_returnflag"].tolist() == [1, 0, 1, 2]


def test_table_from_columnar_accepts_numpy_object_date_values():
    values = np.array(["1998-09-02", "1998-01-01"], dtype=object)

    table = table_from_columnar({"l_shipdate": values}, device="cpu")

    assert table.columns["l_shipdate"].tolist() == [19980902, 19980101]


def test_table_from_columnar_accepts_numpy_object_numeric_values():
    values = np.array([Decimal("10.50"), Decimal("20.25")], dtype=object)

    table = table_from_columnar({"l_quantity": values}, device="cpu")

    assert table.columns["l_quantity"].tolist() == [10.5, 20.25]
