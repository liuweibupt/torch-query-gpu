import numpy as np
import torch

from tpch_torch.relational import table_from_columnar_typed


class NonIterableArray(np.ndarray):
    def __new__(cls, values):
        return np.asarray(values).view(cls)

    def __iter__(self):
        raise AssertionError("typed columnar encoding must not iterate numpy arrays")


def test_table_from_columnar_typed_encodes_numpy_without_python_iteration():
    columnar = {
        "l_returnflag": NonIterableArray(["R", "A", "R"]),
        "l_shipdate": NonIterableArray(np.array([19940101, 19940102, 19940103], dtype=np.int32)),
        "l_orderkey": NonIterableArray(np.array([3, 4, 5], dtype=np.int64)),
        "l_extendedprice": NonIterableArray(np.array([10.5, 20.25, 30.0], dtype=np.float64)),
    }

    table = table_from_columnar_typed(columnar, device="cpu")

    assert table.columns["l_returnflag"].tolist() == [1, 0, 1]
    assert table.dictionaries["l_returnflag"] == ("A", "R")
    assert table.columns["l_shipdate"].dtype == torch.int32
    assert table.columns["l_orderkey"].dtype == torch.int64
    assert table.columns["l_extendedprice"].dtype == torch.float64
