from decimal import Decimal

import numpy as np
import torch

from tpch_torch.backend.type_mapping import (
    column_meta_from_duckdb_type,
    encode_decimal_array,
    encode_strings_dynamic,
)
from tpch_torch.record_batch import LogicalDType


def test_duckdb_type_mapping_covers_p1_types():
    cases = {
        "BIGINT": (LogicalDType.INT64, torch.int64),
        "INTEGER": (LogicalDType.INT64, torch.int64),
        "FLOAT": (LogicalDType.FP32, torch.float32),
        "DOUBLE": (LogicalDType.FP64, torch.float64),
        "BOOLEAN": (LogicalDType.BOOL, torch.bool),
        "DATE": (LogicalDType.DATE, torch.int32),
        "VARCHAR": (LogicalDType.STRING_DICT, torch.int64),
        "DECIMAL(12,2)": (LogicalDType.DECIMAL, torch.int64),
    }

    for duckdb_type, expected in cases.items():
        meta = column_meta_from_duckdb_type(duckdb_type, nullable=True)
        assert (meta.logical_dtype, meta.torch_dtype) == expected

    decimal_meta = column_meta_from_duckdb_type("DECIMAL(12,2)")
    assert decimal_meta.precision == 12
    assert decimal_meta.scale == 2


def test_encode_decimal_array_uses_scaled_int64():
    meta = column_meta_from_duckdb_type("DECIMAL(12,2)")

    tensor = encode_decimal_array(np.array([Decimal("12.34"), Decimal("0.05")], dtype=object), meta, "cpu")

    assert tensor.dtype == torch.int64
    assert tensor.tolist() == [1234, 5]


def test_dynamic_string_dictionary_expands_existing_vocabulary():
    first_tensor, first_meta = encode_strings_dynamic(np.array(["B", "A"], dtype=object), "cpu")
    second_tensor, second_meta = encode_strings_dynamic(
        np.array(["C", "A"], dtype=object),
        "cpu",
        existing=first_meta,
    )

    assert first_meta.dictionary == ("A", "B")
    assert first_tensor.tolist() == [1, 0]
    assert second_meta.dictionary == ("A", "B", "C")
    assert second_tensor.tolist() == [2, 0]


def test_dynamic_string_dictionary_preserves_existing_ids():
    existing = column_meta_from_duckdb_type("VARCHAR")
    existing = type(existing).string_dict(("B", "A"), nullable=existing.nullable)

    tensor, meta = encode_strings_dynamic(np.array(["C", "B"], dtype=object), "cpu", existing=existing)

    assert meta.dictionary == ("B", "A", "C")
    assert tensor.tolist() == [2, 0]


def test_column_type_from_duckdb_type_preserves_duckdb_repr_and_decimal_scale():
    from tpch_torch.backend.type_mapping import column_type_from_duckdb_type
    from tpch_torch.record_batch import LogicalDType

    amount = column_type_from_duckdb_type("amount", "DECIMAL(15,2)", nullable=True)
    text = column_type_from_duckdb_type("comment", "VARCHAR", nullable=False)

    assert amount.name == "amount"
    assert amount.duckdb_type_id == "DECIMAL"
    assert amount.duckdb_type_repr == "DECIMAL(15,2)"
    assert amount.logical_dtype == LogicalDType.DECIMAL
    assert amount.precision == 15
    assert amount.scale == 2
    assert amount.nullable is True
    assert text.duckdb_type_id == "VARCHAR"
    assert text.logical_dtype == LogicalDType.STRING
