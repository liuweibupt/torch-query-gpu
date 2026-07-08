"""DECIMAL expression helpers for physical tensor execution."""

from __future__ import annotations

from numbers import Real

import torch

from tpch_torch.backend.physical_types import PhysicalValue
from tpch_torch.backend.type_mapping import (
    DECIMAL_BASE,
    align_decimal_tensors,
    decimal_literal_scale,
    encode_decimal_literal,
    rescale_decimal_tensor,
)
from tpch_torch.errors import UnsupportedPlanError
from tpch_torch.record_batch import ColumnMeta, LogicalDType


def decimal_comparison_tensors(
    left: PhysicalValue,
    right: PhysicalValue,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return scale-aligned tensors for DECIMAL comparisons."""

    if not _contains_decimal(left, right):
        return None
    if _contains_non_decimal_numeric_tensor(left, right):
        return _float_operands(left, right)
    _validate_decimal_operands(left, right)
    left_tensor, right_tensor, _ = _aligned_decimal_operands(left, right)
    return left_tensor, right_tensor


def decimal_case_where(
    condition: torch.Tensor,
    then_value: PhysicalValue,
    else_value: PhysicalValue,
) -> PhysicalValue | None:
    """Evaluate CASE branch selection when a branch is DECIMAL."""

    if not _contains_decimal(then_value, else_value):
        return None
    if _contains_non_decimal_numeric_tensor(then_value, else_value):
        then_tensor, else_tensor = _float_operands(then_value, else_value)
        result = torch.where(condition, then_tensor, else_tensor)
        valid = _combine_validity(then_value, else_value, result)
        return PhysicalValue(tensor=result, valid=valid, meta=ColumnMeta.fp64())
    _validate_decimal_operands(then_value, else_value)
    then_tensor, else_tensor, meta = _aligned_decimal_operands(then_value, else_value)
    result = torch.where(condition, then_tensor, else_tensor)
    return PhysicalValue(tensor=result, valid=_combine_validity(then_value, else_value, result), meta=meta)


def decimal_arithmetic(
    left: PhysicalValue,
    operator: str,
    right: PhysicalValue,
) -> PhysicalValue | None:
    """Evaluate arithmetic when at least one operand is DECIMAL."""

    if not _contains_decimal(left, right):
        return None
    if _contains_non_decimal_numeric_tensor(left, right):
        return _decimal_numeric_tensor_arithmetic(left, operator, right)
    _validate_decimal_operands(left, right)
    if operator in {"+", "-"}:
        return _decimal_add_subtract(left, operator, right)
    if operator == "*":
        return _decimal_multiply(left, right)
    if operator == "/":
        return _decimal_divide(left, right)
    return None


def _decimal_numeric_tensor_arithmetic(
    left: PhysicalValue,
    operator: str,
    right: PhysicalValue,
) -> PhysicalValue:
    left_tensor, right_tensor = _float_operands(left, right)
    if operator == "+":
        result = left_tensor + right_tensor
    elif operator == "-":
        result = left_tensor - right_tensor
    elif operator == "*":
        result = left_tensor * right_tensor
    elif operator == "/":
        result = left_tensor / right_tensor
    else:
        raise UnsupportedPlanError(f"unsupported DECIMAL arithmetic operator: {operator}")
    return PhysicalValue(tensor=result, valid=_combine_validity(left, right, result), meta=ColumnMeta.fp64())


def _decimal_add_subtract(left: PhysicalValue, operator: str, right: PhysicalValue) -> PhysicalValue:
    left_tensor, right_tensor, meta = _aligned_decimal_operands(left, right)
    result = left_tensor + right_tensor if operator == "+" else left_tensor - right_tensor
    return PhysicalValue(tensor=result, valid=_combine_validity(left, right, result), meta=meta)


def _decimal_multiply(left: PhysicalValue, right: PhysicalValue) -> PhysicalValue:
    reference = _reference_tensor(left, right)
    left_tensor, left_scale, left_precision = _own_scale_decimal_operand(left, reference)
    right_tensor, right_scale, right_precision = _own_scale_decimal_operand(right, reference)
    result = left_tensor * right_tensor
    meta = ColumnMeta.decimal(
        precision=max(left_precision + right_precision, 1),
        scale=left_scale + right_scale,
    )
    return PhysicalValue(tensor=result, valid=_combine_validity(left, right, result), meta=meta)


def _decimal_divide(left: PhysicalValue, right: PhysicalValue) -> PhysicalValue:
    reference = _reference_tensor(left, right)
    result = _decimal_operand_to_float(left, reference) / _decimal_operand_to_float(right, reference)
    return PhysicalValue(tensor=result, valid=_combine_validity(left, right, result), meta=ColumnMeta.fp64())


def _aligned_decimal_operands(
    left: PhysicalValue,
    right: PhysicalValue,
) -> tuple[torch.Tensor, torch.Tensor, ColumnMeta]:
    if _is_decimal_value(left) and _is_decimal_value(right):
        return align_decimal_tensors(
            left.require_tensor(),
            left.meta,
            right.require_tensor(),
            right.meta,
        )
    reference = _reference_tensor(left, right)
    scale = max(_operand_scale(left), _operand_scale(right))
    precision = max(_operand_precision(left), _operand_precision(right))
    left_tensor = _operand_to_scale(left, reference, scale)
    right_tensor = _operand_to_scale(right, reference, scale)
    return left_tensor, right_tensor, ColumnMeta.decimal(precision=precision, scale=scale)


def _own_scale_decimal_operand(
    value: PhysicalValue,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    scale = _operand_scale(value)
    return _operand_to_scale(value, reference, scale), scale, _operand_precision(value)


def _operand_to_scale(value: PhysicalValue, reference: torch.Tensor, scale: int) -> torch.Tensor:
    if _is_decimal_value(value):
        from_scale = int(value.meta.scale or 0)
        return rescale_decimal_tensor(value.require_tensor(), from_scale, scale)
    literal = encode_decimal_literal(value.literal, scale)
    return torch.tensor(literal, dtype=torch.int64, device=reference.device)


def _decimal_operand_to_float(value: PhysicalValue, reference: torch.Tensor) -> torch.Tensor:
    if _is_decimal_value(value):
        scale = int(value.meta.scale or 0)
        return value.require_tensor().to(dtype=torch.float64) / float(DECIMAL_BASE**scale)
    if _is_numeric_tensor(value):
        return value.require_tensor().to(dtype=torch.float64)
    return torch.tensor(float(value.literal), dtype=torch.float64, device=reference.device)


def _float_operands(left: PhysicalValue, right: PhysicalValue) -> tuple[torch.Tensor, torch.Tensor]:
    reference = _reference_tensor(left, right)
    return _decimal_operand_to_float(left, reference), _decimal_operand_to_float(right, reference)


def _reference_tensor(left: PhysicalValue, right: PhysicalValue) -> torch.Tensor:
    if _is_decimal_value(left):
        return left.require_tensor()
    if _is_decimal_value(right):
        return right.require_tensor()
    raise UnsupportedPlanError("DECIMAL expression requires at least one DECIMAL tensor")


def _operand_scale(value: PhysicalValue) -> int:
    if _is_decimal_value(value):
        return int(value.meta.scale or 0)
    return decimal_literal_scale(value.literal)


def _operand_precision(value: PhysicalValue) -> int:
    if _is_decimal_value(value):
        return int(value.meta.precision or 1)
    return _literal_precision(value.literal)


def _literal_precision(value: object) -> int:
    text = str(value)
    digits = [character for character in text if character.isdigit()]
    return max(len(digits), decimal_literal_scale(value) + 1, 1)


def _validate_decimal_operands(left: PhysicalValue, right: PhysicalValue) -> None:
    if _is_decimal_or_numeric_operand(left) and _is_decimal_or_numeric_operand(right):
        return
    raise UnsupportedPlanError("DECIMAL expressions currently support DECIMAL tensors and numeric literals")


def _contains_decimal(left: PhysicalValue, right: PhysicalValue) -> bool:
    return _is_decimal_value(left) or _is_decimal_value(right)


def _contains_non_decimal_numeric_tensor(left: PhysicalValue, right: PhysicalValue) -> bool:
    return _is_non_decimal_numeric_tensor(left) or _is_non_decimal_numeric_tensor(right)


def _is_decimal_or_numeric_operand(value: PhysicalValue) -> bool:
    return _is_decimal_value(value) or _is_numeric_literal(value) or _is_non_decimal_numeric_tensor(value)


def _is_decimal_value(value: PhysicalValue) -> bool:
    return value.meta is not None and value.meta.logical_dtype == LogicalDType.DECIMAL


def _is_numeric_literal(value: PhysicalValue) -> bool:
    return value.tensor is None and isinstance(value.literal, Real) and not isinstance(value.literal, bool)


def _is_non_decimal_numeric_tensor(value: PhysicalValue) -> bool:
    return not _is_decimal_value(value) and _is_numeric_tensor(value)


def _is_numeric_tensor(value: PhysicalValue) -> bool:
    if value.tensor is None or value.dictionary is not None:
        return False
    if value.require_tensor().dtype is torch.bool:
        return False
    if value.meta is None:
        return True
    return value.meta.logical_dtype in {
        LogicalDType.INT64,
        LogicalDType.FP32,
        LogicalDType.FP64,
        LogicalDType.UNKNOWN,
    }


def _combine_validity(left: PhysicalValue, right: PhysicalValue, like: torch.Tensor) -> torch.Tensor | None:
    valid = None
    if left.valid is not None:
        valid = left.valid
    if right.valid is not None:
        valid = right.valid if valid is None else valid & right.valid
    if valid is None:
        return None
    return valid.to(device=like.device)
