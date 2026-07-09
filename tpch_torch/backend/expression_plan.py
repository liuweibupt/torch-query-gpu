"""Typed expression DAG lowering for TensorRecordBatch filter/projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from tpch_torch.backend.type_mapping import rescale_decimal_tensor
from tpch_torch.record_batch import (
    BatchMeta,
    ColumnStorage,
    ColumnType,
    LogicalDType,
    TensorRecordBatch,
)

_BINARY_OPS = {"add", "sub", "mul", "div"}
_TMP_PREFIX = "__expr"


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple["Expr", ...] = ()
    value: object | None = None


@dataclass(frozen=True)
class TensorPrimitive:
    op: str
    inputs: tuple[str, ...]
    output: str
    output_type: ColumnType
    attrs: Mapping[str, object]


@dataclass(frozen=True)
class TensorPrimitivePlan:
    primitives: tuple[TensorPrimitive, ...]
    outputs: Mapping[str, str]
    output_types: Mapping[str, ColumnType]
    constants: Mapping[str, tuple[object, ColumnType]]

    def execute(self, batch: TensorRecordBatch) -> TensorRecordBatch:
        env = dict(batch.columns)
        env_types = dict(batch.types)
        _materialize_constants(batch, self.constants, env, env_types)
        for primitive in self.primitives:
            env[primitive.output] = _execute_primitive(primitive, env, env_types)
            env_types[primitive.output] = primitive.output_type
        return _build_output_batch(batch, self.outputs, self.output_types, env)


def col(name: str) -> Expr:
    return Expr("col", value=name)


def lit(value: object) -> Expr:
    return Expr("lit", value=value)


def add(left: Expr, right: Expr) -> Expr:
    return Expr("add", (left, right))


def sub(left: Expr, right: Expr) -> Expr:
    return Expr("sub", (left, right))


def mul(left: Expr, right: Expr) -> Expr:
    return Expr("mul", (left, right))


def div(left: Expr, right: Expr) -> Expr:
    return Expr("div", (left, right))


def compile_projection(
    batch: TensorRecordBatch,
    expressions: Mapping[str, Expr],
) -> TensorPrimitivePlan:
    compiler = _ProjectionCompiler(batch)
    outputs: dict[str, str] = {}
    output_types: dict[str, ColumnType] = {}
    for output_name, expression in expressions.items():
        ref, output_type = compiler.lower(_fold_constants(expression))
        outputs[output_name] = ref
        output_types[output_name] = _rename_type(output_type, output_name)
    return TensorPrimitivePlan(
        primitives=tuple(compiler.primitives),
        outputs=outputs,
        output_types=output_types,
        constants=compiler.constants,
    )


def project_expressions(
    batch: TensorRecordBatch,
    expressions: Mapping[str, Expr],
) -> TensorRecordBatch:
    return compile_projection(batch, expressions).execute(batch)


class _ProjectionCompiler:
    def __init__(self, batch: TensorRecordBatch) -> None:
        self.batch = batch
        self.primitives: list[TensorPrimitive] = []
        self.constants: dict[str, tuple[object, ColumnType]] = {}
        self.memo: dict[Expr, tuple[str, ColumnType]] = {}
        self.next_id = 0

    def lower(self, expression: Expr) -> tuple[str, ColumnType]:
        if expression in self.memo:
            return self.memo[expression]
        result = self._lower_uncached(expression)
        self.memo[expression] = result
        return result

    def _lower_uncached(self, expression: Expr) -> tuple[str, ColumnType]:
        if expression.op == "col":
            name = str(expression.value)
            return name, self.batch.types[name]
        if expression.op == "lit":
            return self._lower_literal(expression.value)
        if expression.op in _BINARY_OPS:
            return self._lower_binary(expression)
        raise NotImplementedError(f"unsupported expression op: {expression.op}")

    def _lower_literal(self, value: object) -> tuple[str, ColumnType]:
        name = self._tmp_name("lit")
        column_type = _literal_type(name, value)
        self.constants[name] = (value, column_type)
        return name, column_type

    def _lower_binary(self, expression: Expr) -> tuple[str, ColumnType]:
        left_ref, left_type = self.lower(expression.args[0])
        right_ref, right_type = self.lower(expression.args[1])
        name = self._tmp_name(expression.op)
        output_type = _binary_output_type(name, expression.op, left_type, right_type)
        self.primitives.append(
            TensorPrimitive(
                expression.op,
                (left_ref, right_ref),
                name,
                output_type,
                attrs={},
            )
        )
        return name, output_type

    def _tmp_name(self, label: str) -> str:
        name = f"{_TMP_PREFIX}_{label}_{self.next_id}"
        self.next_id += 1
        return name


def _fold_constants(expression: Expr) -> Expr:
    if expression.op not in _BINARY_OPS:
        return expression
    left = _fold_constants(expression.args[0])
    right = _fold_constants(expression.args[1])
    if left.op == "lit" and right.op == "lit":
        return lit(_compute_literal(expression.op, left.value, right.value))
    return Expr(expression.op, (left, right))


def _compute_literal(op: str, left: object, right: object) -> object:
    if op == "add":
        return left + right
    if op == "sub":
        return left - right
    if op == "mul":
        return left * right
    if op == "div":
        return left / right
    raise NotImplementedError(op)


def _literal_type(name: str, value: object) -> ColumnType:
    if isinstance(value, bool):
        return ColumnType.boolean(name)
    if isinstance(value, int):
        return ColumnType.int64(name)
    if isinstance(value, float):
        return ColumnType.fp64(name)
    raise TypeError(f"unsupported literal type: {type(value).__name__}")


def _binary_output_type(
    name: str,
    op: str,
    left: ColumnType,
    right: ColumnType,
) -> ColumnType:
    if _is_decimal(left) or _is_decimal(right):
        return _decimal_output_type(name, op, left, right)
    if left.logical_dtype == LogicalDType.FP64 or right.logical_dtype == LogicalDType.FP64:
        return ColumnType.fp64(name)
    if left.logical_dtype == LogicalDType.FP32 or right.logical_dtype == LogicalDType.FP32:
        return ColumnType.fp32(name)
    return ColumnType.int64(name)


def _decimal_output_type(name: str, op: str, left: ColumnType, right: ColumnType) -> ColumnType:
    left_scale = int(left.scale or 0)
    right_scale = int(right.scale or 0)
    precision = max(int(left.precision or 18), int(right.precision or 18))
    if op in {"add", "sub"}:
        return ColumnType.decimal(name, precision=precision, scale=max(left_scale, right_scale))
    if op == "mul":
        return ColumnType.decimal(name, precision=precision, scale=left_scale + right_scale)
    return ColumnType.fp64(name)


def _materialize_constants(
    batch: TensorRecordBatch,
    constants: Mapping[str, tuple[object, ColumnType]],
    env: dict[str, torch.Tensor],
    env_types: dict[str, ColumnType],
) -> None:
    for name, (value, column_type) in constants.items():
        tensor = _literal_tensor(value, batch.row_count, batch.batch_meta.device, column_type)
        env[name] = tensor
        env_types[name] = column_type


def _literal_tensor(
    value: object,
    row_count: int,
    device: torch.device,
    column_type: ColumnType,
) -> torch.Tensor:
    dtype = _torch_dtype(column_type)
    return torch.full((row_count,), value, dtype=dtype, device=device)


def _execute_primitive(
    primitive: TensorPrimitive,
    env: Mapping[str, torch.Tensor],
    env_types: Mapping[str, ColumnType],
) -> torch.Tensor:
    left, right = primitive.inputs
    if _is_decimal(primitive.output_type) and primitive.op in {"add", "sub"}:
        left_tensor, right_tensor = _align_decimal_inputs(
            env[left],
            env_types[left],
            env[right],
            env_types[right],
            int(primitive.output_type.scale or 0),
        )
    else:
        left_tensor, right_tensor = env[left], env[right]
    return _apply_binary_op(primitive.op, left_tensor, right_tensor)


def _align_decimal_inputs(
    left: torch.Tensor,
    left_type: ColumnType,
    right: torch.Tensor,
    right_type: ColumnType,
    target_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    left_aligned = rescale_decimal_tensor(left, int(left_type.scale or 0), target_scale)
    right_aligned = rescale_decimal_tensor(right, int(right_type.scale or 0), target_scale)
    return left_aligned, right_aligned


def _apply_binary_op(op: str, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if op == "add":
        return left + right
    if op == "sub":
        return left - right
    if op == "mul":
        return left * right
    if op == "div":
        return left / right
    raise NotImplementedError(op)


def _build_output_batch(
    batch: TensorRecordBatch,
    outputs: Mapping[str, str],
    output_types: Mapping[str, ColumnType],
    env: Mapping[str, torch.Tensor],
) -> TensorRecordBatch:
    storages = {
        name: _storage_for_output(env[ref], output_types[name])
        for name, ref in outputs.items()
    }
    meta = BatchMeta(
        row_count=batch.row_count,
        chunk_size=batch.batch_meta.chunk_size,
        chunk_index=batch.batch_meta.chunk_index,
        source_offset=batch.batch_meta.source_offset,
        device=batch.batch_meta.device,
        schema_version=batch.batch_meta.schema_version,
    )
    return TensorRecordBatch.from_storages(columns=storages, types=output_types, batch_meta=meta)


def _storage_for_output(tensor: torch.Tensor, column_type: ColumnType) -> ColumnStorage:
    if column_type.logical_dtype == LogicalDType.DECIMAL:
        return ColumnStorage.decimal64(tensor)
    return ColumnStorage.fixed(tensor)


def _torch_dtype(column_type: ColumnType) -> torch.dtype:
    if column_type.logical_dtype in {LogicalDType.INT64, LogicalDType.DECIMAL}:
        return torch.int64
    if column_type.logical_dtype == LogicalDType.FP32:
        return torch.float32
    if column_type.logical_dtype == LogicalDType.BOOL:
        return torch.bool
    return torch.float64


def _rename_type(column_type: ColumnType, name: str) -> ColumnType:
    return ColumnType(
        name=name,
        duckdb_type_id=column_type.duckdb_type_id,
        duckdb_type_repr=column_type.duckdb_type_repr,
        logical_dtype=column_type.logical_dtype,
        nullable=column_type.nullable,
        precision=column_type.precision,
        scale=column_type.scale,
    )


def _is_decimal(column_type: ColumnType) -> bool:
    return column_type.logical_dtype == LogicalDType.DECIMAL
