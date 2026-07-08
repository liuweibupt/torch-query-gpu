# P1/P2 TensorRecordBatch, Decimal, and Multi-Precision Operators Design

## Goal

Implement the README P1/P2 line after the P0 correctness fixes:

- introduce an explicit `TensorRecordBatch` / `ColumnMeta` type layer;
- define DuckDB → PyTorch dtype mapping;
- represent DECIMAL as `int64 + scale` for correctness-first tensor execution;
- propagate metadata through scan/filter/projection enough for the physical interpreter;
- extend join and grouped SUM coverage to INT64 / FP32 / FP64 / DECIMAL;
- add a first tensor hash-style join probe prototype and tests without claiming it is a mature GPU hash table.

## Architecture

### ColumnMeta

`ColumnMeta` is immutable and describes one tensor column:

- `logical_dtype`: `int64`, `fp32`, `fp64`, `decimal`, `string_dict`, `bool`, `date`, `unknown`;
- `torch_dtype`: physical tensor dtype;
- `nullable`: whether a validity mask may be present;
- `scale` / `precision`: DECIMAL metadata;
- `dictionary`: string dictionary vocabulary.

Metadata is a value object and can be stored both in the new record batch and in the existing `PhysicalValue` compatibility layer.

### TensorRecordBatch

`TensorRecordBatch` is a columnar immutable-ish boundary object:

- `columns`: name → tensor;
- `meta`: name → `ColumnMeta`;
- `validity`: name → optional bool mask;
- `row_count` inferred and validated;
- `filter(mask)`, `gather(indices)`, `project(items)` return new batches.

The first implementation is a reusable type substrate, not a full replacement for `PhysicalTable` in one patch. `PhysicalValue` gets an optional `meta` field so both systems can interoperate.

### Decimal representation

DECIMAL is represented as scaled `int64`:

```text
DECIMAL(12,2): 123.45 -> tensor int64 12345, ColumnMeta(scale=2, precision=12)
```

Rules for this batch:

- scan encodes DuckDB DECIMAL values into int64+scale;
- decimal comparison/join aligns scale to the larger scale before comparing;
- decimal `+` / `-` aligns scales and returns decimal at max scale;
- decimal `*` multiplies int64 payloads and returns scale sum;
- decimal `/` returns fp64 for correctness-first behavior;
- decimal `SUM` keeps int64+scale metadata;
- decimal `AVG` returns fp64.

### P2 operators

- Sort/searchsorted join already supports integer and floating keys after P0; this batch adds explicit decimal scale-aware key comparison through `ColumnMeta`.
- Grouped `SUM` preserves decimal metadata for decimal inputs and continues to preserve tensor dtype for INT64/FP32/FP64.
- The first hash-style join prototype is a tensor dictionary-encoding probe helper for single-column keys. It is intended for testing API boundaries and future strategy selection; it is not documented as a fully optimized GPU hash table.

## Non-goals

- No full query-planner replacement with `TensorRecordBatch`.
- No full SQL DECIMAL overflow/rounding model.
- No production-grade GPU open-addressing hash table in this batch.
- No compressed-column metadata integration; that remains P3.

## Validation

- Focused tests for metadata construction, record batch filter/gather/project, dtype mapping, decimal encoding/decoding, decimal arithmetic, decimal join, mixed dtype join, grouped decimal SUM, and hash-style probe.
- Existing P0/physical/TPC-H tests remain green.
