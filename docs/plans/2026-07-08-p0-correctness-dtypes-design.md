# P0 Correctness and dtype Design

## Goal

Implement the new README P0 TODO items first, before broad type-system or hash-join work:

1. Prevent floating-point join keys from being truncated to `int64`.
2. Make grouped integer `MIN` / `MAX` reductions safe.
3. Add basic NULL-aware boolean and aggregate semantics over the existing `PhysicalValue.valid` mask.

## Scope

This batch stays inside the current physical interpreter data model. It does **not** introduce the full P1 `TensorRecordBatch + ColumnMeta` catalog/type system yet. Decimal precision is still represented by the current tensor encoding path; this batch avoids new decimal abstractions and focuses on obvious correctness bugs in existing operators.

## Design

### Join key dtype handling

`physical_join.py` currently casts single-key and composite-key join inputs to `int64`. That is incorrect for floating keys because values such as `1.2` and `1.8` collapse to the same integer. The fix is to coerce join tensors to a shared comparable dtype without truncation:

- integer/integer joins use `torch.promote_types`;
- any floating join uses `float64` for correctness-first comparison;
- sorted/searchsorted join paths operate on the coerced dtype;
- metadata-backed sorted-unique joins use the same dtype path;
- multi-condition joins first generate pairs from the first condition, then tensor-filter those pairs by remaining conditions, avoiding packed `int64` composite keys.

SEMI/ANTI membership probes receive the same single-key no-truncation treatment.

### MIN/MAX initialization

`_scatter_reduce()` currently calls `torch.full(..., float("inf"), dtype=int64)` for integer tensors. That can fail or create unsafe sentinel values. The fix is a small dtype-aware fill helper:

- integer `amin`: `torch.iinfo(dtype).max`;
- integer `amax`: `torch.iinfo(dtype).min`;
- floating `amin` / `amax`: `+inf` / `-inf`.

### NULL-aware operators

`PhysicalValue.valid` already models optional/null values for outer joins and empty aggregates. This batch makes physical expressions and aggregates use it consistently:

- comparisons/arithmetic keep validity propagation;
- string comparisons, `IN`, prefix/contains/suffix also propagate input validity;
- `NOT` preserves unknown as unknown;
- `AND`/`OR` implement SQL three-valued logic for the existing validity mask;
- `CASE WHEN` treats unknown conditions as not matched;
- scalar and grouped `SUM/MIN/MAX/AVG/COUNT(DISTINCT)` ignore invalid rows and return invalid/null when no valid input contributes.

## Non-goals

- No full SQL NULL parser/binder work.
- No nullable scan metadata for arbitrary DuckDB NULL arrays beyond what current `PhysicalValue.valid` can represent.
- No GPU hash join implementation in this batch; it remains P2/P4.
- No dynamic string dictionary expansion; it remains P1.

## Validation

- New focused tests cover floating join keys, metadata-backed floating join keys, SEMI membership with floating keys, integer grouped min/max, scalar/grouped null aggregate behavior, and NULL-aware boolean/CASE/string predicates.
- Existing physical/TPC-H tests must remain green.
