# Q1 Fusion and Physical Coverage Design

## Goal

Improve Q1 performance and expand TPC-H automatic operator coverage without restoring query-id template execution. The invariant remains:

```text
SQL -> DuckDB JSON physical plan -> TQPOperatorGraph -> physical/fused PyTorch operators
```

## Approach

### Q1 performance

The Q1 default path already enters `execute_physical_plan()`. Its current cost comes from generic physical interpretation: multiple projection nodes, repeated expression evaluation, generic `torch.unique(..., dim=0)` grouping, and repeated output materialization steps. This batch adds a graph-lowered fusion hook that recognizes the DuckDB Q1 physical shape and executes the heavy scan/filter/project/group/order segment with one fused tensor primitive.

The fusion is selected from the lowered graph shape and source SQL, not from a standalone user-facing script or DuckDB result fallback. Unsupported shapes continue through the existing interpreter.

### TPC-H automatic coverage

Add a physical-coverage probe that attempts to run each TPC-H query through `execute_physical_plan()` with graph recipes disabled. This produces an explicit supported/blocked matrix and makes migration progress measurable. The probe is a test/documentation aid; it must not hide failures in normal execution.

## New modules

- `tpch_torch/backend/physical_fusion.py`: graph-shape fusion entry point and fused Q1 executor.
- `tpch_torch/backend/physical_patterns.py`: small predicates for recognizing Q1-like physical graphs.
- `tpch_torch/physical_coverage.py`: coverage probe for TPC-H physical interpreter support.

## Q1 fused data flow

```text
TQPOperatorGraph(Q1)
  -> try_execute_fused_physical_plan()
  -> fetch required lineitem columns as tensors
  -> apply shipdate scan filter once
  -> gather selected rows once
  -> compute discount/charge expressions once
  -> encode returnflag/linestatus dense group ids
  -> torch.bincount sums/counts
  -> decode tiny grouped result and sort
```

## Error handling

Fusion only runs when the graph is recognized as canonical Q1. If not recognized, it returns `None` and the normal physical interpreter runs. This is an explicit optimizer decision, not a fallback from failure. Normal interpreter failures still surface as `UnsupportedPlanError`.

## Testing

- Failing test first: Q1 physical execution calls fusion and not the old direct graph helper.
- Correctness: Q1 fused output matches DuckDB baseline on fixture and SF=0.01/SF=1 smoke.
- Coverage: physical-only TPC-H probe reports Q1/Q6/Q12/Q14/Q19 supported and does not call `tpch_graph_qXX` recipes.
- Performance smoke: benchmark Q1 before/after and document hot median; do not treat noisy benchmark as correctness.
