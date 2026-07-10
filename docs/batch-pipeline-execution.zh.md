# Batch Pipeline Execution 设计说明

本项目开始把显式 scan chunk execution 演进为成熟数据库常见的 vectorized / morsel-driven / pipeline execution 形态。

## 当前落地范围

已落地的第一阶段只覆盖安全的单表局部链路：

```text
ScanBatchOperator.next_batch()
  -> FilterBatchOperator.next_batch()
  -> ProjectBatchOperator.next_batch()
  -> rows
```

入口仍是 SQL：

```text
SQL
  -> DuckDB/Sirius physical graph
  -> TQPOperatorGraph
  -> ScanChunkConfig(table, chunk_size)
  -> BatchOperator pipeline
  -> TensorRecordBatch-backed PhysicalTable chunks
  -> PyTorch tensor operators
```

这意味着 `ScanChunkConfig` 不再通过每个 chunk 重建并执行整棵 `PhysicalPlanExecutor`；它会构造 pull-based batch operators，然后逐个调用 `next_batch()`。

## 对成熟数据库方案的对应

| 成熟数据库概念 | 当前项目对应 |
| --- | --- |
| DuckDB `DataChunk` / Velox `RowVector` / Arrow `RecordBatch` | `TensorRecordBatch` |
| Vectorized operator | `BatchOperator.next_batch()` |
| Pipeline-friendly operator | scan / filter / project |
| Pipeline breaker | aggregate / join build / sort / distinct / window |
| Morsel/chunk scheduling | `ScanChunkConfig.chunk_size` + scan row ranges |
| Local + global 两阶段执行 | 已有 `PartitionConfig` aggregate prototype，后续并入 pipeline |

## 算子分类

### 局部算子

这些算子可以直接按 chunk 流水执行：

- Scan
- Filter
- Projection
- Cast
- Arithmetic expression
- CASE WHEN
- 简单 per-row string dictionary id 操作

### 可分解全局算子

这些算子需要 local state + final merge：

- `SUM` / `COUNT` / `MIN` / `MAX`
- `AVG = SUM / COUNT`
- Group By Aggregate
- Top-K
- DISTINCT
- Semi/Anti join 的 probe side

推荐实现：

```text
Scan chunks -> local operator state -> final merge operator -> output batches
```

### 强全局依赖算子

这些算子不能简单 chunk 后 concat：

- Hash Join / Sort Merge Join
- Global Sort
- ORDER BY + LIMIT
- Window Function
- CTE materialization
- Correlated subquery
- Set operations
- NULL-aware MARK join

当前 `ScanChunkConfig` 对这些 plan 显式 `UnsupportedPlanError`，不做整表 fallback。

## 当前关键代码

- `tpch_torch/backend/physical_pipeline.py`
  - `BatchOperator`
  - `ScanBatchOperator`
  - `FilterBatchOperator`
  - `ProjectBatchOperator`
  - `execute_batch_pipeline()`
- `tpch_torch/backend/physical_chunked.py`
  - `ScanChunkConfig`
  - safe plan analysis
  - batch pipeline dispatch
- `tests/test_scan_chunk_execution.py`
  - 验证 scan/filter/project correctness
  - 验证 scan chunk metadata
  - 验证不再 per chunk 调用 `PhysicalPlanExecutor.execute()`
  - 验证 join/aggregate 显式拒绝

## 下一步演进 TODO

1. 把 `PhysicalTable.projected()` 扩展为可保留 child batch metadata，避免 projection 后 batch_meta 重置。
2. 实现 `LocalAggregateBatchOperator` 与 `FinalAggregateBatchOperator`，把 Q1/Q6 aggregate chunk 统一到 pipeline。
3. 实现 hash join 的 build/probe batch pipeline：build side 先构建全局 tensor hash state，probe side 按 chunk 输出 joined batches。
4. 实现 local top-k + final top-k merge，替代对全局 sort/limit 的显式拒绝。
5. 增加 pipeline scheduler，把 chunk 级任务调度从串行 pull 发展到 CPU thread / GPU stream 可配置执行。
