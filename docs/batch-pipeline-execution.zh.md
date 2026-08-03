# Batch Pipeline Execution 设计说明

本项目开始把显式 scan chunk execution 演进为成熟数据库常见的 vectorized / morsel-driven / pipeline execution 形态。

## 当前落地范围

已落地的第一阶段覆盖安全的单表局部链路：

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

第二阶段已把 `PartitionConfig` 的单表 aggregate fragments 并入同一套 batch pipeline：

```text
ScanBatchOperator
  -> FilterBatchOperator
  -> ProjectBatchOperator
  -> LocalAggregateBatchOperator
  -> optional local Project/Sort
  -> FinalMerge on host
```

第三阶段把 scan source 从 `LIMIT/OFFSET` 分块改为 DuckDB Arrow `RecordBatchReader`：

```text
DuckDB SELECT once
  -> fetch_record_batch(rows_per_batch=chunk_size)
  -> Arrow RecordBatch
  -> TensorRecordBatch-backed PhysicalTable
  -> downstream BatchOperator
```

这样同一个 scan 只执行一次 DuckDB 查询，避免大表上每个 chunk 重复 OFFSET 跳过。scan projection 还会把 `DECIMAL(p,s)` 下推为 scaled `int64`，把 TPC-H 低基数字符串下推为静态 dictionary id，把 DATE 下推为 `YYYYMMDD` int，减少 Python object conversion。

第四阶段加入 scan predicate pushdown：`ScanBatchOperator` 会把 DuckDB scan node 的 `Filters` 规划为 pushed filters 与 residual filters。pushed filters 进入 Arrow scan source 的 `WHERE`，因此只为 pushed filters 服务的列不再需要传到 PyTorch；residual filters 继续由 PyTorch tensor filter 执行，避免不支持表达式被静默跳过。

## 对成熟数据库方案的对应

| 成熟数据库概念 | 当前项目对应 |
| --- | --- |
| DuckDB `DataChunk` / Velox `RowVector` / Arrow `RecordBatch` | `TensorRecordBatch` |
| Vectorized operator | `BatchOperator.next_batch()` |
| Pipeline-friendly operator | scan / filter / project |
| Pipeline breaker | aggregate / join build / sort / distinct / window |
| Morsel/chunk scheduling | `ScanChunkConfig.chunk_size` / `PartitionConfig.chunk_size` + Arrow RecordBatch stream |
| Local + global 两阶段执行 | `PartitionConfig` 已使用 `LocalAggregateBatchOperator -> FinalMerge` |

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
  - `SortBatchOperator`
  - `execute_batch_pipeline()`
- `tpch_torch/backend/physical_scan.py`
  - `fetch_physical_table_stream()`
  - DuckDB Arrow RecordBatch streaming scan
  - scan-time DECIMAL / DATE / static dictionary encoding
- `tpch_torch/backend/physical_scan_pushdown.py`
  - scan predicate pushdown planner
  - pushed/residual filter split
- `tpch_torch/backend/physical_pipeline_aggregate.py`
  - `LocalAggregateBatchOperator`
- `tpch_torch/backend/physical_chunked.py`
  - `ScanChunkConfig`
  - safe plan analysis
  - batch pipeline dispatch
- `tpch_torch/backend/physical_partitionable.py`
  - `PartitionConfig`
  - partial rows from batch pipeline
  - final aggregate merge
- `tpch_torch/backend/triton_hash_join.py`
  - explicit unique-key Triton hash join primitive
  - atomicCAS build + double hashing
  - 4-thread group probe
- `tests/test_scan_chunk_execution.py`
  - 验证 scan/filter/project correctness
  - 验证 scan chunk metadata
  - 验证不再 per chunk 调用 `PhysicalPlanExecutor.execute()`
  - 验证 batch pipeline 使用 Arrow stream scan，而不是 OFFSET/LIMIT fetch
  - 验证 join/aggregate 显式拒绝

## 下一步演进 TODO

1. 把 `PhysicalTable.projected()` 扩展为可保留 child batch metadata，避免 projection 后 batch_meta 重置。
2. 把 FinalMerge 也封装成显式 `FinalAggregateOperator`，减少 row-dict host merge。
3. 实现完整 hash join 的 build/probe batch pipeline：build side 构建全局 tensor hash state，probe side 按 chunk 输出 joined batches。
4. 将 Triton hash join primitive 从 unique build key 扩展到 SQL multimap：duplicate key chaining / prefix-sum output sizing / NULL policy。
5. 实现 local top-k + final top-k merge，替代对全局 sort/limit 的显式拒绝。
6. 增加 pipeline scheduler，把 chunk 级任务调度从串行 pull 发展到 CPU thread / GPU stream 可配置执行。
