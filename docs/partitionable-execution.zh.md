# CoddSpeed-style Partitionable Execution

本文记录本仓库加入的最小版 **partitionable execution**。它参考 CoddSpeed 论文中的 coprocessor iterator 思路：host 侧把可分片大表按 chunk 送入 coprocessor/GPU runtime，每个 chunk 执行同一个 query fragment，再由 host 合并 partial results。

## 1. CoddSpeed 论文中的核心思想

CoddSpeed 关注生产系统中 accelerator memory 小于 host memory 的问题。论文给出的定义可以概括为：如果查询 `Q` 对表 `T` 满足

```text
Q(T) = union_k Q(T_k), where T = union_k T_k
```

则该查询对 `T` 是 partitionable 的。对于带聚合的 SQL，结果不能简单拼接，需要把 fragment 改写为 local aggregate，然后由 host/global aggregate 合并 partial results。

CoddSpeed 的 iterator 状态大致是：

```text
CreateQueryPlan → Send non-partitionable chunks → Prepare
  → Send partitionable chunk → Execute → Results
  → repeat chunks → host/global merge or fallback
```

本仓库当前只实现其中的 correctness-first 子集：单表 scan-heavy aggregate fragment 的 chunk execution 与 host merge。

## 2. 当前实现位置

```text
SQL
  → DuckDB/Sirius-like frontend
  → TQPOperatorGraph
  → PyTorchBackend.execute(..., partition_config)
  → PyTorchGraphExecutor
  → execute_partitionable_physical_plan()
  → BatchOperator pipeline
  → Scan/Filter/Project → LocalAggregateBatchOperator
  → host merge partial aggregate rows
```

关键文件：

- `tpch_torch/backend/physical_partitionable.py`
  - `PartitionConfig(table, chunk_size)`
  - `row_ranges(row_count, chunk_size)`
  - `execute_partitionable_physical_plan(...)`
  - graph shape 校验与 partial aggregate merge
- `tpch_torch/backend/physical_pipeline.py`
  - `BatchOperator.next_batch()`
  - `ScanBatchOperator` / `FilterBatchOperator` / `ProjectBatchOperator`
  - chunk scan 生成 TensorRecordBatch-backed table
- `tpch_torch/backend/physical_pipeline_aggregate.py`
  - `LocalAggregateBatchOperator`
- `tpch_torch/backend/graph.py`
  - 显式 `partition_config` 分发到 partitionable executor
- `tpch_torch/benchmark.py` 与 `scripts/benchmark_query.py`
  - 冷/热 benchmark 支持 `--partition-table` / `--partition-chunk-size`

## 3. 为什么迁移到 batch pipeline

早期 partitionable path 是对每个 chunk 构造 `PhysicalPlanExecutor(scan_ranges=...)`，等价于“按 chunk 重跑整棵 physical executor”。这能验证 CoddSpeed-style host/chunk 边界，但不是成熟数据库的 vectorized pipeline 形态。

当前实现改为：

```text
ScanBatchOperator.next_batch()
  -> FilterBatchOperator.next_batch()
  -> ProjectBatchOperator.next_batch()
  -> LocalAggregateBatchOperator.next_batch()
  -> optional local Project/Sort
  -> FinalMerge
```

这样 Q1/Q6 都不是退回 query-specific 脚本，而是从同一个 frontend-lowered physical graph 构造 batch operators；aggregate 作为 pipeline breaker 被拆成 local aggregate 和 final merge。

## 4. 支持范围

当前支持：

- 一个 partitionable table scan；
- `scan/filter/project/aggregate/sort`；
- 一个 aggregate node；
- `SUM` / `COUNT` / `COUNT(*)` / `MIN` / `MAX` / `AVG`；
- 无 group key 的 SUM 聚合，例如 TPC-H Q6；
- group-by aggregate，例如 TPC-H Q1；
- `AVG` 依赖输出中存在 `COUNT`/`COUNT(*)`，用 weighted merge 合并。

当前显式不支持：

- join partitioning；
- 多表 co-partition；
- CTE / delim join / window / set operation 的 partitionable fragment；
- failed partition 的 host fallback；
- 多 runtime 并发、overlap H2D 与 compute。

## 5. Host merge 方法

无 group key：

```text
partial_1: {revenue: 5}
partial_2: {revenue: 14}
→ final: {revenue: 19}
```

有 group key：

```text
key = (l_returnflag, l_linestatus)
SUM/COUNT: sum(partials)
MIN/MAX: min/max(partials)
AVG: sum(partial_avg * partial_count) / sum(partial_count)
```

这对应 CoddSpeed 中 “coprocessor 只返回每个 partition 的完整 result，partial results 如何组合由 host engine 负责” 的设计。

## 6. 使用方式

默认路径不变：

```bash
python -m scripts.benchmark_query \
  --db data/tpch_sf1.duckdb \
  --query 6 \
  --device cpu \
  --cold-runs 0 --warmup-runs 1 --hot-runs 3
```

启用 partitionable execution：

```bash
python -m scripts.benchmark_query \
  --db data/tpch_sf1.duckdb \
  --query 6 \
  --device cpu \
  --cold-runs 0 --warmup-runs 1 --hot-runs 3 \
  --partition-table lineitem \
  --partition-chunk-size 100000
```

Python API：

```python
from tpch_torch.backend.physical_partitionable import PartitionConfig
from tpch_torch.runner import run_sql_with_frontend

result = run_sql_with_frontend(
    con,
    sql,
    device="cuda",
    frontend="sirius",
    partition_config=PartitionConfig(table="lineitem", chunk_size=100_000),
)
```

## 7. 与性能结果的关系

partitionable execution 的首要收益不是让 SF=1 CPU eager PyTorch demo 更快，而是降低单次 device resident input 的峰值规模，使大表可以分 chunk 进入 coprocessor。当前实现每个 chunk 都会重复一次 DuckDB column fetch、tensor conversion、physical interpreter 调度和 Python host merge，因此小数据或 CPU 场景通常会变慢。

预期收益出现的场景是：

- 整表 tensor 无法放入 GPU 显存；
- 小表或 common fragment 可以 `Prepare` 后复用；
- chunk fetch 与 GPU compute 可以 overlap；
- local aggregate 已下沉到 batch pipeline；final merge 后续可继续从 Python row merge 改成 tensor merge；
- join fragment 能对大事实表分片、小维表常驻。

当前版本先把 CoddSpeed 的系统边界和正确性链路接入 engine，后续优化再围绕 runtime 并发、显存统计、non-partitionable table residency、join partitioning 展开。

## 8. 当前 SF=1 观测结果

测试环境使用仓库现有 `/work/torch-query-gpu/data/tpch_sf1.duckdb`，CPU，`cold-runs=0`、`warmup-runs=1`、`hot-runs=3`、partition chunk size = 1,000,000 rows。命令通过 `python -m scripts.benchmark_query` 从仓库根目录运行。

| Query | Path | Hot median | 相对 baseline |
| --- | --- | ---: | ---: |
| Q6 | 默认 physical interpreter | 461.311 ms | 1.00× |
| Q6 | partitionable over `lineitem` | 784.060 ms | 0.59× |
| Q1 | 默认 fused physical primitive + resident tensor cache | 111.731 ms | 1.00× |
| Q1 | partitionable over `lineitem` + local aggregate pipeline | 1920.953 ms | 0.058× |

解释：当前版本为了严格遵守 partitionable execution 的内存边界，每个 chunk 都重新从 DuckDB 拉取列、转换 tensor、执行 local fragment，并在 host Python 合并 partial aggregates。Q1 默认路径有 resident tensor cache 和整表 fused `bincount`；partitionable Q1 无法复用整表 resident tensor cache，因此在 SF=1 CPU 上显著更慢。partitionable path 当前主要证明 CoddSpeed-style host/chunk/coprocessor 执行边界和 mature DB-style local/final aggregate pipeline 是通的；它的价值在整表无法放入 GPU 显存、或未来实现 non-partitionable table residency、chunk prefetch/overlap、多 runtime 并发后才会体现。
