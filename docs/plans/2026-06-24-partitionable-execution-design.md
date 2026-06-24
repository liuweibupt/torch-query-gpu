# CoddSpeed Partitionable Execution 设计

## 背景

CoddSpeed 的 partitionable execution 用于解决 coprocessor/GPU 显存小于 host 数据的问题。论文定义：若查询 `Q` 对表 `T_i` 可分片，则 `Q(T_i) = union_k Q(T_i^k)`；对于带聚合的查询，需要把不可直接 union 的全局聚合拆成 local aggregate 与 host/global merge。

当前仓库的执行链路已经是：

```text
SQL → Sirius/DuckDB JSON frontend → TQPOperatorGraph → PyTorch physical executor
```

本次设计不新增 query-specific Python 脚本，而是在 PyTorch physical executor 外增加显式 opt-in 的 partitionable fragment executor。

## 目标

1. 支持从 SQL 前端 lowering 后的 physical graph 自动识别可分片 fragment。
2. 先覆盖单表 scan-heavy 聚合查询：TPC-H Q6（ungrouped SUM）和 Q1（GROUP BY + SUM/AVG/COUNT）。
3. 分片执行时每个 chunk 仍走同一份 physical graph 和 PyTorch tensor 算子。
4. host 端做 partial aggregate merge，模拟 CoddSpeed host iterator 聚合 partial result 的职责。
5. 不支持的 fragment 显式抛 `UnsupportedPlanError`，不 silent fallback 到默认整表路径。

## 非目标

- 本批不实现 join partitioning、runtime 并发、多 GPU runtime、failed partition 的 host fallback。
- 本批不追求性能最优；主要验证链路、显存分片模型和正确性。

## 架构

```text
run_sql_with_frontend(..., partition_config)
  → compile_tqp_plan(sql)
  → PyTorchBackend.execute(..., partition_config)
  → PyTorchGraphExecutor.execute(..., partition_config)
  → execute_partitionable_physical_plan(graph, config)
      → validate single-table aggregate fragment
      → row ranges over partition table
      → PhysicalPlanExecutor(scan_ranges={table: range}, enable_fusion=False)
      → merge partial aggregate rows on host
```

其中 `enable_fusion=False` 很关键：Q1 默认 fused path 会直接读完整 lineitem resident tensor cache；分片执行必须关闭 fused fast path，确保每个 partition 读取 SQL graph 中的 scan chunk。

## 支持范围

本批支持：

- 一个 partitionable table scan；
- 无 join/CTE/delim/limit；
- 一个 aggregate node；
- aggregate 函数：`sum` / `sum_no_overflow` / `count` / `count_star` / `min` / `max` / `avg`；
- `avg` 需要输出中存在 count aggregate，用加权方式合并；Q1 的 TPC-H schema 非空，使用 `count_star` 合并平均值。

## 结果合并

- 无 group key：把每个 chunk 的单行 partial aggregate 合并成一行。
- 有 group key：以输出中的 group columns 作为 key，逐组累加 partial aggregate。
- `SUM`/`COUNT`：累加。
- `MIN`/`MAX`：取全局 min/max。
- `AVG`：累加 `partial_avg * partial_count`，最后除以全局 count。
- 空 chunk 的 `NULL` partial 值忽略；如果所有 partition 都没有有效值，保留 `None`。
- 如果原 graph 有 sort node，则当前仅对 group key 做 host 端稳定排序，覆盖 Q1 这种 group-key order by。

## CLI / Benchmark

`benchmark_query.py` 增加：

```bash
--partition-table lineitem --partition-chunk-size 100000
```

benchmark 的 sample 仍是 end-to-end timing。默认路径不变；只有显式传入 partition 参数才启用分片执行。

## 风险

- 对 SF=1 CPU 跑法，partitionable execution 可能更慢，因为重复 frontend/executor 开销与多次 DuckDB chunk fetch 超过显存收益。
- 它的主要收益是降低 peak tensor residency，并让未来 GPU 显存不足时能执行更大表。
- 当前 host merge 在 Python 中完成，后续可迁移为 torch/global aggregate node。
