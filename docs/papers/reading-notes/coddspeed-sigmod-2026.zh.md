# 精读：CoddSpeed - Hardware Accelerated Query Processing in Microsoft Fabric

- PDF：[`../3788853.3803077.pdf`](../3788853.3803077.pdf)
- 论文：Matteo Interlandi et al., SIGMOD Companion, 2026
- DOI：<https://doi.org/10.1145/3788853.3803077>

## 1. 核心问题

CoddSpeed 讨论的是工业级数据平台 Microsoft Fabric 如何接入硬件加速查询。它不只是一个 GPU SQL kernel 论文，而是一个系统架构论文：如何在已有云原生数据平台里，支持 GPU、FPGA、ASIC、NVLink、InfiniBand 等不同 accelerator / network，同时保持 host engine 可用、可回退、可扩展。

论文最成熟的实现是 derived from Tensor Query Processor 的 GPU execution engine，并报告了 Fabric 中数据移动系统和多 GPU/node 场景的结果。

## 2. 总体架构

CoddSpeed 的核心抽象是 **Coprocessor Abstraction Layer, CAL**：

```text
Host data engine, e.g. Fabric DW / SQL Server
  → optimizer 选择可 offload fragment
  → CAL API
  → GPU / FPGA / ASIC coprocessor
  → result back to host iterator
```

CAL 的目标是让不同 host engine 能用统一 API 调用不同硬件后端。

## 3. 设计原则

论文给出一组工程原则：

- **Simplicity**：CAL 必须能被不同 host engine 使用，不能要求主引擎大改。
- **Hardware flexibility**：硬件变化很快，架构要能接 GPU、FPGA、ASIC 甚至网络加速器。
- **Minimalist coprocessor**：coprocessor 不必承担完整 DBMS 责任，而是执行可 offload 的 query fragments。
- **Partitionable execution**：受限于 accelerator memory，查询片段应能按 partition/chunk 逐步送入 coprocessor。
- **Query fallback**：coprocessor fragment 失败时，host 能重新执行该 fragment，而不是整个 query 失败。
- **Data movement as a service**：当 compute 加速后，数据访问、cache、shuffle、network 会成为瓶颈，需要独立服务层处理。

## 4. CAL 能力协商

CAL 用 capability 描述 coprocessor 能做什么。

### 4.1 Local capabilities

描述单个 relational/scalar operator 是否支持。例如：

- 是否支持 `GroupBy`；
- grouping columns 是否少于某阈值；
- 是否支持 left semi join；
- 是否支持某些 scalar functions，如 `LIKE`、`UPPER`、`COS`。

### 4.2 Global capabilities

描述更复杂的限制，例如：

- query fragment 复杂度；
- memory 限制；
- plan shape 限制；
- 是否满足 partitionable 约束。

本仓库未来如果支持国产卡、多 backend，也需要类似 capability registry，不能假设所有 torch op 在所有 device 上都可用且高效。

## 5. Partitionable execution model

Coprocessor memory 通常小于 host memory。CoddSpeed 用 partitionable execution 缓解：

```sql
SELECT F.x, D.y
FROM F JOIN D ON F.fk = D.pk
```

如果大表 `F` 放不进 coprocessor，但小表 `D` 和每个 `F_k` chunk 可以放入，则可以：

1. 把 `D` 送入 coprocessor；
2. 逐块发送 `F_k`；
3. 每块执行 join；
4. host 端 union partial results。

这类似分布式数据库的 partition 思想，但被用于单节点/多节点 accelerator memory management。

## 6. Query fallback

CoddSpeed 要求每个 coprocessor execution 要么返回完整 partition result，要么失败。这样 host 可以重新处理失败 fragment，不必担心 partial result 已被上游消费。

这给本仓库的启发：如果将来引入国产卡 backend 或多执行后端，fallback 应是显式、fragment-level、可验证的，而不是 silent fallback 到 DuckDB。

## 7. Data Movement as a Service

论文强调：当 coprocessor compute throughput 提升后，数据访问和数据移动成为瓶颈。CoddSpeed 引入 Data Abstraction Layer, DAL，处理：

- local cache；
- remote storage；
- staging / caching medium；
- GPU memory、CPU memory、disk 等 tiering；
- Ethernet、NVLink、InfiniBand 等传输选择；
- local writes / remote reads；
- zero-copy data access。

这说明生产级 GPU DB 不能只优化算子，必须把数据移动当作 first-class system concern。

## 8. Host 与 coprocessor 执行流程

Host 编译阶段会把可 offload query fragment 翻译成 Substrait，并创建 coprocessor iterator。

Iterator 状态机大致流程：

1. `CreateQueryPlan`：发送 Substrait plan；
2. `SendChunk`：发送非 partitionable tables；
3. `Prepare`：coprocessor 构建内部结构，例如 hash table；
4. 对 partitionable table 逐 chunk 发送；
5. `Execute`：coprocessor 执行并返回结果；
6. host iterator 将 offloaded result 与 fallback result 合并。

这和当前仓库的 DuckDB JSON plan → PyTorch executor 类似，但 CoddSpeed 更强调 host/coprocessor API、chunking、fallback 和 Substrait 作为跨系统计划格式。

## 9. GPU coprocessor

论文称 GPU engine 是 hardened and optimized version of TQP。公开文本中提到的 GPU 优化包括：

- input-format-based reordering of operators materializing join outputs；
- fast join implementation for PK-FK joins；
- optimized hash table format for unique inputs。

这表明 CoddSpeed 已从研究型 TQP 走向工程化 GPU coprocessor：关注特定 join pattern、输入格式、hash table 布局和与 Fabric 的执行模型集成。

## 10. 实验结果与系统意义

论文报告：

- GPU engine 在 TPC-H 100GB 和 customer workloads 上有 8–20× 级别提升；
- multi-GPU/node TPCH-1TB 场景可达 25–30× 相对 scale-out baseline 的提升；
- 还展示了 FPGA、APU 等不同 coprocessor 通过 CAL 接入的原型结果。

这些结果重点展示的是系统集成与硬件抽象，而不是单个 SQL 算子的微基准。

## 11. 对当前仓库的启发

当前仓库关注：

```text
SQL → DuckDB JSON plan → TQPOperatorGraph → PyTorch tensor backend
```

CoddSpeed 提示下一阶段系统化方向：

1. **Capability registry**：声明 backend 支持的 operator、dtype、shape、memory 限制。
2. **Fragment-level fallback**：失败时显式返回 unsupported/failure，不做 silent fallback。
3. **Partitionable execution**：面对 GPU memory 限制，需要 chunk/partition 级执行模型。
4. **Substrait / open plan boundary**：CoddSpeed 使用 Substrait 发送 fragment；当前仓库因 DuckDB exporter 限制走 JSON plan，但长期仍可保留开放 IR 边界。
5. **Data movement first-class**：benchmark 必须拆 H2D/D2H/cache/shuffle，不只看 kernel time。
6. **Host/coproc API**：把 PyTorch backend 逐步整理成可替换 coprocessor backend，而不是散在 Python executor 中。

## 12. 与 TQP/TQEx 的关系

```text
TQP       : 证明 SQL → tensor program 可行。
TQEx      : 修补 SQL 不规则性与 tensor uniform ops 的 gap。
CoddSpeed : 把 TQP-derived GPU engine 放入生产数据平台，解决 host/coprocessor、fallback、partition、data movement。
```

对本仓库来说，TQP/TQEx 更指导 operator lowering，CoddSpeed 更指导系统边界和工程化架构。


## 13. 本仓库已落地的 partitionable execution 子集

当前 `torch-query-gpu` 已加入一个显式 opt-in 的 CoddSpeed-style partitionable executor：

```text
SQL → DuckDB/Sirius-like frontend → TQPOperatorGraph
  → PyTorchGraphExecutor(partition_config)
  → execute_partitionable_physical_plan
  → chunk scan + local physical aggregate
  → host merge partial aggregate rows
```

实现对应论文中的几个点：

- **partitionable over one table**：当前限定一个大表，例如 `lineitem`；
- **fragment-level execution**：每个 row range 都执行同一个 physical graph fragment；
- **local/global aggregate split**：Q6/Q1 这类聚合在 chunk 内先 local aggregate，再由 host 合并；
- **no silent fallback**：不满足单表 aggregate shape 时显式抛 `UnsupportedPlanError`；
- **coprocessor minimalist**：PyTorch executor 只处理 chunk fragment，partial results 的组合由 host Python 层完成。

当前还没有实现论文中的 non-partitionable table `Prepare` 缓存、failed chunk fallback、多 runtime 并发、join partitioning 和 data movement service。
