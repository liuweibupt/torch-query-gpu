# 精读：TQEx - Tensor-based Query Engine Enhanced by Bridging the Gap

- PDF：[`../3769835.pdf`](../3769835.pdf)
- 论文：Haitao Zhang et al., Proc. ACM Manag. Data / SIGMOD, 2025
- DOI：<https://doi.org/10.1145/3769835>

## 1. 核心问题

TQP 证明了 SQL 可以编译为 tensor programs，并借助 TCR 获得跨硬件可移植性。但 TQEx 指出：SQL workload 与 tensor operations 之间存在根本差异。SQL 数据和计算通常不规则，而 tensor op 更偏好规则、批量、loop-free、uniform 的计算。

TQEx 的目标是系统分析这个 gap，并提出 bridge guidelines 与具体算子实现，减少 TQP 中因 padding、loop、重复访问导致的存储和计算开销。

## 2. 主要贡献

1. 首次系统分析 SQL 与 tensor 的 gap：数据不规则、计算不规则、tensor 偏好大 tensor 和 loop-free 结构。
2. 提出两条核心 guideline：loop unrolling 与 early elimination of padding。
3. 针对 variable-length string 设计紧凑 tensor 存储和 LIKE 算法。
4. 重新设计 join 与 group aggregate 等 tensor SQL operators。
5. 扩展到 multi-XPU 大规模处理。
6. 实验显示 TQEx 在 TPC-H 上相对 TQP 平均 9.6×、峰值 41.9× 加速，并在支持查询上超过 HeavyDB/DuckDB 等 baseline。

## 3. SQL 与 tensor 的 gap

### 3.1 SQL 的不规则性

SQL workload 常见两类不规则：

- **Data irregularity**：字符串、变长字段、不同 tuple 的数据长度不同；如果 padding 到最大长度，会产生大量无效存储。
- **Computation irregularity**：join bucket 大小、LIKE 匹配长度、CASE 分支等导致每个 tuple 计算量不同。

论文用 TPC-H 中 `O_COMMENT` 字符串长度和 `LINEITEM` self-join 匹配数说明这种不规则性。

### 3.2 Tensor 的偏好

Tensor op 通常对整个 tensor 做 uniform computation。高效 tensor program 偏好：

- **large tensors**：减少 kernel launch overhead；
- **loop-free structure**：减少频繁 kernel launch、中间 materialization 和 locality 破坏。

这解释了为什么简单把 SQL 不规则流程 padding 成 uniform tensor loop 会慢。

## 4. 两条 bridge guidelines

### 4.1 Loop unrolling

把迭代结构转成 batched tensor operations。以 join probe 为例，不逐轮取每个 probe tuple 的下一个 candidate，而是生成所有 candidate id pairs，并批量比较 join keys。

效果：

- 避免多轮 kernel launch；
- 保留连续 bucket 的 spatial locality；
- 减少反复读写中间 tensor。

### 4.2 Early elimination of padding

先把无效 padding 或不可能匹配的候选过滤掉，再做批量计算，避免对 dummy work 做 tensor op。

效果：

- 降低计算量；
- 减少中间 tensor；
- 对 string、join、CASE/LIKE 等不规则场景尤其重要。

## 5. 变长字符串存储与 LIKE

TQEx 不使用 TQP 那种 padding matrix 存字符串，而采用三 tensor 表示：

```text
str_set : concatenated characters
start   : each string start offset
len     : each string length
```

这接近 Arrow 风格的 variable-length buffer，但关键在于如何用 tensor ops 操作它。

### 5.1 Start-position-aware matching

适用于 equal、prefix、suffix。核心流程：

1. 按长度预过滤字符串；
2. 用 `arange` + broadcast 生成候选 substring index matrix；
3. 批量 `index_select` / masked select 提取候选 substring；
4. 与 pattern 做 broadcast comparison；
5. 对每行做 logical reduce，得到 match mask。

### 5.2 Start-position-unaware matching

适用于 `%ab%` 这类 pattern。需要生成每个字符串所有可能起点的候选 substring，再做批量匹配；核心仍是 loop unrolling + padding elimination。

## 6. Join 算子

### 6.1 Index join

TQEx 的 index join：

1. 对 build table join key 排序，得到 `sorted_keys` 和 `sorted_ID`；
2. 用 `unique_consecutive` 得到 unique index 和每个 key 的 count；
3. 用 prefix sum 定位每个 bucket 在 `sorted_ID` 中的范围；
4. probe 侧通过 binary search 查 unique index；
5. 利用 bucket start/count 批量生成 match pairs。

如果 index 小到能进 cache，binary search 更友好；index 过大时排序 probe 侧可改善 locality。

### 6.2 Improved hash join

TQEx 改进 TQP hash join 的关键：

- **compact hash table**：每个 hash bucket 连续存储，所有 bucket 拼成 1D tensor，避免 padding matrix；
- **bucket dependency elimination**：为同 bucket 内每行分配 inner-bucket offset，使行可以独立并发插入；
- 构建阶段只访问 build table 一次；
- probe 阶段对所有 probe tuples 同时 lookup。

这解决了 TQP hash join 多轮 build/probe 和重复访问的问题。

## 7. Group aggregate

基础流程仍是 sort + `unique_consecutive` + group id + scatter/reduce。但 TQEx 注意到：排序后同 group 的 tuple 相邻，直接 scatter 到同一个 `agg_res[i]` 会导致大量并发写竞争。

论文提出改进策略：将每个 group 拆到多个 virtual slots，先做分散聚合，再把 virtual slots reduce 回真实 group。这样可以降低同一地址的 contention。

这对当前仓库很重要：`index_add`/`scatter_reduce` 虽然表达简单，但 group skew 会造成性能问题。

## 8. Multi-XPU 扩展

TQEx 将查询处理扩展到多个 XPU。核心问题包括：

- 数据分区；
- 跨 device 数据移动；
- join/aggregate 的局部与全局阶段；
- 通信开销与计算并行的平衡。

虽然论文细节依赖其实现，但对本项目的启发是：单 GPU tensor primitives 只是第一步，多卡需要 partition-aware physical plan 和通信层。

## 9. 实验结论

论文报告：

- TPC-H 上平均 9.6×、最高 41.9× 快于 TQP；
- 比 HeavyDB 快 27.9×；
- SF=100 支持查询上，比 DuckDB 快 12.2×、比 HeavyDB 快 22.7×；
- ablation 显示 string storage/LIKE、join、aggregate 等优化对性能贡献明显。

注意：这些结果依赖论文环境与支持查询集合，应作为方向参考，不直接等价于本仓库性能目标。

## 10. 对当前仓库的启发

当前仓库已实现：

- SQL → DuckDB physical plan → TQPOperatorGraph；
- PyTorch tensor backend；
- 部分 join、aggregate、Q1 fusion、RLE primitive。

TQEx 提示下一步应重点做：

1. **消除 padding 思维**：字符串和变长结果不要强行变固定矩阵。
2. **join candidate generation 批量化**：减少 per-bucket/per-row loop。
3. **group skew 处理**：`index_add`/`scatter_reduce` 需要 virtual slot 或分段聚合策略。
4. **字符串列模型**：从 dictionary-only 逐步引入 `str_set/start/len` 形式。
5. **硬件 aware operator selection**：不同 device 上 `unique/sort/scatter` 的性能差异很大。

## 11. 与 TQP 的关系

TQP 建立了“SQL 可以映射到 tensor program”的可行性；TQEx 则进一步说明：要想高效，必须尊重 SQL 的不规则性和 tensor 的规则性之间的差异。换言之：

```text
TQP: 可行性与可移植性
TQEx: gap 分析与 operator-level 优化
```
