# 精读：TQP - Query Processing on Tensor Computation Runtimes

- PDF：[`../p2811-he.pdf`](../p2811-he.pdf)
- 备用 PDF：[`../tqp-query-processing-on-tensor-computation-runtimes.pdf`](../tqp-query-processing-on-tensor-computation-runtimes.pdf)
- 论文：Dong He et al., PVLDB 15(11), 2022
- DOI：<https://doi.org/10.14778/3551793.3551833>

## 1. 核心问题

TQP 试图回答一个问题：数据库系统能否复用 AI 领域快速发展的 Tensor Computation Runtime（TCR），把 SQL 查询编译成 tensor program，并在 PyTorch、TVM、ONNX Runtime 等后端上运行。

它的动机是：硬件正在变得异构，DBMS 如果为每种 GPU/ASIC/加速器重写执行引擎，工程成本很高；而 AI runtime 已经为这些硬件提供 tensor API、kernel、device runtime 和编译/部署生态。

## 2. 关键结论

TQP 的贡献不是“用 PyTorch 写几个 SQL 算子”，而是提出一条系统路线：

```text
SQL / physical plan
  → TQP IR graph
  → operator plan
  → tensor program
  → PyTorch eager / TorchScript / ONNX / TVM
  → CPU / GPU / accelerator
```

论文给出的结果显示：TQP 能覆盖完整 TPC-H，并在 GPU 上达到或超过当时开源 GPU 数据库的部分查询性能，同时保持跨硬件/后端移植性。

## 3. TCR 能力模型

论文把 PyTorch、TensorFlow、TVM、ONNX Runtime 等统称为 TCR。TCR 提供：

- 高层 Python tensor API；
- runtime / dispatcher / compiler；
- CPU、GPU、ASIC、edge、distributed 等硬件和部署后端；
- eager execution 与 compiled graph execution 两种模式；
- 可能的优化：公共子表达式消除、operator fusion、code generation、去 Python dependency。

对数据库而言，TCR 的价值是将硬件适配问题转移给 AI runtime 生态。

## 4. Tensor 操作分类

TQP 使用的 tensor ops 覆盖关系算子需要的常见模式：

| 类别 | 代表 op | DB 用途 |
| --- | --- | --- |
| Creation | `from_numpy`, `zeros`, `ones`, `empty`, `fill`, `arange` | 创建列、mask、row id、state。 |
| Indexing/slicing | indexing, `index_select`, mask select, `narrow` | selection、projection、join gather。 |
| Reorganization | `reshape`, `view`, `squeeze`, `gather`, `scatter`, `sort` | 重排、join、排序、group。 |
| Comparison | `eq`, `lt`, `gt`, `le`, `ge`, `isnan` | predicate。 |
| Conditional/search | `where`, `bucketize` | CASE、binary search、join lookup。 |
| Arithmetic/logical | `add`, `mul`, `div`, `sub`, logical ops | projection、filter。 |
| Combine | `concat`, `stack` | 多列 key、拼接结果。 |
| Reduction | `sum`, `max`, `min`, `mean`, `all`, `any` | aggregate。 |
| Grouped reduction | `scatter_add`, `scatter_min`, `scatter_max`, `scatter_mean` | group-by aggregate。 |
| Histogram/distinct | `bincount`, `histc`, `nonzero`, `unique`, `unique_consecutive` | group id、count、distinct。 |

## 5. 设计原则

TQP 明确提出四个设计选择：

1. **避免 data-dependent Python control flow**  
   row-level Python 循环会极慢，尤其 GPU 场景；应尽量转成 tensor mask、gather、scatter、where。
2. **输入关系数据使用 tensor-based columnar format**  
   每个列变成 tensor。字符串和日期类型是难点，因为 TCR 通常不原生支持 SQL string/date 语义。
3. **坚持使用 TCR API，不扩展自定义 tensor op**  
   这样才能保持跨硬件可移植性和低工程成本。
4. **可扩展前端和目标格式**  
   前端可接不同 query parser/optimizer；目标可接 PyTorch、ONNX、TorchScript、TVM 等。

## 6. 编译架构

TQP 编译阶段分四层：

1. **Parsing Layer**：外部数据库前端生成 physical plan，再转换成 TQP IR graph。
2. **Canonicalization and Optimization Layer**：做 IR-to-IR 变换。
3. **Planning Layer**：把 IR operator 映射到 tensor program 实现。
4. **Execution Layer**：生成 executor，按拓扑顺序调用 tensor programs，连接输入/输出 tensor，并跟踪引用用于释放不再使用的 tensor。

IR 是 graph-based，变量不可变且带有唯一 id，这有利于调试、属性附着和 runtime garbage collection。

## 7. 执行模型

Executor 负责：

- 输入数据转 tensor format；
- 数据移入/移出 device memory；
- 在选定 device 上调度 operators；
- 顺序执行 operator plan；
- 利用 TCR 提供的 tensor-level intra-operator parallelism。

论文也指出 TQP 当时主要依赖 intra-operator parallelism，inter-operator parallelism 与 data-parallel 策略仍是探索方向。

## 8. 关系算子实现细节

### 8.1 Expression

表达式树后序遍历：叶子节点映射到列 tensor 或常量；内部节点通过字典映射到 tensor op，例如 `*` → `torch.mul`。

### 8.2 Sort-based join

核心步骤：

1. 排序左右 join key；
2. 对 key 建 histogram；
3. 左右 histogram 相乘得到每个 key 的 join 输出大小；
4. prefix sum 得到输出 offset；
5. 用 bucketize 把输出 row 映射回 join bucket；
6. 生成 left/right output row id。

这种算法体现了 TQP 的风格：把 join 的不规则匹配关系转成 histogram、prefix sum、bucketize、gather 等 tensor primitives。

### 8.3 Hash-based join

算法用 hash 值、`bincount`、`scatter`、`masked_select` 等构建/探测 hash table。它在无 hash collision 时路径接近最优；发生 collision 时需要多轮 build/probe，最多循环到最大 hash bucket size。

这也是后续 TQEx 批评和优化的重点：TQP 的 hash join 为处理不规则 bucket 引入了循环和重复访问。

### 8.4 Aggregation

聚合流程：

1. 拼接 group-by columns；
2. 排序 group keys 并重排数据列；
3. 用 `unique_consecutive` 得到 unique groups 和 inverse group id；
4. 对每个 group 计算 aggregate expression。

## 9. 实验与性能洞察

TQP 在 TPC-H 上比较 Spark、DuckDB、BlazingSQL、OmnisciDB 等系统。性能 breakdown 给出几个重要观察：

- 同一 tensor algorithm 在 CPU 和 GPU 上热点不同；
- CPU 上常见瓶颈是 `unique`、`masked_select`、indexing；
- GPU 上排序、`unique`、`nonzero` 等可能占主导；
- `nonzero` 会引入 host/device synchronization；
- 一些简单查询中 GPU memory allocation 时间占比较高；
- hand-optimized tensor plans 能显著改进性能，说明需要 TCR-aware optimizer。

## 10. 对当前仓库的启发

当前仓库与 TQP 的对应关系：

| TQP | 本仓库 |
| --- | --- |
| 外部 DB 前端 physical plan | DuckDB JSON physical plan。 |
| TQP IR graph | `TQPOperatorGraph`。 |
| operator plan / executor | `PyTorchGraphExecutor` + physical interpreter。 |
| tensor program | `backend/physical*.py`, `operators.py`, graph nodes。 |
| PyTorch eager / device runtime | 当前 CPU/CUDA backend。 |

应该继续补齐：

- 把 physical interpreter 中的复杂 shape 拆成更显式的通用 graph nodes；
- 建立 TCR-aware optimizer，而不是只追求查询能跑；
- 增加 operator-level profiling，识别 `sort/unique/nonzero/scatter` 等热点；
- 进一步探索 TorchScript/`torch.compile`/vendor compiler 对 scan-filter-project 和 map-reduce 的 fusion。

## 11. 局限与注意点

- TQP 为保持可移植性，不扩展自定义 tensor op；但这可能牺牲某些 DB 专用场景性能。
- 字符串、日期、decimal、null 语义不是 TCR 的强项，需要 DB 层补齐。
- TQP 的 TPC-H 性能结果依赖当时 PyTorch、CUDA、GPU 数据库版本；不能直接当作当前硬件性能承诺。
- 多 GPU、out-of-memory、数据移动服务在 TQP 中还不是成熟系统能力。
