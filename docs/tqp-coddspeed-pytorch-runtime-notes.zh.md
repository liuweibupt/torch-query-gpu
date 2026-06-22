# TQP / CoddSpeed 利用 PyTorch/TCR 的能力与 kernel 并行机制补充说明

本文补充 [`docs/ai-tensor-db-on-domestic-gpu.zh.md`](ai-tensor-db-on-domestic-gpu.zh.md) 中的两个问题：

1. TQP、TQP++、CoddSpeed 这条线到底利用了 PyTorch / Tensor Computation Runtime 的哪些能力？
2. PyTorch 里面的 kernel 并行是怎么发生的？它和数据库执行引擎里的并行有什么关系？

> 来源边界：TQP 结论来自本仓库已下载 PDF；TQP++ 与 CoddSpeed 当前只有公开 Microsoft Research 页面和仓库 source notes，全文 PDF 未下载到。因此本文对 TQP++ / CoddSpeed 只写公开摘要可支持的系统级结论，不断言其内部具体 PyTorch API 调用。

## 1. 先区分 PyTorch、TCR 与论文里的用法

TQP 论文使用更宽泛的概念：**Tensor Computation Runtime, TCR**。PyTorch 是 TCR 的一个实现，另外还包括 TensorFlow、ONNX Runtime、TVM 等运行时或编译器。

因此论文里“利用 PyTorch”的本质不是“调用某几个 PyTorch 函数”，而是：

```text
SQL / relational operator
  → tensor program
  → TCR API, e.g. PyTorch tensor API
  → TCR runtime / compiler / kernel library
  → CPU / GPU / accelerator
```

这对国产卡很关键：如果国产卡生态优先兼容 PyTorch 或提供 PyTorch-like frontend，就可以把 DB 的执行逻辑放在 tensor program 层，而不是直接面向每个硬件后端手写 SQL kernels。

## 2. TQP 明确利用了 PyTorch/TCR 哪些能力

TQP 利用的能力可以分为八类。

### 2.1 Tensor columnar representation

TQP 把关系表转成 tensor-based columnar format：

```text
table column → tensor
table        → collection of tensors
```

这和数据库列式执行天然接近。DB 里的 selection、projection、join、group-by 都可以看作对一批列 tensor 的变换。

### 2.2 Tensor primitive API

TQP 论文列出了一组关系算子会用到的 tensor operation 类别：

| Tensor op 类别 | 代表能力 | 对应关系算子用途 |
| --- | --- | --- |
| Creation | `from_numpy`, `zeros`, `ones`, `empty`, `fill`, `arange` | 构造列、mask、row id、group state。 |
| Indexing / slicing | indexing, `index_select`, mask select, range/narrow | selection、projection、join 后 gather。 |
| Reorganization | `reshape`, `view`, `squeeze`, `gather`, `scatter`, `sort` | 排序、join row 重排、group key 编码。 |
| Comparison | `eq`, `lt`, `gt`, `le`, `ge`, `isnan` | SQL predicate。 |
| Conditional | `where` | `CASE WHEN`、条件表达式。 |
| Search | `bucketize` / binary-search style op | range lookup、join/search。 |
| Arithmetic / logical | `add`, `mul`, `div`, `sub`, logical ops | projection expression、predicate composition。 |
| Join / combine | `concat`, `stack` | 多 key、多列组合。 |
| Reduction | `sum`, `max`, `min`, `mean`, `all`, `any` | aggregate。 |
| Grouped reduction | `scatter_add`, `scatter_min`, `scatter_max`, `scatter_mean` | group-by aggregate。 |
| Histogram / distinct | `bincount`, `histc`, `nonzero`, `unique`, `unique_consecutive` | count、group id、distinct。 |

这说明 TQP 不是只复用矩阵乘法这类 AI 热点，而是大量依赖 sort/search/gather/scatter/reduction 这类更像数据库的 tensor primitives。

### 2.3 Device abstraction 与硬件可移植性

TQP 的一个目标是“free-ride” AI 社区对硬件后端的投入。PyTorch/TCR 隐藏了低层硬件细节，让 tensor program 可以面向 CPU、GPU、ASIC 或其他 accelerator。

对国产卡来说，这意味着系统可以优先保证：

```text
TQP lowering 生成的 tensor primitives
  在国产卡 PyTorch backend 上有可用且高性能的实现
```

而不是先为每种 SQL 算子手写国产卡 kernel。

### 2.4 数据转换与 host/device memory movement

TQP execution 部分明确包含：

1. 把输入数据转换成 tensor format；
2. 把数据移动到/移出 device memory；
3. 在选择的 device 上调度 operators。

这属于 runtime 能力，不只是算子能力。当前 demo 也有类似环节：DuckDB/NumPy 列通过 `torch.as_tensor(..., device=device)` 变成 CPU/CUDA tensor。

### 2.5 Tensor-level intra-operator parallelism

TQP 明确利用 TCR 提供的 tensor-level intra-operator parallelism。也就是说，一个 DB operator 被编译成一个或多个 tensor op，真正的并行发生在 tensor op 的 backend kernel 内部。

```text
group-by aggregate
  → scatter_add / bincount / reduction
  → TCR kernel 内部并行
```

TQP 原始设计并不是靠 Python 多线程去并行每一行数据。

### 2.6 Eager execution

TQP 支持 PyTorch interpreted/eager mode：executor 按 operator plan 顺序调用 tensor program。

这和当前 demo 类似：

```text
Python physical executor
  → torch op
  → PyTorch dispatcher
  → CPU/CUDA kernel
```

好处是实现简单、便于调试；代价是 Python overhead、kernel launch overhead 和中间 tensor materialization 更明显。

### 2.7 TorchScript / ONNX / TVM lowering

TQP 不只支持 PyTorch eager。论文描述 executor 可以 lowering 到：

- TorchScript；
- ONNX；
- TVM machine-level code。

这说明 TQP 试图利用的不只是 PyTorch 算子库，还包括 tensor program 的编译/导出生态。不是所有查询都能编译到所有目标，因为不同目标支持的 tensor op 覆盖不同。

### 2.8 DataLoader 作为 out-of-memory 方向

TQP 论文提到正在探索利用 PyTorch DataLoader 支持 out-of-memory computation。这一点应理解为未来方向，而不是 TQP 主路径的核心依赖。

DB 里的 out-of-memory 不能简单等同训练 batch：join、group-by、sort 都有跨 batch 状态。因此 DataLoader 可以借鉴 prefetch/batch movement 思路，但还需要 DB-specific pipeline、spill 和 state management。

## 3. TQP 特别避免什么

TQP 强调避免 **data-dependent Python control flow**。

不推荐：

```python
for row in rows:
    if row.quantity < 24:
        output.append(row)
```

推荐：

```python
mask = quantity < 24
output = quantity[mask]
```

原因是 Python 数据相关循环会吞掉 GPU 并行收益，也很难被 tensor compiler 优化。schema-level 的循环可以接受，例如遍历列名生成多个 expression；row-level 或 data-dependent 分支应尽量变成 mask、gather、scatter、where、reduce。

## 4. TQP++ 和 CoddSpeed 的可确认信息

### 4.1 TQP++

本仓库记录的 TQP++ 公开摘要强调：

- ML-compiler-native analytical query processor；
- Antares compilation framework；
- tiered GPU resource scheduling；
- map-reduce-oriented fusion；
- multi-gated execution graph，根据运行时数据选择 operator algorithms。

这说明 TQP++ 相比原始 TQP，更强调 compiler、fusion、resource scheduling 和 runtime algorithm gating。它不像当前 demo 这样主要依赖 PyTorch eager op，而是把 analytical query processing 更深地放进 ML compiler/runtime 体系。

### 4.2 CoddSpeed

CoddSpeed 公开摘要与本仓库 source note 可确认：

- 它是 Microsoft Fabric 中的硬件加速查询系统；
- 包含 derived from TQP 的 GPU execution engine；
- 关注 GPU、FPGA、ASIC、NVLink、InfiniBand 等 accelerator / network；
- data movement 是 accelerated analytics 的 first-class concern。

但当前无法确认它内部是否仍直接使用 PyTorch eager、TorchScript、ONNX、TVM 或替换为内部 runtime/compiler。因此准确说法是：

```text
CoddSpeed 继承 TQP 的 query-to-accelerator execution 思路，
但公开资料不足以断言它具体依赖 PyTorch 的哪些 API。
```

对国产卡分享来说，CoddSpeed 的启发更偏系统层：不要只讲单个 kernel，要把数据移动、多硬件、资源调度、平台集成放进同一张图。

## 5. PyTorch kernel 并行是怎么发生的

PyTorch 的并行分 CPU 与 GPU 两类看。

### 5.1 CPU：intra-op 与 inter-op 并行

在 CPU 上，一个 PyTorch op 通常进入 C++/ATen backend。许多 op 内部会用线程池、OpenMP、TBB、MKL/oneDNN 等做 **intra-op parallelism**：

```text
torch.sum(x)
  → ATen CPU kernel
  → kernel 内部把 tensor 分片
  → 多 CPU 线程并行处理
```

还有一类是 **inter-op parallelism**：多个独立 ops 或 graph tasks 之间并行执行。它在 TorchScript/graph execution 场景更典型。

可调参数包括：

- `torch.set_num_threads(n)`：影响 intra-op 线程数；
- `torch.set_num_interop_threads(n)`：影响 inter-op 线程数；
- `OMP_NUM_THREADS`、`MKL_NUM_THREADS` 等环境变量。

对 DB 的含义：CPU 上的 tensor DB prototype 不需要自己为每个 element 写 Python 线程；只要把工作表达成大 tensor op，PyTorch CPU backend 就可能在 op 内部并行。

### 5.2 GPU：kernel launch 与 kernel 内部并行

在 GPU 上，Python 调用一个 CUDA tensor op 时，通常发生：

```text
Python torch op
  → PyTorch dispatcher
  → CUDA backend implementation
  → enqueue CUDA kernel 到当前 stream
  → GPU kernel 内部以 grid/block/thread 并行执行
```

并行主要在 kernel 内部：

- elementwise op：大量元素映射到大量 GPU threads；
- reduction：多级并行 reduce；
- sort/top-k/search：调用专门的并行算法；
- scatter/gather：大量索引访问并行执行，但可能受内存访问模式影响。

PyTorch 的 CUDA op 通常是异步 enqueue。Python 返回时 kernel 可能还没执行完，所以计时需要：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# torch ops
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

当前仓库 benchmark/runner 就使用了 CUDA event 和 `torch.cuda.synchronize()` 来避免把异步执行误计为已经完成。

### 5.3 Streams：不同 kernel 能否并发

CUDA stream 是有序队列。默认情况下，同一 stream 上的 kernels 按顺序执行：

```text
stream0: kernel A → kernel B → kernel C
```

如果使用多个 stream，并且 kernels 之间没有依赖，硬件资源允许时可以 overlap：

```text
stream0: scan/filter kernel
stream1: H2D copy or independent aggregate kernel
```

PyTorch 支持 `torch.cuda.Stream` 和 `torch.cuda.stream(...)`。但要正确使用 streams，需要显式处理依赖、tensor lifetime 和 allocator stream semantics。当前 demo 基本没有把多 stream 当成查询调度器；它主要依赖默认 stream 的顺序语义。

### 5.4 DataLoader 并行不是 kernel 并行

DataLoader 多 worker 是 CPU 侧数据读取/预处理/组 batch 的并行：

```text
worker processes/threads
  → load/decode/collate batches
  → main process consumes batch
  → tensor copy to device
```

它可以减少 GPU 等数据，但不等于 GPU kernel 内部并行。DB 可以借鉴它做 scan prefetch 和 host→device overlap，但 join/group/sort 的全局状态需要 DB 自己管理。

### 5.5 Graph/compiler fusion 与 kernel 数量

PyTorch eager 模式下，多个 tensor op 往往会产生多个 kernel launch：

```python
z = (a + b) * c
mask = z > 0
out = torch.where(mask, z, 0)
```

可能对应多个 kernel 和中间 tensor。`torch.compile`、TorchInductor 或厂商 graph compiler 可能把一段 tensor program 捕获成图，并融合 elementwise/reduction 周边操作：

```text
多个 eager ops
  → graph capture
  → fusion/codegen
  → 更少 kernel + 更少中间 tensor
```

对 DB 来说，这对应 TQP++ 提到的 fusion 方向：scan-filter-project、map-reduce aggregation、部分 predicate/expression 可以融合，减少 launch overhead 和 intermediate materialization。

## 6. 这和当前 demo 的差距

| 能力 | TQP 论文 | TQP++ / CoddSpeed 公开信息 | 当前 demo |
| --- | --- | --- | --- |
| tensor columnar format | 明确使用 | 继承 TQP 思路 | 使用。 |
| PyTorch/TCR primitive ops | 明确使用 | TQP++ 更偏 compiler；CoddSpeed 不确认具体 API | 使用 eager torch ops。 |
| device abstraction | 明确使用 | CoddSpeed 强调多 accelerator | 使用 CPU/CUDA device。 |
| data movement 管理 | executor 管理 tensor conversion 与 device movement | CoddSpeed 把 data movement 提升为 first-class concern | 基础 H2D/tensor conversion，有 resident cache，但不完整。 |
| intra-operator parallelism | 利用 TCR tensor-level 并行 | GPU engine 必然依赖 accelerator parallelism | 依赖 PyTorch kernels 内部并行。 |
| TorchScript/ONNX/TVM | 明确支持 | 不确认 | 暂无。 |
| DataLoader/OOM | 论文提到探索 | 不确认 | 暂无。 |
| 多 stream / pipeline scheduling | 不是 TQP 主结论 | CoddSpeed 更关注系统调度 | 暂无。 |
| graph/compiler fusion | TQP 有编译目标；TQP++ 更强调 | TQP++ 明确强调 map-reduce fusion | 只有 Q1 局部 fused primitive。 |

## 7. 对国产卡 DB 路线的含义

如果把这条线迁移到国产卡，评估重点应包括：

1. **tensor primitive 覆盖**：`unique`、`argsort`、`searchsorted`、`bincount`、`scatter_reduce`、`topk` 等是否支持 device backend；
2. **kernel 内部并行质量**：这些 op 是否真的有高性能并行 kernel，而不是回退 CPU 或串行实现；
3. **动态图与变长输出**：filter/join 后 cardinality 数据相关，compiler/runtime 是否支持；
4. **graph fusion 能力**：能否融合 scan-filter-project / map-reduce；
5. **数据移动能力**：host/device copy、pinned memory、异步 copy、device residency；
6. **显存 allocator 和 profiling**：能否解释 peak memory、fragmentation、kernel time 和 copy time；
7. **fallback 策略**：某个 torch op 不可用或太慢时，DB 层需要替代算法、Triton/custom kernel 或 CPU fallback 的显式策略。

最重要的判断是：

```text
TQP 证明“关系代数可以表达为 tensor program”；
PyTorch/TCR 负责把 tensor program 映射到硬件并提供 kernel 内部并行；
CoddSpeed/TQP++ 提醒我们，生产系统还必须解决 fusion、调度和数据移动。
```

## 8. 参考入口

- TQP PDF：[`docs/papers/tqp-query-processing-on-tensor-computation-runtimes.pdf`](papers/tqp-query-processing-on-tensor-computation-runtimes.pdf)
- TQP++ source note：[`docs/papers/tqp-plusplus-msr-page.md`](papers/tqp-plusplus-msr-page.md)
- CoddSpeed source note：[`docs/papers/coddspeed-msr-page.md`](papers/coddspeed-msr-page.md)
- PyTorch CUDA semantics：<https://docs.pytorch.org/docs/stable/notes/cuda.html>
- PyTorch CPU threading / TorchScript inference note：<https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html>
- PyTorch DataLoader：<https://docs.pytorch.org/docs/stable/data.html>
- `torch.compile`：<https://docs.pytorch.org/docs/stable/generated/torch.compile.html>
