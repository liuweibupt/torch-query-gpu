# TQP 路线算子与优化 Roadmap（中文）

本文档是 [`docs/operator-roadmap.md`](operator-roadmap.md) 的中文执行版，跟踪从当前正确性优先原型走向 TQP → TQEx → TQP++ → CoddSpeed 路线所需的算子、压缩数据执行与优化工作。

## 1. 论文资料状态

| 来源 | 本地状态 | 抽取可信度 |
| --- | --- | --- |
| TQP, *Query Processing on Tensor Computation Runtimes* | `docs/papers/tqp-query-processing-on-tensor-computation-runtimes.pdf` | 已抽取全文；PDF 没有单独 appendix。 |
| TQEx, *Tensor-based Query Engine Enhanced by Bridging the Gap* | 当前环境 ACM PDF 被 403/Cloudflare 阻挡；DOI/Crossref 元数据和 abstract 可访问。 | 仅 abstract-derived，待补全文/附录。 |
| TQP++, *Bridging ML Compilers and Analytical Query Processing on GPUs* | Microsoft Research 页面可访问；preprint endpoint 被阻挡。 | 仅页面/abstract-derived，待补全文/附录。 |
| CoddSpeed, *Hardware Accelerated Query Processing in Microsoft Fabric* | Microsoft Research/DOI metadata 可访问；ACM PDF 被阻挡。 | 仅页面/abstract-derived，待补全文/附录。 |
| *GPU Acceleration of SQL Analytics on Compressed Data* | `docs/papers/gpu-acceleration-sql-analytics-compressed-data.pdf` | 已抽取 arXiv v2 全文和附录。 |

标记规则：

- **verified**：来自本地可读全文/附录。
- **abstract-derived**：只来自 abstract/公开页面，不能视为完整清单。

## 2. 当前仓库基线

- 默认链路：原始 SQL → DuckDB/Sirius-like planner admission → `TQPPlan` → PyTorch CPU/CUDA 算子。
- 实验链路：原始 SQL → DuckDB native Substrait JSON → `TQPPlan` → PyTorch；无伪造 JSON，无自动 fallback。
- TPC-H：默认 Sirius-like 路径下 Q1-Q22 均先 lowering 到 `TQPOperatorGraph`；Q1/Q12/Q14/Q19 已由 DuckDB physical-plan interpreter 执行；Q6 仍由直接 graph primitive 执行；剩余复杂 Q2-Q22 仍由通用 Join/Subquery/CTE/Aggregate graph nodes 组合的 graph recipes 执行。
- Generic SQL：单表 projection/filter/aggregate/order/limit 子集。
- 压缩执行：已有 Plain/RLE/Index mask 原型，Q6 可通过 `--compressed-masks` 显式开启。

## 3. TQP verified 算子清单

### Tensor operation families

- Creation：`from_numpy`, `zeros`, `ones`, `empty`, `fill`, `arange`, `zeros_like`, `ones_like`。
- Indexing/slicing：tensor indexing, `index_select`, `masked_select`, `narrow`。
- Reorganization：`reshape`, `view`, `squeeze`, `gather`, `scatter`, `sort`。
- Comparison：`eq`, `lt`, `gt`, `le`, `ge`, `isnan`, `where`, `bucketize`。
- Arithmetic/logical：`add`, `mul`, `div`, `sub`, `fmod`, `remainder`, `logical_and`, `logical_or`, negation, shift operations。
- Joining/stacking：`cat`, `stack`。
- Reductions：`sum`, `max`, `min`, `mean`, `scatter_add`, `scatter_min`, `scatter_max`, `scatter_mean`, `all`, `any`, `bincount`, `histc`, `nonzero`, `unique`, `unique_consecutive`。

### Relational operators / SQL features

- Selection/filter：bitmap masks 与 index-based selection。
- Projection：表达式树 post-order DFS。
- Sort。
- Group-by aggregation：包含 sort-based grouping。
- Natural joins：sort-based 与 hash-based 两类算法。
- Non-equi join、left outer join、left semi join、left anti join。
- Comparison/arithmetic/date expressions。
- `IN`, `CASE`, `LIKE`。
- Aggregates：`SUM`, `AVG`, `MIN`, `MAX`, `COUNT`，含 distinct/non-distinct 变体。
- NULL handling。
- Scalar/nested/correlated subqueries。
- Prediction UDF / ML model operators。

## 4. TQP 算法 TODO

- [x] Plain columnar tensor table representation。
- [x] 当前 TPC-H executor 的 bitmap-style filter masks。
- [x] Generic SQL boolean filter tree：`AND` / `OR` / `NOT`。
- [x] Generic bitmap selection：comparison / `IN` / `LIKE`。
- [x] DuckDB physical projection expression 子集：column refs、`#N`、arithmetic、comparison、`CASE`、`prefix`/`contains`/`suffix`、internal compress/decompress wrappers。
- [x] Generic stable multi-key `ORDER BY`，支持 `ASC` / `DESC`。
- [ ] Generic sort-based equi-join：sort、histograms、prefix sums、`bucketize`、quotient/remainder 输出索引生成。
- [ ] Generic hash equi-join：hash buckets、scatter、probe、collision iteration、duplicate accumulation。
- [ ] Join variants：non-equi、left outer、left semi、left anti。
- [ ] Sort-based group-by aggregation：concatenated keys、sort、`unique_consecutive`、inverse ids、scatter reductions。
- [x] 当前 query templates 的 sum/count group reductions。
- [x] 可复用 min/max/mean group reductions。
- [ ] Count-distinct aggregation。
- [ ] Scalar/nested/correlated subquery lowering。
- [ ] NULL-aware boolean and aggregate semantics。
- [ ] ML prediction operator boundary。

## 5. TQP 优化 TODO

- [x] Generic grouped aggregate 使用 tensor group id + grouped reductions，避免 Python row-group loops。
- [x] DuckDB physical inner equi-join 移除 Python key row loop，使用 tensor `searchsorted` 产生 row-index pairs。
- [x] Physical join sorted/unique build-side fast path：跳过 `argsort` 与重复展开。
- [ ] 移除剩余 hot path 中的数据相关 Python row loops；保留 schema/operator 级循环。
- [x] 保持 columnar late materialization：physical join 先产生 row-index pairs，再物化 payload columns。
- [ ] row-level work 优先用 tensor ops，不用 Python control flow。
- [ ] 编译执行路径：TorchScript / TVM / `torch.compile` / Antares / codegen / CSE / fusion / Python dependency removal。
- [x] `LookupIndex`：复用 pre-sorted dimension-key lookup probes。
- [x] 第一批 sorted/unique 感知：已排序唯一 build-side join key 跳过冗余 sort。
- [ ] optimizer 感知更多 sorted/unique columns，避免冗余 `sort` / `unique` / `unique_consecutive`。
- [ ] 基于 collision degree、key cardinality、device 选择 hash join 或 sort join。
- [ ] 跟踪 backend bottlenecks：`unique`、indexing、`masked_select`、`scatter_add`、`nonzero` 同步、sort 成本。
- [ ] 分离 pipeline/capture 中的数据移动与查询执行。
- [ ] 缓存 frontend compilation 和 tensor operator plans。
- [x] Table-aware static dictionary encoding for TPC-H low-cardinality strings，减少大列 `numpy.unique`。
- [x] Singleton / literal-list membership mask：单元素走 equality，同列 literal `OR` 折叠为一次 membership。
- [x] `PhysicalTable.filter/gather` 对共享 alias value 去重，避免 `col` 与 `table.col` 重复 selection。
- [ ] 显式 operator graph 完成后再加入 inter-operator parallelism 和 distributed/data-parallel execution。

## 6. 压缩数据执行 TODO（verified）

### Encodings

- [x] Plain tensor columns。
- [x] Dictionary-encoded string columns。
- [ ] RLE columns：value、inclusive start、inclusive end tensors，按 start/end 排序且 range 不重叠。
- [x] Index mask positions：排序且唯一的位置 tensor。
- [ ] Plain + Index composite encoding：outlier separation 与 bit-width reduction。
- [ ] RLE + Index composite encoding：同时处理连续 runs 和 isolated impure segments。
- [ ] Centered bit-width reduction for numeric ranges。
- [ ] Storage/catalog encoding metadata，后端可选择 plain/RLE/index/composite execution。
- [ ] Snappy/zstd/LZ4/gzip 等 heavyweight codec 暂不进入主线，先完成 lightweight encoded execution。

### 附录/论文 primitives

- [x] `plain_to_rle` for boolean masks。
- [x] `rle_to_index`。
- [x] `range_intersect` for RLE/RLE intersection。
- [x] `idx_in_rle`。
- [x] `rle_contain_idx`。
- [x] `idx_in_idx`。
- [x] `range_union` for RLE/RLE union。
- [x] `merge_sorted_idx`。
- [ ] `compact_rle`。
- [ ] `compact_rle_index`。
- [x] `complement_rle` from Appendix A.1。
- [x] `complement_index` from Appendix A.1。
- [x] `rle_to_plain`。
- [ ] `plain_to_rle_index`。
- [ ] `plain_to_plain_index`。
- [ ] `range_arange` helper，用于 range algorithms 和 RLE join-index expansion。

### Encoded mask logical operators

- [x] AND Plain/Plain。
- [x] AND RLE/RLE：`range_intersect`。
- [x] AND RLE/Plain：正确性优先显式转换。
- [x] AND RLE/Index primitives：`idx_in_rle` / `rle_contain_idx`。
- [x] AND Index/Index：`idx_in_idx`。
- [x] OR RLE/RLE：`range_union`。
- [x] OR Index/Index：`merge_sorted_idx`。
- [x] OR mixed RLE/Plain/RLE/Index：正确性优先转换。
- [x] NOT Plain。
- [x] NOT RLE：`complement_rle`。
- [x] NOT Index：`complement_index`，返回 RLE。
- [ ] Composite mask rewrites：用 De Morgan 展开 RLE+Index / Plain+Index。

### Alignment / arithmetic / comparison / selection

- [ ] heterogeneous encodings 的通用 alignment operator。
- [ ] RLE/RLE alignment：相交 positional ranges，重建 aligned values，不展开到行。
- [ ] Plain/RLE、Plain/Index、RLE/Index、composite alignment。
- [ ] compressed column scalar arithmetic/comparison：无需 positional alignment 时只操作 value tensors。
- [ ] Binary arithmetic：`+`, `-`, `*`, `/`, modulo/remainder。
- [ ] Binary comparison：`=`, `!=`, `<`, `<=`, `>`, `>=`。
- [x] Q6 encoded mask → row indices → Plain revenue columns selection。
- [ ] General selection：encoded `MaskColumn` 与目标 `DataColumn` alignment，并保留输出 encoding 决策。
- [ ] 按论文表格保留 output encoding，不静默 materialize plain tensors。

### Group-by / aggregation on encoded data

- [ ] aligned group-by columns 上的 grouping phase。
- [ ] 使用 scatter over inverse ids 的 aggregation phase。
- [ ] RLE `COUNT`：sum run lengths。
- [ ] RLE `SUM`：sum value × run length。
- [ ] RLE `MIN` / `MAX`：只 reduce value tensors。
- [ ] `AVG`：`SUM / COUNT` 后处理。
- [ ] `STD` / `VAR`：sum of squared values + sum/count 后处理。
- [ ] Appendix A.2 group-by walkthrough 转回归测试。
- [ ] RLE group-by columns 已携带 filtered ranges 时，避免重复 filter aggregate columns。

### Join operators on encoded data

- [ ] Plain/RLE/Index join columns 复用 GPU hash join over value tensors。
- [ ] 产生 Join Index tensors，而不是立刻 materialize payload columns。
- [ ] RLE join columns：hash join run values，再映射 run ids 到 row ranges。
- [ ] Index join columns：hash join encoded values，再恢复 row positions。
- [ ] RLE/RLE many-to-many join-index expansion：run-length products。
- [ ] Plain/RLE 与 RLE/Index join-index encodings from Table 6。
- [ ] Apply Join Index to payload columns，支持 unsorted / duplicate join indices。
- [ ] 对 unsorted RLE/Index join indices 的 sorted side 使用 bucketize。
- [ ] duplicate-free side 已知时优化 one-to-one / one-to-many joins。
- [ ] semi-joins 与 PK/FK joins 作为一等 join-index patterns。
- [ ] Appendix A.3 join-index example 转回归测试。

### Appendix D compression-aware optimizer rules

- [ ] 优先对 RLE columns 应用 predicates，再处理 Plain columns。
- [ ] 同一个 RLE column 上多个 predicates 合成为 value tensor predicate，再一次性应用 start/end tensors。
- [ ] 优先 joins / semi-joins involving RLE columns，避免 Plain operation 过早破坏 runs。
- [ ] filter → group-by → aggregate 中，如果 RLE group-by columns 已携带 filtered ranges，避免冗余 filter。
- [ ] General selection pushdown for Plain and compressed execution。
- [ ] 显式 NULL support；不要长期依赖论文实验中的 no-NULL shortcut。

### Layout / ordering

- [ ] Encoding choice heuristics：small columns Plain；压缩率超过阈值用 RLE；大量单元素 runs + longer-run compression 用 RLE+Index；outlier-driven bit-width reduction 用 Plain+Index；否则 Plain/centered Plain。
- [ ] TPC-H Q1/Q2/Q6/Q11/Q14/Q15/Q17/Q19 的 query-specific ordering experiments。
- [ ] V-order 或 cardinality-ordering 作为可选 storage layout preparation，不作为 SQL rewrite。
- [ ] validation output 中跟踪 compression ratio、run count、average run length、HBM footprint。

## 7. TQEx / TQP++ / CoddSpeed abstract-derived TODO

### TQEx

- [ ] 重新下载或通过授权方式获取 TQEx PDF，抽取所有 operators、appendices、implementation details。
- [ ] 建模 irregular SQL workloads 与 uniform tensor operations 的 gap。
- [ ] 增加 variable-length data 的 storage/computation strategy。
- [ ] 根据全文 revisiting tensor join 和 aggregate algorithms。
- [ ] 扩展 multi-XPU / multi-device processing。

### TQP++

- [ ] 重新下载或获取 TQP++ preprint，抽取完整算法与附录。
- [ ] 定义 ML-compiler-native operator graph，支持 eager PyTorch 之外的 lowering。
- [ ] 增加 Antares-compatible lowering experiments。
- [ ] 增加 tiered GPU resource scheduling。
- [ ] 增加 map-reduce-oriented fusion，减少 intermediate materialization。
- [ ] 增加 multi-gated execution graph，根据 runtime data、encoding、cardinality 选择算法。

### CoddSpeed

- [ ] 重新下载或获取 CoddSpeed paper，抽取完整系统细节。
- [ ] 保持 GPU engine 与 SQL admission 解耦，且 hardware-independent。
- [ ] 把 data movement 作为一等 plan property。
- [ ] 建模 accelerator/interconnect placement：GPUs、FPGAs、ASICs、NVLink、InfiniBand。
- [ ] plan annotations：HBM residency、CPU↔GPU transfer、GPU↔GPU transfer、remote/distributed movement。
- [ ] execution metrics 区分 compile、transfer、kernel、allocation、materialization 时间。

## 8. 实施批次

### Batch 1：基础可复用 primitives — completed

- [x] 文档化 source status、verified operator inventory、完整 TODO。
- [x] 下载并跟踪 compressed SQL analytics arXiv PDF。
- [x] Plain mask helpers：`logical_and_all`, `logical_or_all`, `gather_by_mask`。
- [x] Plain grouped reductions：`grouped_min`, `grouped_max`, `grouped_mean`。
- [x] Plain top-k helper：`topk_indices`。
- [x] RLE mask container 和 primitives：`plain_to_rle`, `rle_to_index`, `range_intersect`, `range_union`, `complement_rle`。

### Batch 2：Generic SQL expression / aggregate 扩展 — in progress

- [x] Boolean filter expression tree：comparison / `IN` / `LIKE` + `AND` / `OR` / `NOT`。
- [ ] Full arithmetic/date/string expression tree beyond filters。
- [x] Generic `MIN`, `MAX`, `AVG`, `COUNT(col)`。
- [ ] Basic `HAVING`。
- [x] Generic `IN`, `LIKE`, `OR`, `NOT`。
- [ ] Generic `CASE`。
- [x] Multi-key order-by 和 DESC/ASC。
- [ ] Tensor top-k integration for `ORDER BY ... LIMIT`。

### Batch 3：Generic joins / subquery lowering

- [x] DuckDB physical `HASH_JOIN` correctness-first generic inner equi-join。
- [ ] PK/FK lookup join fast path 作为优化版 generic join。
- [x] Physical-plan inner equi-join 先产生 tensor row-index pairs，再物化 payload columns。
- [x] Sorted unique build-side physical join fast path。
- [ ] GPU hash equi-join fast path：buckets / probe / collision handling。
- [ ] Semi/anti joins。
- [ ] TPC-H 形状所需的 mark/delimiter-style subquery patterns。

### Batch 4：Compressed storage / mask execution

- [ ] Encoding metadata 与 RLE/Index column storage。
- [x] RLE/Index/Index encoded logical mask primitives。
- [x] Plain/RLE/Index correctness-first encoded logical mask dispatch。
- [x] Q6 `--compressed-masks` encoded selection path。
- [ ] Full compressed column alignment 与 output-encoding decisions。
- [ ] General encoded selection 与 predicate pushdown。

### Batch 5：Compressed aggregate / join execution

- [ ] RLE-aware aggregation。
- [ ] Compressed Join Index generation and application。
- [ ] Compression-aware join ordering 与 filter/group-by optimizations。

### Batch 6：Compiler / fusion / scheduling

- [x] 第一版 `TQPPlan.operator_graph` 与 DuckDB JSON physical plan lowering。
- [x] 将 Q2-Q22 兼容执行器拆成由通用 Join/Subquery/CTE/Aggregate nodes 组合的 graph recipes。
- [x] DuckDB physical-plan interpreter v1：generic joins/aggregates 与 TPC-H Q1/Q12/Q14/Q19。
- [ ] 继续用 physical-plan interpreter 替换剩余 query-id recipes，覆盖 delimiter/mark/nested-loop/subquery/CTE nodes。
- [ ] projection/filter/aggregate/map-reduce chains 的 fusion passes。
- [ ] Device/data-movement scheduler and metrics。
- [ ] `torch.compile` / Antares / alternative compiler experiments。
