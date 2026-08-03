# Scan 分块与取数优化设计

本文记录当前 TQP 原型在 scan / chunk / partitionable execution 上的设计取舍，以及本轮按照成熟数据库实践落地的优化。

## 1. 调研结论：不是纯 push，也不是 row-at-a-time Volcano

成熟分析型数据库通常采用 **vectorized pipeline**：scan/source 产生有界 batch，filter/project 等局部算子逐 batch 流水执行；join build、global aggregate、sort、window 等算子作为 pipeline breaker 建立全局状态或做 final merge。

| 方案 | 特点 | 优点 | 缺点 | 当前 TQP 取舍 |
| --- | --- | --- | --- | --- |
| Row-at-a-time Volcano pull | 每次 `next()` 拉一行 | 简单、可组合 | Python/GPU 场景函数调用和调度开销过高 | 不采用行级 Volcano |
| Vectorized Volcano / pull batch | 每次 `next_batch()` 拉一个 `RecordBatch/DataChunk/TensorRecordBatch` | API 简洁、便于测试；符合 DuckDB/Arrow chunk 边界 | 串行 pull 不天然 overlap I/O/H2D/compute | 当前采用，作为稳定执行 ABI |
| Push pipeline | source 主动把 batch push 到下游队列/port | 易做 prefetch、多线程、多 GPU 调度 | Python 原型复杂度高；全局 breaker 需要 backpressure/生命周期管理 | 下一阶段在 scan manager 内部演进 |
| Morsel-driven scheduler | storage-aware split/morsel + worker/GPU task 调度 | 成熟 OLAP 常用；适合 SF100+ | 需要 split provider、内存预算和 pipeline barrier | 作为目标架构 |

结论：**短期外部 ABI 使用 pull-based `BatchOperator.next_batch()`；scan 内部从 OFFSET/LIMIT 改成 Arrow RecordBatch 流；中长期引入 Sirius-style split provider/coalescer/scheduler。**

## 2. Arrow / DuckDB / Sirius 的分块方式

### Arrow

Arrow Dataset `Scanner` 以 `RecordBatch` 为扫描单位，暴露 `batch_size`、`batch_readahead`、`fragment_readahead`、`use_threads` 等参数，并通过 `to_batches()` 输出 batch 流。参考：<https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Scanner.html>。

成熟点：

- `RecordBatch` 是列式、定长边界，适合 tensor 化；
- batch size 是最大行数边界，不要求等长；
- readahead 与 fragment/file/row-group 组合，为异步 I/O 和预取留接口；
- batch 是执行边界，不是 SQL 语义边界；global operator 仍需要 barrier/finalize。

### DuckDB Python Arrow 导出

DuckDB Python 可以把查询结果导出为 Arrow `RecordBatchReader`，逐 batch 读取；当前官方文档推荐 `to_arrow_reader(chunk_size)`，并说明结果可以“一批一批读”。参考：<https://duckdb.org/docs/current/guides/python/export_arrow.html>。

本仓库当前固定依赖 DuckDB 1.2.x，因此实现上使用该版本仍存在的：

```python
con.execute(sql).fetch_record_batch(rows_per_batch=chunk_size)
```

其语义等价于：一次执行 SQL，返回 Arrow batch reader；后续通过 `read_next_batch()` 拉取 batch。这样避免了旧实现每个 chunk 重发：

```sql
SELECT ... FROM lineitem LIMIT N OFFSET K
```

OFFSET/LIMIT 在大表上会造成重复跳过/重复 plan 执行，不适合作为 SF100 scan 分块方案。

### Sirius

本轮对 Sirius 前端/执行文档和源码做了对照阅读（本地快照：`sirius-db/sirius@7b7be9c5802c9de3c29f1f5a22ee3169fb87795c`）。Sirius 的 scan 方案不是简单 OFFSET 分块，而是：

```text
GPU_SCAN operator
  <- split_connector
  <- load_balancing_scan_batch_coalescer
  <- split_provider / cached_databatch_provider
  <- gpu_ingestible(parquet / duckdb-native)
```

关键点：

- scan 是统一的 `sirius_gpu_scan_operator` source；
- source 从 `split_connector` 拉取已经 coalesce / device-placement 后的 split；
- `gpu_ingestible` 负责按格式枚举 split，并在执行侧 materialize 成 GPU table；
- `load_balancing_scan_batch_coalescer` 按目标 batch 大小合并小 split，并为多 GPU 做 placement；
- pipeline 之间通过 barrier 区分 `PIPELINE`、`PARTIAL`、`FULL`。

参考：

- <https://github.com/sirius-db/sirius/blob/main/docs/super-sirius/scan.md>
- <https://github.com/sirius-db/sirius/blob/main/docs/super-sirius/pipeline-execution.md>
- <https://github.com/sirius-db/sirius/blob/main/docs/super-sirius/memory-management.md>

## 3. 当前落地方案

### 3.1 新 scan source

新增：

```python
# tpch_torch/backend/physical_scan.py
fetch_physical_table_stream(
    con,
    table_name,
    fetched_columns,
    order_columns,
    device,
    *,
    chunk_size,
) -> Iterator[PhysicalTable]
```

执行过程：

```text
DuckDB SELECT once
  -> Arrow RecordBatchReader(rows_per_batch=chunk_size)
  -> RecordBatch
  -> TensorRecordBatch-backed PhysicalTable
  -> BatchOperator.next_batch()
```

`ScanBatchOperator._iter_scan_chunks()` 已改为调用该 stream source；因此：

- `ScanChunkConfig` 不再通过 OFFSET/LIMIT 分片；
- `PartitionConfig` 的 Q1/Q6 local aggregate pipeline 也不再通过 OFFSET/LIMIT 分片；
- 每个输出 batch 都保留 `BatchMeta(row_count, chunk_size, chunk_index, source_offset, device)`。

### 3.2 scan-time 编码下推

为了让 scan 不再被 Python object conversion 拖垮，本轮把两类编码下推到 DuckDB SELECT：

| 列类型 | 旧路径 | 新路径 |
| --- | --- | --- |
| `DECIMAL(p,s)` | DuckDB/PyArrow 输出 `Decimal` object，Python 循环转 scaled int64 | SQL 侧 `((col) * 10^s)::bigint AS col`，PyTorch 直接接 int64 |
| TPC-H 静态字典字符串 | Arrow 输出字符串/object，Python 字典编码 | SQL 侧 `CASE col WHEN ... THEN id ELSE -1 END::bigint AS col`，PyTorch 直接接 dictionary id |
| DATE | 已有 `strftime(col, '%Y%m%d')::integer` | 保持 SQL 侧编码为 `YYYYMMDD` int |

关键代码片段：

```python
# tpch_torch/backend/physical_scan.py
def _select_expression(table_name: str, column: str, duckdb_type: str) -> str:
    if column in DATE_COLUMNS_EXTENDED:
        return f"strftime({column}, '%Y%m%d')::integer as {column}"
    decimal_meta = column_meta_from_duckdb_type(duckdb_type)
    if decimal_meta.logical_dtype == LogicalDType.DECIMAL:
        return f"(({column}) * {10 ** int(decimal_meta.scale or 0)})::bigint as {column}"
    dictionary = static_string_dictionary(table_name, column)
    if dictionary is not None:
        return f"{_static_dictionary_case(column, dictionary)} as {column}"
    return column
```

这不是 query-specific Python 脚本，而是 scan operator 的 typed encoding rule：只要 DuckDB catalog 表明列是 DECIMAL，或列属于已知 TPC-H 静态字典域，就在 scan projection 阶段生成更适合 tensor backend 的物理表示。

### 3.3 group-by dictionary dense ids

Q1 的 group key 是两个低基数字典列：`l_returnflag` × `l_linestatus`。旧 aggregate path 用 `torch.unique(..., dim=0)` 在每个 chunk 上找 group key，SF=1 下这一步比 scan 还贵。

本轮新增 dictionary dense group-id fast path：

```text
(l_returnflag_id, l_linestatus_id)
  -> dense_id = l_returnflag_id * card(l_linestatus) + l_linestatus_id
  -> bincount/lookup 得 observed groups + inverse
  -> scatter/index_add 聚合
```

它是通用的 dictionary group-by 优化，不绑定 Q1 SQL 字符串。

### 3.4 scan predicate pushdown 与列裁剪

进一步演进后，batch pipeline 在构造 `ScanBatchOperator` 时会把 DuckDB scan node 的 `Filters` 分成两类：

```text
pushable filter：能被 DuckDB 作为 base-table WHERE 验证通过
residual filter：仍由 PyTorch tensor filter 执行
```

实现位置：

- `tpch_torch/backend/physical_scan_pushdown.py`
  - `plan_scan_filter_pushdown()`
  - `where_clause()`
- `tpch_torch/backend/physical_pipeline.py`
  - `_scan_operator()` 先规划 pushdown，再只为 residual filters 补取 filter columns
- `tpch_torch/backend/physical_scan.py`
  - `fetch_physical_table_stream(..., scan_filters=...)`

关键效果：

```text
Q1 projected columns:
  l_returnflag, l_linestatus, l_quantity, l_extendedprice, l_discount, l_tax

Q1 scan filter:
  l_shipdate <= DATE '1998-09-02'

旧路径：
  fetch projected columns + l_shipdate
  PyTorch filter l_shipdate

新路径：
  DuckDB Arrow reader:
    SELECT projected columns
    FROM lineitem
    WHERE l_shipdate <= DATE '1998-09-02'
  不再传输 l_shipdate
```

这属于 **storage/source-level predicate pushdown**：它优化的是 scan source 的数据产出边界，不是 query-specific Python fallback。不能安全下推的 predicate 会保留为 residual，并继续由 PyTorch filter 算子执行。

## 4. 哪些算子是全局依赖的？

| 算子类别 | 是否能直接 chunk 后 concat | 正确做法 |
| --- | --- | --- |
| Scan / Filter / Projection / Cast / Arithmetic / CASE | 可以 | 每个 chunk 独立执行，保持 batch metadata |
| `SUM` / `COUNT` / `MIN` / `MAX` | 不能 concat 终态，但可分解 | local aggregate + final merge |
| `AVG` | 不能直接平均 partial avg | local 输出 sum/count 或 weighted avg + count，final 合并 |
| Group By Aggregate | 可分解 | 每 chunk local group state，final 按 group key merge |
| Top-K | 可分解 | local top-k + final top-k |
| Hash Join | 依赖 build/probe 全局状态 | 小表/build side 常驻，probe side chunk；或两边 hash partition 后 co-partition join |
| Sort Merge Join | 依赖全局有序性 | local sort + global merge，或 range partition 后 sort/merge |
| Global Sort / ORDER BY | 不能直接 concat | external sort / local sort + k-way merge |
| DISTINCT | 不能 concat | local distinct + final distinct |
| Window | 通常依赖 partition/order frame | 按 window partition/range 分片，处理边界 frame |
| CTE materialization / correlated subquery / mark join | 依赖中间关系或 NULL-aware 状态 | 显式 materialize 或专用全局 state |

因此当前策略保持两条显式入口：

- `ScanChunkConfig`：只允许 scan/filter/project，不支持就抛 `UnsupportedPlanError`；
- `PartitionConfig`：允许单表 aggregate fragment，因为它带 local/final merge。

## 5. Q1 / SF100 目标路径

面向 SF100 Q1，推荐路径是：

```bash
python -m scripts.benchmark_query \
  --db data/tpch_sf100.duckdb \
  --query 1 \
  --device cuda \
  --cold-runs 0 --warmup-runs 1 --hot-runs 3 \
  --partition-table lineitem \
  --partition-chunk-size 4000000
```

执行图：

```text
SQL Q1
  -> DuckDB/Sirius-like physical graph
  -> PartitionConfig(lineitem, chunk_size)
  -> ScanBatchOperator(fetch_physical_table_stream)
       DuckDB Arrow RecordBatchReader
       scan predicate pushdown + filter-only column pruning
       DECIMAL/DATE/static-string scan-time encoding
  -> FilterBatchOperator
  -> ProjectBatchOperator
  -> LocalAggregateBatchOperator
       dictionary dense group-id fast path
  -> TensorFinalMerge
```

当前已经解决的 scan 侧瓶颈：

1. 去掉每 chunk OFFSET/LIMIT 重扫；
2. 去掉 DECIMAL Python object 循环；
3. 去掉 Q1 静态字符串 Python 编码；
4. 避免低基数字典 group-by 的 `torch.unique(dim=0)`。
5. 将 DuckDB scan node 自带的安全 predicates 下推到 scan source，并裁剪只为 filter 服务的列。
6. 将 partitionable final merge 从 Python row-dict 合并改为通用 tensor reductions。

当前还未达到成熟 Sirius/cuDF 级别的部分：

- scan 仍经 DuckDB/PyArrow/NumPy/PyTorch 复制链路，不是直接 GPU datasource → cudf/torch tensor；
- pull loop 仍串行，未 overlap DuckDB scan、H2D copy、GPU compute；
- final merge 仍在 Python row dict 上完成；
- chunk size 还没有自适应显存预算；
- hash join / sort / distinct 尚未实现完整 pipeline breaker state。

## 6. SF=1 观测

测试库：`data/tpch_sf1.duckdb`。CPU 命令：

```bash
python -m scripts.benchmark_query \
  --db data/tpch_sf1.duckdb \
  --query 1 \
  --device cpu \
  --cold-runs 0 --warmup-runs 0 --hot-runs 1 \
  --partition-table lineitem \
  --partition-chunk-size 1000000
```

CPU hot median 约 **727.935 ms**；CUDA hot median 约 **561.710 ms**（`warmup-runs=1 hot-runs=3`）。同一环境下 scan-only 读取 Q1 所需列从约 **0.86 s** 降到约 **0.45 s**，说明 scan source 的取数/编码/列裁剪已有明显改善；tensor final merge 又减少了 Python row-dict 合并开销。剩余主要热点转移到 per-batch projection、local aggregate 和 H2D/compute overlap。
