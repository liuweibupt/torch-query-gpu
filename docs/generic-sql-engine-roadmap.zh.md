# 通用 SQL 引擎化路线：对齐 Sirius 的工程方案

本文回答一个关键边界问题：当前项目不能只满足 TPC-H 固定集合，而要逐步成为“任意 SQL 输入、能覆盖则由 TQP/PyTorch 执行、不能覆盖则显式报告缺失算子”的通用数据库执行引擎。

## 1. 结论

成熟方案不是自己重写完整 SQL parser，而是把系统拆成两层：

```text
SQL frontend / optimizer：复用 DuckDB 这类成熟数据库
Execution backend：实现自己的 GPU/Tensor physical operators
```

Sirius 的公开文档说明它作为 DuckDB extension 接入：通过 optimizer hook 捕获优化后的计划，生成 GPU plan；成功时用 Sirius physical execution 替换 DuckDB physical plan，不成功时 Sirius 会回到 DuckDB CPU 执行。Sirius 的 GPU 栈基于 CUDA-X / cuDF / RMM，支持 filter、projection、join、group-by、order-by、aggregation、top-n、limit、CTE 等，并继续补 window 等高级算子。

本仓库采用同样的前端复用原则，但有一个不同的正确性策略：

```text
Sirius：生产系统体验优先，unsupported 可 graceful CPU fallback。
本仓库：研究原型正确性优先，unsupported 必须 UnsupportedPlanError，不允许静默 DuckDB-result fallback。
```

因此“任意 SQL”的实现目标应该表述为：

```text
任意 DuckDB 可 parse/plan 的 SQL 都能进入 TQP plan admission；
只要 physical plan 中每个节点都有 TQP/PyTorch operator，就由 PyTorch 执行；
缺失节点精确报出 operator / expression / type gap。
```

为了满足“功能上先通用”的需求，本仓库新增一个**显式** universal compatibility mode：

```text
--execution-mode strict      ：只允许 strict TQP/PyTorch physical operators，缺算子即失败。
--execution-mode universal   ：先试 strict；缺算子时用 DuckDB 执行 SQL，再把 Arrow result chunks 转成 TensorRecordBatch/TensorTable。
```

这个模式不是 query-id 模板，也不是静默 fallback；它是用户显式选择的兼容执行模式，用于保证任意 DuckDB 可执行 SQL 能先通过统一 columnar ABI 出结果。strict operator coverage 仍按 Roadmap 持续补齐。

实现约束：`universal` 必须是 `strict` 的功能超集。已经由 strict TQP/PyTorch physical operators 覆盖的查询，包括 TPC-H Q1-Q22，会先走 strict；只有 strict 报出缺失算子时才进入 DuckDB result → TensorRecordBatch materialization。批量验证 CLI 会按 query 流式打印进度，避免长时间无输出被误判为失败。

本轮新增 framework-level admission：

```text
tpch-torch-explain --db ... --sql-file ...
```

该命令不执行 SQL，只做 DuckDB parse/bind/optimize、`EXPLAIN JSON` lowering、`TQPOperatorGraph` slot binding，以及 static strict coverage report。它用于回答“这个 SQL 是否已经进入统一框架”和“strict 后端还缺哪个算子/表达式”，避免只能在执行阶段才发现 gap。

## 2. 对 Sirius / 成熟数据库的抽象拆解

### 2.1 Sirius-like 前端原则

Sirius 不是靠 query-id 模板实现 TPC-H，而是：

1. DuckDB 负责 SQL parse / bind / optimize。
2. optimizer hook 捕获 optimized logical/physical plan。
3. plan generator 把数据库 plan lowering 成 GPU execution plan。
4. scan、join、aggregate、sort、limit、CTE 等都作为后端算子实现。
5. 用 memory tier / partition / spilling 解决超过显存的数据集。

本仓库对应关系：

| Sirius / 成熟 DB 组件 | 本仓库现状 | 差距 |
| --- | --- | --- |
| DuckDB parser/binder/optimizer | `compile_sirius_plan()` 使用 DuckDB `json_serialize_sql`、`DESCRIBE`、`EXPLAIN JSON` | Python API 拿不到完整 bound expression C++ 对象 |
| Plan IR | `TQPOperatorGraph` + `TQPSlot` / `TQPBoundExpression` + `sql_admission.py` static coverage | slot AST 尚未完全替换字符串 expression executor |
| GPU physical operators | `backend/physical*.py` + PyTorch tensor ops | window/set/outer join/recursive 等还需扩展 |
| Vectorized pipeline | scan/partitionable batch pipeline 已有 `next_batch()` | join/sort/window 仍主要 materialized whole-table |
| Memory manager | resident tensor cache + chunk config | 还没有 RMM 级 allocator / spill manager |
| Unsupported behavior | strict 下显式 `UnsupportedPlanError`；universal 下显式 DuckDB result → TensorRecordBatch materialization | universal 是功能兼容模式，不代表对应算子已由 PyTorch 实现 |

### 2.2 通用化的核心不是“任意 SQL parser”

真正需要补的是 **physical operator coverage** 和 **expression/type coverage**：

```text
DuckDB physical JSON node
  -> TQPOperatorNode(kind, metadata, slots)
  -> PhysicalPlanExecutor dispatch
  -> PyTorch/TensorRecordBatch operator
```

因此每当新的 SQL 失败，不应该给该 SQL 写特化脚本，而应该把失败归类到：

- 缺少 physical node：例如 `UNION`、`WINDOW`、`UNNEST`。
- 缺少 join type：例如 full outer、asof、复杂 non-equi join。
- 缺少 expression：例如复杂 scalar function、regexp、interval arithmetic。
- 缺少 type：例如 nested/list/struct、timestamp with timezone。
- 缺少 execution model：例如 recursive CTE、ordered window frame、global set semantics。

## 3. 本轮已经落地的第一批通用 SQL 扩展

### 3.1 SET operation：`UNION`

新增：

```text
DuckDB UNION node
  -> OperatorKind.SET
  -> physical_union.execute_union_node()
```

实现语义：

- `UNION ALL`：按输出位置 concat 每个 child 的 tensor column。
- `UNION DISTINCT`：DuckDB 会规划成 `UNION` 后接 `HASH_GROUP_BY` 去重；本仓库复用已有 grouped aggregate/group-key unique path。
- 类型策略：DuckDB 必须已经完成类型 coercion；如果两个 child tensor dtype / dictionary / decimal metadata 不一致，显式报错，不做隐式错误转换。

### 3.2 WINDOW subset

新增：

```text
DuckDB WINDOW node
  -> OperatorKind.WINDOW
  -> physical_window.execute_window_node()
```

当前支持的 correctness-first subset：

| Window 形态 | 支持情况 |
| --- | --- |
| `row_number() over (partition by ... order by ...)` | 支持 |
| `rank() over (partition by ... order by ...)` | 支持 |
| `dense_rank() over (partition by ... order by ...)` | 支持，DuckDB physical 名称 `RANK_DENSE` |
| `sum/count/avg/min/max(x) over ()` | 支持 |
| `sum/count/avg/min/max(x) over (partition by ...)` | 支持 |
| aggregate window with `ORDER BY` frame | 暂不支持，显式报错 |
| window `NULLS FIRST/LAST` 带真实 NULL key | 暂不支持，显式报错 |

这不是 TPC-H 特化；它覆盖所有 lowering 到上述 DuckDB `WINDOW` physical projection 的 SQL。

### 3.3 Universal compatibility mode + TensorRecordBatch

新增：

```text
tpch_torch/backend/universal.py
```

执行链路：

```text
arbitrary SQL
  -> DuckDB execute
  -> Arrow RecordBatchReader(rows_per_batch=N)
  -> TensorRecordBatch.from_storages()
  -> TensorTable.from_batch()
  -> result rows
```

该路径覆盖 DuckDB 能执行的嵌套 SQL，包括 strict physical interpreter 尚未实现的 window frame、复杂 projection 或其他高级 plan。它仍然经过 `TensorRecordBatch`：

- INT / HUGEINT / unsigned integer：`ColumnStorage.fixed(torch.int64)`。
- FP32 / FP64：`ColumnStorage.fixed(torch.float32/float64)`。
- DECIMAL：`ColumnStorage.decimal64(scaled int64)`，保留 precision/scale。
- DATE：`YYYYMMDD int32`，`ColumnType.date`。
- VARCHAR / unknown scalar：dictionary ids，保留 validity mask。

限制：nested/list/struct 结果列目前会被视为字符串兼容列；这保证功能路径可跑，但不是最终 nested columnar execution。

### 3.4 Framework admission / explain coverage

新增：

```text
tpch_torch/sql_admission.py
scripts/explain_query.py
```

执行链路：

```text
arbitrary SQL
  -> compile_sirius_plan()
  -> TQPOperatorGraph
  -> analyze_strict_coverage(graph)
```

coverage report 包含：

- `strict_admissible`：静态看是否所有 node 都有 strict TQP/PyTorch 执行入口。
- `node_count`：lowering 后的 physical graph 节点数。
- `gaps`：`node_id / node_name / node_kind / reason`，例如完整 window frame 会报告 `aggregate WINDOW with ORDER BY frame is not supported yet`。

这个报告不是 runtime correctness proof：表达式、NULL、dtype、数据分布仍可能在执行时暴露更细 gap。但它把“SQL 解析/计划是否进入框架”和“后端缺算子”分开，符合成熟数据库的 admission → planning → execution 分层。

## 4. 后续 Roadmap：从 TPC-H coverage 到通用 DBMS coverage

### P0：SQL admission 和失败定位

- [x] 默认 SQL 输入经过 DuckDB parser/DESCRIBE/EXPLAIN JSON。
- [x] plan lowering 生成 typed `TQPOperatorGraph`。
- [x] `TQPSlot` 统一列名和 `#N` ordinal。
- [x] 显式 `--execution-mode universal`：任意 DuckDB 可执行 SQL 可通过 Arrow → TensorRecordBatch materialization 跑通。
- [x] `tpch-torch-explain`：任意 DuckDB 可 parse/plan SQL 可 lowering 到 `TQPOperatorGraph`，并输出 strict static coverage gaps。
- [ ] 增加 `tpch-torch-explain-coverage`：输出缺失 physical node、表达式、类型、是否可 chunk/pipeline。
- [ ] 每个 `UnsupportedPlanError` 附带 node id、operator name、metadata snippet。

### P1：补齐通用 relational physical nodes

- [x] `UNION` / `UNION DISTINCT` 基础路径。
- [x] `WINDOW` 第一批 ranking + partition aggregate。
- [ ] `FULL OUTER JOIN`。
- [ ] `CROSS_PRODUCT` / `POSITIONAL_JOIN` / 更完整 non-equi join。
- [ ] `UNNEST` / `LIST` / `STRUCT` 基础展开。
- [ ] `INTERSECT` / `EXCEPT`：若 DuckDB lowering 成 ANTI/SEMI + aggregate，则复用；否则新增 set node。
- [ ] recursive CTE：需要 fixpoint execution，不应塞进普通 materialized CTE。

### P2：表达式与类型覆盖

- [x] arithmetic / comparison / CASE / CAST / EXTRACT / LIKE / IN。
- [x] DECIMAL scaled int64 metadata propagation。
- [ ] timestamp / interval arithmetic。
- [ ] regexp / string functions / substring variants。
- [ ] NULL three-valued logic 全表达式覆盖。
- [ ] nested/list/struct 类型：需要 Arrow-like offsets/children storage。

### P3：执行模型与调度

- [x] scan/partitionable aggregate batch pipeline。
- [ ] join-aware chunk pipeline：build/probe 分离、partitioned hash join、spill。
- [ ] sort/window external merge：局部 run + global merge。
- [ ] operator scheduler：pipeline breakers 标记、memory budget、device placement。
- [ ] cache/pin API：对齐 Sirius `pin_table` 思路，支持 hot run resident columns。

## 5. 当前边界表述

更新后的准确表述应是：

```text
本项目不是“所有 SQL 永远成功”的完整 DBMS；
但它已经不是 TPC-H query-id 模板系统。
它是一个 DuckDB-planned、coverage-driven 的 TQP/PyTorch physical engine。
任意 SQL 都可尝试 admission；执行成功取决于 physical node/operator coverage。
strict 模式新增 SQL coverage 必须通过通用 operator 实现，不允许 query-specific Python 脚本或 DuckDB result fallback。
universal 模式用于功能兼容：任意 DuckDB 可执行 SQL 可以先落到 TensorRecordBatch ABI，但该路径中的缺失算子由 DuckDB 执行，不计入 strict PyTorch operator coverage。
```

## 参考

- Sirius GitHub: <https://github.com/sirius-db/sirius>
- Sirius `gpu_execution` 文档：<https://github.com/sirius-db/sirius/blob/main/docs/gpu_execution.md>
- DuckDB JSON plan / Substrait 相关前端使用见本仓库 `tpch_torch/duckdb_plan_json.py` 与 `tpch_torch/frontend/sirius.py`。
