# torch-query-gpu

> 中文 README · TQP-style SQL analytics on PyTorch/CUDA tensors

`torch-query-gpu` 是一个**正确性优先**的 TQP 风格原型：从原始 SQL 出发，复用 DuckDB/Sirius-like 前端完成 SQL 解析与计划准入，再把计划交给 PyTorch 后端，在 CPU 或 CUDA tensor 上执行分析型查询。

```text
目标链路：SQL → DuckDB/Sirius-like Frontend → TQP IR → PyTorch/CUDA Operators → Result Rows
```

## 当前状态

- ✅ 默认链路支持 TPC-H Q1-Q22：原始 SQL → DuckDB `EXPLAIN (FORMAT JSON)` lowering → `TQPOperatorGraph` → PyTorch graph executor。
- ✅ CLI 能直接读取 `--query`、`--sql` 或 `--sql-file`，不需要手工导出 JSON。
- ✅ strict Substrait 路径仍可显式使用：`--frontend substrait`，只运行 DuckDB 原生 exporter 能导出的 SQL。
- ✅ Generic SQL subset 已支持单表 projection/filter/aggregate/order/limit。
- ✅ Q1-Q22 默认路径已迁入 DuckDB physical-plan interpreter：SQL → DuckDB JSON physical plan → `TQPOperatorGraph` → `execute_physical_plan()` → PyTorch tensor operators。
- ✅ Q2-Q22 已从早期 graph recipes 继续推进到 DuckDB physical-plan interpreter；旧 `queries/qXX` 模板和 `compiled_tpch` root 均不再作为执行 fallback。
- ✅ DuckDB physical-plan interpreter v1 已接入：Generic equi-join / join+group aggregate / final aggregate expression / basic HAVING / searched CASE / TOP_N 可从 SQL 直接 lowering 到 PyTorch；TPC-H Q1-Q22 已迁到该通用 interpreter，不再走 query-id recipe。
- ✅ 新增 physical-only TPC-H coverage probe：直接调用 `execute_physical_plan()` 衡量哪些 TPC-H 查询已脱离 graph recipe。
- ✅ Q6 有 correctness-first 压缩 mask 原型：`--compressed-masks`。
- ✅ 论文驱动优化新增：physical SEMI/ANTI membership probe、sorted group-by `unique_consecutive` fast path、RLE `COUNT/SUM/MIN/MAX/AVG` primitive。
- ✅ Q1 hot benchmark 使用 per-connection resident tensor cache，warmup 后复用已转换 lineitem tensors；Q1 fused aggregation 使用 masked `torch.bincount`，避免 selected-row payload gather。
- ✅ 提供冷/热端到端 benchmark：`tpch-torch-benchmark`。
- ⚠️ 当前不是完整 SQL 数据库：frontend 能接收 DuckDB 可 parse/plan 的 SQL；PyTorch 后端已能解释一批 DuckDB physical plan nodes；basic HAVING 已支持，复杂 generic subquery/CTE/window/set operation 仍显式失败。

## 一图看懂架构

```mermaid
flowchart LR
    SQL["SQL / TPC-H Query<br/>--query / --sql / --sql-file"] --> Runner["runner.load_sql"]
    Runner --> Frontend{"frontend"}
    Frontend -->|默认 sirius| Sirius["DuckDB/Sirius-like Frontend<br/>Parser · Binder · Planner · Optimizer<br/>EXPLAIN metadata"]
    Frontend -->|显式 substrait| Substrait["DuckDB Native Substrait<br/>get_substrait_json(original_sql)"]
    Sirius --> Graph["DuckDB JSON physical plan<br/>→ TQPOperatorGraph"]
    Substrait --> IR["TQPPlan IR"]
    Graph --> IR["TQPPlan IR<br/>operator_graph boundary"]
    IR --> Backend["PyTorchBackend"]
    Backend --> GraphExec["PyTorchGraphExecutor.forward-like execute"]
    GraphExec -->|Q1-Q22 + generic joins| Physical["DuckDB physical-plan interpreter<br/>backend/physical*.py"]
    GraphExec -->|Q6 --compressed-masks| Primitives["Q6 compressed mask experimental primitive"]
    GraphExec -->|single-table generic subset| Generic["tpch_torch/backend/generic.py"]
    Physical --> Nodes["Physical tensor nodes<br/>Scan · Filter · Project · Join · Aggregate · Sort/TopN"]
    Primitives --> Torch["PyTorch Tensor Operators<br/>CPU / CUDA"]
    Nodes --> Torch
    Generic --> Torch
    Torch --> Rows["Result Rows"]
    Rows -. correctness only .-> DuckDB["DuckDB baseline validation<br/>not a fallback"]
```

### 分层职责

| 层 | 模块 | 做什么 |
| --- | --- | --- |
| CLI | `scripts/run_query.py`, `scripts/validate_query.py`, `scripts/benchmark_query.py` | 接收 SQL 来源、frontend、device、benchmark 参数。 |
| Runner | `tpch_torch/runner.py` | 读取 SQL，编译 `TQPPlan`，调用后端，validation 时比较 DuckDB baseline。 |
| Frontend | `tpch_torch/frontend/sirius.py`, `tpch_torch/frontend/substrait.py` | 把原始 SQL 编译成 `TQPPlan`；默认是 Sirius-like DuckDB planner admission。 |
| IR | `tpch_torch/ir/plan.py` | 前端与后端之间的不可变边界对象。 |
| Backend | `tpch_torch/backend/pytorch.py`, `tpch_torch/backend/graph.py`, `tpch_torch/backend/generic.py`, `tpch_torch/backend/physical*.py` | 只通过 `TQPOperatorGraph` 进入 PyTorch graph executor；Q1-Q22 与 generic joins 可由 DuckDB physical-plan interpreter 执行；不执行 compiled TPC-H fallback root。 |
| Graph nodes / Operators | `tpch_torch/backend/graph_nodes.py`, `tpch_torch/backend/physical*.py`, `tpch_torch/operators.py`, `tpch_torch/compressed*.py` | Scan、filter、project、lookup/hash/equi join、membership-only semi/anti join、scalar/grouped scalar subquery、CTE、aggregate、sort/top-k、Plain/RLE/Index mask 与 RLE aggregate primitives。 |

## Q1 是怎么实现的

Q1 当前走 graph-lowered fused physical primitive：DuckDB JSON physical plan 先被 lowering 成 `TQPOperatorGraph`，backend 进入 `execute_physical_plan()` 后由 `physical_fusion.py` 识别 canonical Q1 graph shape，并把 scan/filter/project/group/order 的重计算段融合为 dense-id grouped tensor reductions。未识别的 graph 仍显式走普通 physical interpreter。

```mermaid
flowchart TD
    Q1SQL["TPC-H Q1 SQL"] --> Frontend["DuckDB/Sirius-like frontend<br/>或 strict Substrait frontend"]
    Frontend --> Graph["DuckDB JSON plan → TQPOperatorGraph"]
    Graph --> Plan["TQPPlan.operator_graph"]
    Plan --> Backend["PyTorchGraphExecutor"]
    Backend --> Physical["execute_physical_plan()"]
    Physical --> Fusion["physical_fusion.try_execute_fused_physical_plan"]
    Fusion --> Scan["fetch lineitem tensors<br/>scan filter once"]
    Scan --> Project["fused expressions<br/>discounted price / charge"]
    Project --> Group["dense group id + torch.bincount<br/>sum / avg / count_star"]
    Group --> Rows["decode tiny grouped result"]
```

关键代码位置：

- `tpch_torch/duckdb_plan_json.py`：DuckDB `EXPLAIN (FORMAT JSON)` lowering 到 `TQPOperatorGraph`。
- `tpch_torch/backend/graph.py`：`PyTorchGraphExecutor` 将 Q1-Q22 分发到 `execute_physical_plan()`。
- `tpch_torch/backend/physical.py`：解释 DuckDB physical nodes，并在入口调用 graph-lowered fusion hook。
- `tpch_torch/backend/physical_fusion.py`：识别 canonical Q1 graph shape，执行 fused dense-id grouped reductions。
- `tpch_torch/backend/physical_expr.py`：解释普通 physical path 中的 arithmetic / internal compress-decompress / projection ref 表达式。

```python
# tpch_torch/backend/pytorch.py
if plan.operator_graph is not None:
    return PyTorchGraphExecutor().execute(
        con, plan, device=device, use_compressed_masks=use_compressed_masks
    )
```

```python
# tpch_torch/backend/graph.py
if plan.query_id in _PHYSICAL_TPCH_QUERIES:  # Q1-Q22
    return execute_physical_plan(con, graph, device=device)

# tpch_torch/backend/physical.py
fused_rows = physical_fusion.try_execute_fused_physical_plan(...)
if fused_rows is not None:
    return fused_rows
```

## 安装

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

## 生成 TPC-H 数据

```bash
# 默认 SF=1
python -m scripts.gen_sf1 --db data/tpch_sf1.duckdb --sf 1

# 或安装 editable 后使用 entrypoint
tpch-torch-gen-sf1 --db data/tpch_sf1.duckdb --sf 1
```

## 运行与验证

### 运行单个 TPC-H 查询

```bash
tpch-torch-run \
  --db data/tpch_sf1.duckdb \
  --query 1 \
  --device cuda \
  --frontend sirius
```

没有 CUDA 的机器请使用 `--device cpu`。如果显式请求 `--device cuda` 但 PyTorch 检测不到 CUDA，命令会直接报错，不会静默回退到 CPU。

### 验证单个查询

```bash
tpch-torch-validate \
  --db data/tpch_sf1.duckdb \
  --query 1 \
  --device cuda \
  --frontend sirius
```

Validation 会运行同一条 PyTorch 链路，并把结果与 DuckDB 对同一条原始 SQL 的执行结果比较。DuckDB 只用于 baseline，不是 fallback 输出。

### 直接运行 SQL 文本或 SQL 文件

```bash
tpch-torch-validate \
  --db data/tpch_sf1.duckdb \
  --sql "select count(*) as n from lineitem" \
  --device cuda

cat queries/my_query.sql | sed -n '1,120p'
tpch-torch-run \
  --db data/tpch_sf1.duckdb \
  --sql-file queries/my_query.sql \
  --device cuda
```

当前 generic SQL 支持分两层：

1. `backend/generic.py`：单表 SQL parser/executor。
2. `backend/physical.py`：DuckDB JSON physical-plan interpreter v1，直接解释 DuckDB plan nodes。

已支持：

```text
single-table SELECT
WHERE comparisons / IN / LIKE / NOT LIKE / AND / OR / NOT
column projection、nested SELECT alias、EXTRACT(year FROM date)、searched/simple CASE 与简单 arithmetic projection
COUNT(*), COUNT(col), SUM, MIN, MAX, AVG
simple GROUP BY
basic HAVING over aggregate aliases / aggregate expressions
ORDER BY output columns ASC / DESC
LIMIT / duplicate-free single-key TOP_N tensor top-k path
DuckDB physical SEQ_SCAN / FILTER / PROJECTION
DuckDB physical HASH_JOIN inner / multi-column equi-join（correctness-first）
DuckDB physical HASH_GROUP_BY / PERFECT_HASH_GROUP_BY / UNGROUPED_AGGREGATE
DuckDB physical ORDER_BY / TOP_N / LIMIT（无重复 key 的单 key TOP_N 使用 torch.topk）
final aggregate expression alias，例如 100 * sum(x) / sum(y)；支持 aggregate alias/order 归一化
```

本轮算子优化已覆盖 physical-plan interpreter 的几条热路径：

- `HASH_JOIN` row-index 生成从 Python `dict/list/tolist()` 改为 tensor `searchsorted` 路径；右侧 build key 已排序且唯一时跳过 `argsort` 和重复展开。
- 已知 TPC-H 低基数字符串列使用 table-aware static dictionary encoding，避免大列上反复 `numpy.unique`。
- `IN` / 同列 literal `OR` 使用 membership mask；singleton membership 直接走 equality。
- `PhysicalTable.filter/gather` 对共享 alias 的 `PhysicalValue` 只转换一次，减少 physical plan 中 `col` / `table.col` alias 的重复 tensor selection。
- 本批新增通用 physical lowering：multi-column equi-join、CAST wrapper、NOT LIKE (`!~~`)、nested SELECT alias、EXTRACT(year)、parent-required join key retention、aggregate ORDER BY alias matching、basic HAVING、multi-branch/simple CASE、duplicate-free single-key TOP_N tensor top-k。

### 验证全部 TPC-H

```bash
tpch-torch-validate \
  --db data/tpch_sf1.duckdb \
  --queries all \
  --device cuda \
  --frontend sirius \
  --keep-going
```

这条命令走完整默认链路：**SQL → DuckDB/Sirius-like frontend → TQPPlan → PyTorch backend**。

### Strict Substrait 路径

```bash
tpch-torch-run \
  --db data/tpch_sf1.duckdb \
  --query 6 \
  --device cuda \
  --frontend substrait \
  --json

# 探测 DuckDB 原生 Substrait exporter 当前覆盖情况
tpch-torch-probe-substrait --db data/tpch_sf1.duckdb --queries all --json
```

Substrait 策略：

```text
original SQL
  -> DuckDB get_substrait_json(original_sql)
  -> TQPPlan carrying real Substrait JSON
  -> PyTorch backend
```

如果 DuckDB exporter 导不出原始 SQL，该路径会显式失败；项目不会改写 SQL、伪造 JSON 或自动切到 Sirius-like 路径。

### Q6 压缩 mask 原型

```bash
tpch-torch-run \
  --db data/tpch_sf1.duckdb \
  --query 6 \
  --device cuda \
  --compressed-masks

tpch-torch-validate \
  --db data/tpch_sf1.duckdb \
  --query 6 \
  --device cuda \
  --compressed-masks
```

`--compressed-masks` 当前只改变 Q6 的 PyTorch predicate mask 执行方式：Plain/RLE/Index mask dispatch。它还不是完整压缩列存储或压缩 join/aggregate 执行。

## 冷/热性能计时

```bash
tpch-torch-benchmark \
  --db data/tpch_sf1.duckdb \
  --query 1 \
  --device cuda \
  --frontend sirius \
  --cold-runs 3 \
  --warmup-runs 5 \
  --hot-runs 20

# JSON 输出，便于脚本收集
tpch-torch-benchmark \
  --db data/tpch_sf1.duckdb \
  --query 6 \
  --device cuda \
  --compressed-masks \
  --json
```

计时语义：

- **cold**：每个样本新建 DuckDB connection，运行完整 frontend + tensor fetch/encoding + PyTorch backend + result materialization，再关闭连接。不刷新 OS page cache，也不重启 Python。
- **hot**：复用一个 DuckDB connection，先执行 `--warmup-runs`，再记录 `--hot-runs`；Q1 的 lineitem tensor table 会在该连接内常驻复用，更接近 TQP 论文中“数据已转换为 PyTorch tensors 后测 query execution”的口径。
- **CUDA**：每个样本前后调用 `torch.cuda.synchronize()`，报告 wall-clock ms，因此包含 CPU 侧 frontend/fetch/materialization 与 GPU work。
- Benchmark 不做 DuckDB validation；正确性请单独运行 `tpch-torch-validate`。

### 最近一次 Q1 性能对比（SF=1）

同一台机器（CPU + RTX 4090），命令均为 `--cold-runs 3 --warmup-runs 3 --hot-runs 10 --frontend sirius`。计时为端到端 wall-clock，包含 DuckDB frontend、tensor fetch/encoding、PyTorch 执行和结果 materialization；未刷新 OS page cache。

| 版本 | device | cold median | hot median | 变化（hot median） |
| --- | --- | ---: | ---: | ---: |
| `c8b3bd9`（full graph recipes merge 前） | CPU | 794.123 ms | 794.571 ms | baseline |
| 当前 physical-interpreter 分支 | CPU | 973.831 ms | 904.851 ms | +13.9% |
| `c8b3bd9`（full graph recipes merge 前） | CUDA | 675.461 ms | 650.313 ms | baseline |
| 当前 physical-interpreter 分支 | CUDA | 745.996 ms | 703.522 ms | +8.2% |

注意：该历史表是在 Q1 迁入 physical interpreter 之前记录的 direct primitive 路径数据。Q1 现在已经改为 SQL-lowered physical interpreter 路径，后续需要重新建立新的冷/热性能基线。

### 当前算子优化 smoke benchmark（SF=1）

命令均为 `--cold-runs 1 --warmup-runs 1 --hot-runs 3 --frontend sirius --device cpu`，计时端到端且较短，仅用于确认优化方向：

| Query | 优化前 hot median | 优化后 hot median | 说明 |
| --- | ---: | ---: | --- |
| Q12 | 2293.140 ms | 2049.648 ms | static dictionary + alias 去重；join 仍受 row expansion 影响 |
| Q14 | 857.160 ms | 703.668 ms | sorted unique build-side join fast path |
| Q19 | 2111.015 ms | 1744.495 ms | tensor join + membership/OR 折叠 + alias 去重 |

Q1 已在后续改为 DuckDB physical-plan interpreter 路径；上表 smoke benchmark 主要覆盖 Q12/Q14/Q19 和 generic physical-plan 算子优化。

### Q1 graph-lowered fusion smoke benchmark（SF=1）

命令均为 `--query 1 --frontend sirius --cold-runs 1 --warmup-runs 1 --hot-runs 5`，计时端到端且样本很短，仅用于确认优化方向：

| 版本 | device | cold median | hot median | 说明 |
| --- | --- | ---: | ---: | --- |
| 普通 physical interpreter | CPU | 8969.746 ms | 9345.923 ms | generic projection/groupby path |
| Q1 fusion：dense grouped reductions | CPU | 1515.715 ms | 1114.154 ms | 仍每次 DuckDB→tensor fetch |
| Q1 resident + masked bincount | CPU | 943.278 ms | 198.580 ms | hot 复用 resident tensors，aggregation 不做 payload gather |
| Q1 resident + masked bincount | CUDA | 719.700 ms | 12.365 ms | hot 接近 TQP 论文 execution-time 口径；cold 仍含首次 DuckDB→GPU tensor conversion |

该优化没有绕过 SQL lowering：Q1 仍从 DuckDB JSON physical plan 进入 `TQPOperatorGraph`，只是 physical backend 在 graph shape 上选择 fused primitive。论文中 TQP 报告的是输入列已离线转换为 PyTorch tensors 后的 query execution time；因此应主要看 hot/resident 结果，cold 结果用于暴露 ingestion/transfer 成本。


## TPC-H 支持矩阵

| Query set | 默认 Sirius-like frontend | Strict DuckDB Substrait frontend | PyTorch backend | 当前后端形态 |
| --- | --- | --- | --- | --- |
| Q1-Q22 | yes | partial; DuckDB exporter still blocks several complex queries | yes | DuckDB physical-plan interpreter v1 by default |
| Q6 `--compressed-masks` | yes | yes | yes | explicit compressed-mask primitive experiment |

说明：strict Substrait 的 blocked 是 DuckDB 原生 exporter 覆盖限制，不代表 PyTorch backend 没有这些查询的 executor。默认 Sirius-like 路径下 Q1-Q22 都先 lowering 到 `TQPOperatorGraph` 再进入 PyTorch backend。

## Roadmap 摘要

完整清单见：

- 中文执行版：[`docs/operator-roadmap.zh.md`](docs/operator-roadmap.zh.md)
- 英文原版：[`docs/operator-roadmap.md`](docs/operator-roadmap.md)

当前批次状态：

- [x] TPC-H Q1-Q22 通过 DuckDB JSON physical plan lowering 到 `TQPOperatorGraph` 后进入 PyTorch graph executor。
- [x] Strict DuckDB Substrait path：覆盖 DuckDB exporter 能导出的查询。
- [x] Batch 1 primitives：grouped min/max/mean、mask helpers、top-k、首批 RLE mask primitives。
- [x] Batch 2 部分 generic SQL：`MIN`、`MAX`、`AVG`、`COUNT(col)`、boolean filters、`IN`、`LIKE`、basic `HAVING`、generic `CASE`、`ORDER BY ASC/DESC`、duplicate-free single-key top-k `LIMIT`。
- [x] Q1 已增加 graph-lowered fused physical primitive：仍由 SQL/DuckDB graph 触发；hot path 复用 resident tensors，并用 masked `torch.bincount` 做融合聚合。
- [x] Q6 默认路径已迁到 DuckDB physical-plan interpreter；`--compressed-masks` 保留显式 compressed mask primitive 实验。
- [x] Generic equi-join / join+aggregate / final aggregate expression 已通过 DuckDB physical-plan interpreter v1 跑通。
- [x] Physical-plan 算子热路径优化：tensor join index、sorted-unique build fast path、SEMI/ANTI membership probe、sorted group-by fast path、static dictionary encoding、membership mask、alias 去重 gather/filter。
- [ ] Generic subquery lowering、window、set operations；更复杂 `HAVING` / `CASE` SQL shapes 继续扩展。
- [x] 压缩数据第一批 aggregate primitive：RLE `COUNT` / `SUM` / `MIN` / `MAX` / `AVG` 基于 run lengths 执行，不展开 rows。
- [ ] 完整 compressed storage metadata、encoded column execution、compressed aggregation/join。
- [x] 第一版显式 `TQPOperatorGraph` 与 DuckDB JSON lowering。
- [x] Q2-Q22 默认 physical interpreter coverage 已跑通；历史 graph recipe 源码不再作为默认执行路径。
- [x] Q1-Q22 默认路径已从旧 query-id dispatch 迁到 DuckDB physical-plan interpreter。
- [x] 新增 `tpch_torch.physical_coverage`，用于探测 TPC-H physical-only 自动算子覆盖；当前 Q1-Q22 全部 supported。
- [x] 第一批 graph-lowered fusion：Q1 scan/filter/project/group/order fused dense grouped reductions。
- [ ] 更多 fusion、scheduling、compiler lowering。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [`docs/architecture.zh.md`](docs/architecture.zh.md) | 中文架构说明、关键代码片段、Q1 分层图。 |
| [`docs/q1-end-to-end-execution.zh.md`](docs/q1-end-to-end-execution.zh.md) | 以 TPC-H Q1 为例，详细解释 SQL 解析、DuckDB JSON physical plan、TQPOperatorGraph、PyTorch backend/operator 与冷/热执行口径。 |
| [`docs/gpu-sql-ecosystem-analysis.zh.md`](docs/gpu-sql-ecosystem-analysis.zh.md) | 对比 RAPIDS/cuDF/RMM、Sirius-like 前端、TQP/TQP++/CoddSpeed 与本项目 PyTorch 路线，分析软件栈、显存管理和复用 CUDA 算子的工程取舍。 |
| [`docs/architecture.md`](docs/architecture.md) | 英文架构说明。 |
| [`docs/operator-roadmap.zh.md`](docs/operator-roadmap.zh.md) | 中文 Roadmap / TODO。 |
| [`docs/operator-roadmap.md`](docs/operator-roadmap.md) | 英文完整 Roadmap。 |
| [`docs/papers/README.md`](docs/papers/README.md) | 已下载论文与来源说明。 |

## 开发验证

```bash
# 单元测试，后端测试建议保持 60 秒 timeout
timeout 60 python -m pytest -q

# Python 文件语法检查
timeout 60 python -m compileall -q tpch_torch scripts
```
