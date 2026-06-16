# 当前架构：DuckDB/Sirius-like 前端 → TQP IR → PyTorch Graph Nodes

本文档描述当前仓库的真实执行链路。DuckDB 负责 SQL 解析、绑定、计划准入和 JSON physical plan 输出；Sirius-like 前端 lowering 成 `TQPOperatorGraph`；PyTorch 后端在 CPU/CUDA tensor 上执行 graph nodes。DuckDB 只在 validation 中作为 baseline，不是执行 fallback。

## 1. 一图看懂端到端链路

```mermaid
flowchart LR
    SQL["SQL 文本<br/>--query / --sql / --sql-file"] --> Runner["runner.load_sql"]
    Runner --> Frontend{"frontend"}
    Frontend -->|默认 sirius| Sirius["DuckDB parser/binder/planner<br/>EXPLAIN + JSON physical plan"]
    Frontend -->|严格 substrait| Substrait["DuckDB native Substrait exporter"]
    Sirius --> Graph["TQPOperatorGraph"]
    Substrait --> Plan["TQPPlan"]
    Graph --> Plan
    Plan --> Backend["PyTorchBackend"]
    Backend --> GraphExec["PyTorchGraphExecutor"]
    GraphExec -->|Q1-Q22 + generic joins| Physical["DuckDB physical-plan interpreter"]
    GraphExec -->|Q6 --compressed-masks| Prim["Q6 compressed mask 实验"]
    GraphExec -->|单表 generic subset| Generic["generic.py"]
    Physical --> PhysNodes["physical tensor nodes"]
    Prim --> Torch["PyTorch tensor ops<br/>CPU / CUDA"]
    PhysNodes --> Torch
    Generic --> Torch
    Torch --> Rows["结果 rows"]
    Rows -. correctness only .-> DuckDB["DuckDB baseline"]
```

## 2. 当前后端契约

- TPC-H Q1-Q22 都必须携带 DuckDB JSON lowering 出来的真实 `TQPOperatorGraph` root。
- `OperatorKind.COMPILED_TPCH` root 会被拒绝；不再有 compiled/template fallback。
- Q1-Q22 已迁入 `tpch_torch/backend/physical*.py`，由 DuckDB JSON physical-plan interpreter 直接解释执行。
- Q6 默认也走 physical interpreter；`--compressed-masks` 在 `tpch_torch/backend/graph.py` 中保留为显式 compressed-mask primitive 实验路径。
- Q1-Q22 现在均由 DuckDB JSON physical-plan interpreter 执行；`tpch_graph_qXX` 模块保留作历史对照，不再是默认执行路径。
- 非 TPC-H SQL 先使用显式单表 generic SQL subset；join 和 join+aggregate 形状由 physical-plan interpreter 执行。不支持的 SQL shape 抛出 `UnsupportedPlanError`。

## 3. 模块分层

| 层级 | 关键文件 | 职责 |
| --- | --- | --- |
| CLI | `scripts/run_query.py`, `scripts/validate_query.py`, `scripts/benchmark_query.py` | 解析 SQL 来源、frontend、device、validation/benchmark 参数。 |
| Runner | `tpch_torch/runner.py` | 薄编排：读取 SQL，编译 frontend plan，调用 backend，必要时 validation。 |
| Frontend | `tpch_torch/frontend/sirius.py`, `tpch_torch/frontend/substrait.py` | 把原始 SQL 编译成 `TQPPlan`。 |
| DuckDB lowering | `tpch_torch/duckdb_plan_json.py`, `tpch_torch/planner.py` | 导出 DuckDB 文本/JSON plan，并把 JSON node lowering 到 `TQPOperatorGraph`。 |
| IR | `tpch_torch/ir/plan.py`, `tpch_torch/operator_graph.py` | 不可变前后端边界。 |
| Backend dispatch | `tpch_torch/backend/pytorch.py`, `tpch_torch/backend/graph.py` | TPC-H 强制走 graph；分发 Q1-Q22 physical interpreter、显式 Q6 compressed-mask primitive 实验和 generic SQL。 |
| Physical interpreter | `tpch_torch/backend/physical.py`, `physical_expr.py`, `physical_projection.py`, `physical_required.py`, `physical_join.py`, `physical_sql*.py`, `physical_types.py`, `static_dictionaries.py` | 解释 DuckDB `SEQ_SCAN`、`FILTER`、`PROJECTION`、inner 与 multi-column equi `HASH_JOIN`、grouped/ungrouped aggregate、`ORDER_BY`、`TOP_N`、`LIMIT`、final aggregate expression；包含 tensor join index、SEMI/ANTI membership probe、sorted group-by fast path、membership folding、static dictionary encoding 和 alias 去重 selection。 |
| Graph nodes | `tpch_torch/backend/graph_nodes.py` | Scan、filter、lookup join、semi/anti join、scalar subquery、grouped scalar subquery、CTE materialization、aggregate、sort/limit helpers。 |
| Tensor operators | `tpch_torch/operators.py`, `tpch_torch/compressed*.py` | grouped reductions、dense group id、top-k、Plain/RLE/Index mask、公开 range primitives 与 RLE aggregate primitives。 |

## 4. 关键代码片段

Sirius-like 前端把 DuckDB JSON physical plan 挂到 `TQPPlan.operator_graph`：

```python
physical_plan_json = export_duckdb_physical_plan_json(con, sql)
operator_graph = lower_duckdb_json_to_operator_graph(sql, query_id, physical_plan_json)
return TQPPlan(..., operator_graph=operator_graph)
```

PyTorch 后端拒绝没有 graph 的 TPC-H plan：

```python
if plan.operator_graph is not None:
    return PyTorchGraphExecutor().execute(con, plan, device=device)
if plan.query_id is not None:
    raise UnsupportedPlanError(
        f"TPC-H Q{plan.query_id} requires a frontend-lowered TQP operator graph"
    )
```

DuckDB physical-plan interpreter 已成为独立后端层：

```python
# tpch_torch/backend/graph.py
if plan.query_id in _PHYSICAL_TPCH_QUERIES:  # Q1-Q22
    return execute_physical_plan(con, graph, device=device)
```

```python
# tpch_torch/backend/physical.py
if node.kind == OperatorKind.JOIN:
    return self._execute_join(node)
if node.kind == OperatorKind.AGGREGATE:
    return self._execute_aggregate(node)
```

该 interpreter 仍保持显式失败原则：不支持的 DuckDB physical node 会抛出 `UnsupportedPlanError`，不会改用 DuckDB rows 作为 PyTorch 输出。

本轮 physical 算子优化保持同一执行边界，只替换热点实现：

```python
# tpch_torch/backend/physical_join.py
right_order, sorted_right_values = _sorted_build_keys(right_values)
starts = torch.searchsorted(sorted_right_values, left_values, right=False)
ends = torch.searchsorted(sorted_right_values, left_values, right=True)
if _is_strictly_increasing(sorted_right_values):
    return _unique_build_join_indices(starts, ends - starts, right_order)
```

```python
# tpch_torch/backend/physical_membership.py
positions = torch.searchsorted(sorted_right, left_key)
return in_bounds & (sorted_right[safe_positions] == left_key)
```

```python
# tpch_torch/backend/physical_aggregate.py
if _is_lexicographically_non_decreasing(stacked_keys):
    return torch.unique_consecutive(stacked_keys, dim=0, return_inverse=True)
```

```python
# tpch_torch/backend/physical_expr.py
folded = fold_same_column_literal_or(...)
if folded is not None:
    return folded
```

```python
# tpch_torch/backend/physical_types.py
_transform_unique_values(self.columns, lambda value: value.gather(indices))
```

通用 graph nodes 暴露 join/subquery/aggregate 形态：

```python
@dataclass(frozen=True)
class AntiJoinNode:
    probe_keys: torch.Tensor
    build_keys: torch.Tensor

    def execute(self) -> torch.Tensor:
        return ~SemiJoinNode(self.probe_keys, self.build_keys).execute()

@dataclass(frozen=True)
class GroupedScalarSubqueryNode:
    keys: torch.Tensor
    values: torch.Tensor

    def lookup(self, probe_keys, missing_value=-1):
        build_key, probe_key = _packed_lookup_keys(...)
        return lookup_tensor_values(build_key, self.values, probe_key, missing_value)
```

Q20 的 correlated grouped scalar subquery 已经表达为 graph-node 组合：

```python
shipped_quantity_by_pair = GroupedScalarSubqueryNode.sum(
    (lineitem.columns["l_partkey"][ship_mask], lineitem.columns["l_suppkey"][ship_mask]),
    lineitem.columns["l_quantity"][ship_mask],
)
shipped_quantity = shipped_quantity_by_pair.lookup(
    (partsupp.columns["ps_partkey"], partsupp.columns["ps_suppkey"]),
    missing_value=0.0,
)
qualifying_suppkeys = torch.unique(partsupp.columns["ps_suppkey"][
    SemiJoinNode(partsupp.columns["ps_partkey"], forest_partkeys).execute()
    & (partsupp.columns["ps_availqty"] > 0.5 * shipped_quantity)
])
```

## 5. Q1 physical interpreter 分层图

```mermaid
flowchart TD
    Q1SQL["TPC-H Q1 SQL"] --> Frontend["DuckDB/Sirius-like frontend"]
    Frontend --> Graph["DuckDB JSON → TQPOperatorGraph"]
    Graph --> Physical["execute_physical_plan()"]
    Physical --> Fusion["physical_fusion hook"]
    Fusion --> Scan["fetch tensors + scan filter once"]
    Scan --> Project["fused arithmetic exprs"]
    Project --> Aggregate["dense group id + torch.bincount"]
    Aggregate --> Rows["decoded result rows"]
```

Q1 现在和已迁移的 generic joins 共用 physical interpreter 边界，并增加 graph-lowered fusion hook：DuckDB 输出 physical node graph，`physical.py` 进入 fusion hook，`physical_fusion.py` 用 dense-id grouped tensor reductions 执行 canonical Q1；host 侧只做最终 grouped rows 的 decode/materialization。

## 6. SQL 支持边界

Sirius-like 前端能接收 DuckDB 可解析和计划的 SQL；PyTorch 后端当前可执行：

```text
TPC-H Q1-Q22：TQPOperatorGraph + PyTorch graph nodes
Q1-Q22：DuckDB physical-plan interpreter v1
generic equi-join 与 join+aggregate：DuckDB physical-plan interpreter v1
generic SQL：single-table SELECT / WHERE / projection / aggregate / GROUP BY / ORDER BY / LIMIT
```

Generic joins 已通过 DuckDB physical-plan interpreter 部分支持。TPC-H 所需的 delimiter/mark/nested-loop/subquery/CTE 形状已经覆盖到 correctness-first 级别；更一般的 generic subqueries、windows、set operations、HAVING 仍显式失败。

## 7. TPC-H 支持矩阵

| Query set | 默认 Sirius-like frontend | Strict DuckDB Substrait frontend | PyTorch backend | 当前后端形态 |
| --- | --- | --- | --- | --- |
| Q1-Q22 | yes | 部分；DuckDB 1.2.x exporter 仍阻塞若干复杂查询 | yes | DuckDB physical-plan interpreter v1 by default |
| Q6 `--compressed-masks` | yes | yes | yes | explicit compressed-mask primitive experiment |

Strict Substrait 的 blocked 是 DuckDB exporter 覆盖限制，不是 PyTorch backend fallback。

`tpch_torch/physical_coverage.py` 提供 physical-only coverage probe：对 TPC-H 直接调用 `execute_physical_plan()`，因此可以在不经过 graph recipe dispatch 的情况下衡量迁移进度。

## 8. 冷/热计时方法

`tpch_torch/benchmark.py` 计时的是与 `tpch-torch-run` 同一条端到端路径：

```text
SQL text -> run_sql_with_frontend() -> compile_tqp_plan() -> PyTorchBackend.execute() -> tensor executor -> materialized result rows
```

冷查询：每个样本新建 DuckDB connection，运行完整 frontend + backend + materialization，然后关闭连接。热查询：复用一个 DuckDB connection，先跑 warmup，再记录 hot samples。CUDA 计时在每个样本前后调用 `torch.cuda.synchronize()`。

## 9. 验证命令

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q
timeout 60 /work/torch-query-gpu/.venv/bin/python -m compileall -q tpch_torch scripts
```
