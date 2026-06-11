# TPC-H 模板迁移到显式 TQP Operator Graph 设计

**目标：** 让 TPC-H 查询不再由 `PyTorchBackend` 直接按 `query_id` 调用 `qXX.py` 整查询模板，而是先从 SQL/DuckDB plan lowering 出一个显式 `TQPOperatorGraph`，再由 PyTorch graph executor 执行。

## 背景与当前问题

当前默认链路已经是 SQL → DuckDB/Sirius-like frontend → `TQPPlan` → PyTorch backend，但 TPC-H 主路径仍然是 case-specific：`identify_tpch_query()` 识别 Q1-Q22 后，`PyTorchBackend.execute()` 直接 import `tpch_torch.queries.qXX` 并执行整查询函数。这样保证了 Q1-Q22 能跑，却不是完整的 SQL compiler 自动 lowering。

DuckDB 1.2.x 可以对 Q1-Q22 全部输出 `EXPLAIN (FORMAT JSON)`。这些 JSON physical plans 至少覆盖：`SEQ_SCAN`、`FILTER`、`PROJECTION`、`HASH_JOIN`、`HASH_GROUP_BY`、`PERFECT_HASH_GROUP_BY`、`UNGROUPED_AGGREGATE`、`ORDER_BY`、`TOP_N`、`RIGHT_DELIM_JOIN`、`CTE`、`CTE_SCAN`、`DELIM_SCAN` 等。这可以作为第一版 compiler lowering 的计划来源。

## 设计选择

采用分阶段迁移：

1. **先引入强制 graph 边界。** `TQPPlan` 新增 `operator_graph`，Sirius-like frontend 必须为所有 TPC-H Q1-Q22 生成 graph。`PyTorchBackend` 对 TPC-H 只接受 graph，不再直接从 `query_id` 跳到模板 executor。
2. **第一版 graph 是显式算子 DAG，不是静默 fallback。** graph 由 `TQPOperatorNode` 组成，记录 DuckDB JSON plan node、operator kind、children、query_id、source SQL。执行端必须经过 `PyTorchGraphExecutor.execute(graph)`。
3. **先保证端到端行为不倒退。** 第一批 executor 支持单表 Scan/Filter/Project/Aggregate/Sort/Limit 的通用执行，覆盖 Q1/Q6 以及 generic 单表聚合；复杂 TPC-H 的 graph executor 初期会通过显式 `CompiledTPCHGraph` operator 作为兼容节点承载尚未拆完的复杂子图，并在 Roadmap 中逐步替换为 Join/Subquery/CTE 等真实节点。
4. **禁止模板旁路。** 即使兼容复杂 TPC-H，调用点也必须是 graph executor 内的显式 operator node，`PyTorchBackend` 不允许直接 import qXX。这样后续可以逐个替换 node executor，而不是散落在 backend dispatch 中。

## 第一批实现范围

- 新增 `tpch_torch/operator_graph.py`：定义 `TQPOperatorGraph`、`TQPOperatorNode`、`OperatorKind`。
- 新增 `tpch_torch/duckdb_plan_json.py`：通过 `EXPLAIN (FORMAT JSON)` 导出结构化 physical plan。
- Sirius-like frontend：为 Q1-Q22 和可 generic 执行 SQL 生成 `operator_graph`。
- Backend：优先执行 `operator_graph`；删除 TPC-H 直接 qXX 分发路径。
- Graph executor：
  - 对 Q1/Q6 走通用 graph executor primitives。
  - 对复杂 Q2-Q22 先走显式 `compiled_tpch` graph node，确保没有 backend 旁路，并在文档/TODO 中标注为待拆分子图。
- 测试：
  - 证明 Q1/Q6 通过 graph lowering 运行。
  - 证明 Q1-Q22 的 `TQPPlan.operator_graph` 存在。
  - 证明 backend 不再直接通过 `_EXECUTOR_BY_QUERY` 调 qXX。
  - 验证 Q1-Q22 仍能 correctness validation。

## 后续迁移路线

- Batch A：完全通用 single-table graph，覆盖 Q1、Q6、generic aggregate/filter/order/limit。
- Batch B：实现 hash/lookup join node，替换 Q3/Q5/Q10/Q12/Q14/Q19 等 join+aggregate 查询。
- Batch C：实现 semi/anti/delim join、CTE、subquery、distinct aggregate，替换 Q2/Q4/Q15/Q16/Q17/Q20/Q21/Q22。
- Batch D：引入 optimizer/cost model、compressed column operators、fusion/scheduling。

## 不做的事情

- 不伪造 DuckDB 结果作为 PyTorch 输出。
- 不在 `PyTorchBackend` 中保留隐藏模板 fallback。
- 不声称第一批已经完全拆完所有复杂 TPC-H 内部 join/subquery 算子；复杂查询先由显式 graph node 承载，后续持续拆分。
