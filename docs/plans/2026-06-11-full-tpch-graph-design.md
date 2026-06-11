# 全量 TPC-H 通用 Graph Node 拆分设计

**目标：** 移除 Q2-Q22 的 complex compatibility executor，使 TPC-H 22 个查询都按照同一条流程运行：SQL → DuckDB JSON physical plan → `TQPOperatorGraph` → 通用 graph nodes → PyTorch tensor ops。

## 当前状态

当前 main 已完成第一版 graph boundary：所有 TPC-H 都会生成 `TQPOperatorGraph`，Q1/Q6 已由 graph primitives 执行。但 Q2-Q22 的复杂 join/subquery/CTE 子图仍在 `PyTorchGraphExecutor._execute_complex_tpch_graph()` 中调用旧 `qXX.py` 查询实现。

## 现实约束

DuckDB JSON physical plan 包含大量表达式字符串与内部投影编号，例如 `#0`、`__internal_compress_*`、`sum(#3)`、`CASE WHEN`、`RIGHT_DELIM_JOIN`。一次性实现完整 DuckDB physical-plan interpreter 风险很高。为保持正确性和可验证性，本次采用“显式 graph node executor + 分批替换”的方式：

1. 先加硬测试禁止 `complex compatibility executor`。
2. 为 graph executor 增加通用 node execution 上下文：`GraphTable`、column aliases、row ids、projection slots。
3. 第一批覆盖常规 relational pipeline：`SEQ_SCAN`、`FILTER`、`PROJECTION`、`HASH_JOIN`、`HASH_GROUP_BY`/`PERFECT_HASH_GROUP_BY`、`UNGROUPED_AGGREGATE`、`ORDER_BY`、`TOP_N`。
4. 第二批覆盖 TPC-H 复杂节点：`CTE`/`CTE_SCAN`、`RIGHT_DELIM_JOIN`、`DELIM_SCAN`、`COLUMN_DATA_SCAN`、semi/anti joins、nested-loop scalar subquery。
5. 遇到 DuckDB 表达式字符串过于复杂时，先实现可解释表达式子集；不允许调用 DuckDB result rows 或旧 qXX 整查询模板。

## 成功标准

- `PyTorchGraphExecutor` 不再包含 `_execute_complex_tpch_graph()` 或 `_EXECUTOR_BY_QUERY` 兼容入口。
- 新测试 monkeypatch 所有 `tpch_torch.queries.qXX.execute_qXX` 后，Q1-Q22 validation 仍能通过。
- `tpch-torch-validate --queries all --frontend sirius` 通过。
- README/架构/Roadmap 说明所有 TPC-H 已通过通用 graph node executor。

## 风险处理

如果某个 DuckDB JSON node/表达式在 3 次 root-cause debugging 后仍无法正确解释，必须保留显式 failing test 和 unsupported node 报错，不允许静默 fallback。
