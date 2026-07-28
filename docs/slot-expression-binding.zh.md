# TQP Slot 引用统一与前端表达式解析

本文解释当前 `TQPOperatorGraph` 如何把 DuckDB physical plan 中混杂的 **列名** 与 **`#0/#1` child ordinal** 统一成稳定的 slot reference，并说明表达式解析如何尽量在 frontend/lowering 阶段完成，而不是留给 backend 在执行末端猜。

## 1. 背景：为什么不能继续混用列名和序号

DuckDB `EXPLAIN (FORMAT JSON)` 的 physical plan 是执行器视角，会自然出现两类引用：

```text
SEQ_SCAN / PROJECTION:  b                  # 列名
UNGROUPED_AGGREGATE:   sum_no_overflow(#0) # child output ordinal
HASH_JOIN:             a = c               # 条件文本里的列名
```

例如：

```sql
select sum(b) as total from t
```

DuckDB physical JSON 中 aggregate 通常是：

```text
sum_no_overflow(#0)
```

这里的 `#0` 不是最终输出列名，也不是 base table 列名，而是 **aggregate child 的第 0 个输出 slot**。如果 backend 后续一会儿按列名找，一会儿按 `#0` 找，会带来几个问题：

- 自连接 / 同名列时，列名可能不唯一。
- projection 改名后，raw expression 和最终 output name 不是同一个概念。
- `#0` 只有在“当前 child output order”里才有意义，脱离 parent/child 上下文就不可解释。
- 表达式 evaluator 需要到处处理字符串、别名、序号、列名，难以优化。

因此现在不做简单的“把 `#0` 替换成列名字符串”。更稳定的做法是：

```text
#0  -> child output slot -> TQPSlotRef(slot_id=..., name=..., ordinal=...)
列名 -> matching slot aliases -> TQPSlotRef(slot_id=..., name=..., ordinal=...)
```

对人可读时看 `name`，对执行和优化用稳定的 `slot_id`。

## 2. 总体流程

当前 frontend/lowering 流程如下：

```text
SQL
  -> DuckDB json_serialize_sql(sql)
       解析 SELECT alias 与 parser AST 表达式
  -> DuckDB DESCRIBE sql
       获取 output names / output types / nullable
  -> DuckDB EXPLAIN (FORMAT JSON) sql
       获取 physical operator tree
  -> DuckDB catalog schema
       获取 scan table column types，例如 DECIMAL(10,2)
  -> lower_duckdb_json_to_operator_graph(...)
       生成 normalized metadata
       生成 TQPSlot / TQPSlotRef / TQPBoundExpression / TQPExprNode
  -> PyTorch backend
```

对应关键文件：

| 文件 | 职责 |
| --- | --- |
| `tpch_torch/frontend/duckdb_ast.py` | 用 DuckDB parser JSON 提取 SELECT alias 和 canonical expression。 |
| `tpch_torch/duckdb_plan_json.py` | 导出 DuckDB physical JSON，获取 DESCRIBE output schema 与 scan table schema，lowering 成 `TQPOperatorGraph`。 |
| `tpch_torch/operator_refs.py` | 定义 `TQPSlot`、`TQPSlotRef`、`TQPBoundExpression`、`TQPExprNode`。 |
| `tpch_torch/operator_slot_binding.py` | 根据 parent/child 关系把列名和 `#N` 绑定到 slot refs。 |
| `tpch_torch/operator_expression_binding.py` | 把 canonical expression 解析成 slot-aware expression AST。 |

## 3. 核心数据结构

### 3.1 TQPSlot

`TQPSlot` 表示某个 operator node 的一个输出位置：

```python
@dataclass(frozen=True)
class TQPSlot:
    slot_id: str          # 例如 n0_0.s0
    node_id: str          # producer node id
    ordinal: int          # 当前 node output ordinal
    name: str             # 人可读名称，例如 b / total
    type_name: str | None
    aliases: tuple[str, ...]
    origin_slot_id: str | None
```

每个 node 都会携带：

```python
node.output_slots
```

### 3.2 TQPSlotRef

`TQPSlotRef` 是表达式里对某个输入 slot 的引用：

```python
@dataclass(frozen=True)
class TQPSlotRef:
    slot_id: str
    node_id: str
    ordinal: int
    name: str
```

### 3.3 TQPBoundExpression

`TQPBoundExpression` 保存 raw expression、canonical expression、slot refs 和解析后的 AST：

```python
@dataclass(frozen=True)
class TQPBoundExpression:
    raw: str
    canonical: str
    refs: tuple[TQPSlotRef, ...]
    unresolved: tuple[str, ...]
    output_slot: TQPSlot | None
    expression: TQPExprNode | None
```

### 3.4 TQPExprNode

`TQPExprNode` 是第一版轻量表达式 AST：

```python
@dataclass(frozen=True)
class TQPExprNode:
    kind: str     # slot_ref / literal / binary / call / cast / logical / unknown
    value: Any
    children: tuple[TQPExprNode, ...]
    ref: TQPSlotRef | None
```

当前支持：

- `slot_ref`
- `literal`
- binary arithmetic：`+ - * /`
- comparison：`= != <> > >= < <=`
- logical：`AND / OR / NOT`
- function call：如 `sum_no_overflow(#0)`
- `CAST(expr AS TYPE)`
- `EXTRACT(year FROM expr)`

## 4. `#0` 如何替换成稳定名字/slot

### 4.1 child 先生成 output slots

lowering 是递归执行的，child node 会先被 lower。比如：

```sql
select sum(b) as total from t
```

scan/projection child 先产生 slot：

```python
TQPSlot(
    slot_id="n0_0.s0",
    node_id="n0_0",
    ordinal=0,
    name="b",
    aliases=("b",),
)
```

### 4.2 parent 解析 `#0`

aggregate raw expression 是：

```text
sum_no_overflow(#0)
```

slot binding 会在 parent node 中用 `#0` 查询 child output slots：

```python
#0 -> child_slots[0] -> TQPSlot(slot_id="n0_0.s0", name="b")
```

最终得到：

```python
TQPBoundExpression(
    raw="sum_no_overflow(#0)",
    canonical="sum_no_overflow(#0)",
    refs=(TQPSlotRef(slot_id="n0_0.s0", name="b", ordinal=0),),
    output_slot=TQPSlot(slot_id="n0.s0", name="total", type_name="HUGEINT"),
)
```

注意：raw string 仍保留作兼容和调试；新的 graph 语义层已经不依赖裸 `#0`。

## 5. 列名如何绑定到 slot

scan slot 会带 aliases：

```python
TQPSlot(
    slot_id="n0_0.s1",
    name="b",
    type_name="DECIMAL(10,2)",
    aliases=("b", "t.b"),
)
```

当 expression 出现：

```text
b + 1
```

slot binding 会用 aliases 匹配：

```text
b -> TQPSlotRef(slot_id="n0_0.s1", name="b")
```

如果多个 slot 都匹配同一个名字，当前会记录到 `unresolved`，避免把歧义 silently 绑定错。长期目标是接 DuckDB bound `ColumnBinding`，这样自连接 / 同名列会更稳。

## 6. 表达式解析如何在 frontend/lowering 解决

### 6.1 SELECT alias 来源：DuckDB parser AST

以前 backend 会在执行末端解析 SQL 字符串找 alias。现在 alias 提取在 frontend：

```python
# tpch_torch/frontend/duckdb_ast.py
raw = con.execute("select json_serialize_sql(?)", [sql]).fetchone()[0]
parsed = json.loads(str(raw))
```

例如：

```sql
select a as x, b + 1 as y from t
```

DuckDB parser AST 会告诉我们：

```python
select_aliases = {
    "x": "a",
    "y": "(b + 1)",
}
```

### 6.2 output schema 来源：DuckDB DESCRIBE

输出列名、类型、nullable 在 frontend compile 阶段固定：

```python
# tpch_torch/duckdb_plan_json.py
def describe_output_schema(con, sql):
    rows = con.execute(f"DESCRIBE {sql}").fetchall()
    return tuple(TQPOutputColumn(str(row[0]), str(row[1]), _nullable(row[2])) for row in rows)
```

并写入：

```python
TQPOperatorGraph.output_schema
```

### 6.3 scan table schema 来源：DuckDB catalog

最终输出 schema 只描述 SELECT 结果；scan node 还需要知道 base table column type，特别是 `DECIMAL(p,s)` 不能退化成普通 `INT64`。

现在 `compile_sirius_plan()` 会在拿到 physical JSON 后收集 scan table，并通过 DuckDB catalog 查询表结构：

```python
table_schemas = describe_scan_table_schemas(con, physical_plan_json)
```

lowering 时会写入：

```python
scan.metadata["scan_output_types"]      # 例如 {"amount": "DECIMAL(10,2)"}
scan.metadata["scan_output_nullable"]
scan.output_slots[0].type_name          # 例如 "DECIMAL(10,2)"
```

这样 expression binding 里的 slot 不再只有列名/序号，也能携带 DuckDB logical type。

### 6.4 physical JSON 来源：DuckDB EXPLAIN

physical JSON 仍用于 operator tree：

```python
physical_plan_json = export_duckdb_physical_plan_json(con, sql)
```

但 lowering 不再只保存 raw `extra_info`，还会补充：

```python
slot_projections
slot_aggregates
slot_groups
slot_conditions
output_slots
```

### 6.5 expression AST 示例

Projection：

```sql
select b + 1 as y from t
```

变成：

```python
TQPExprNode(
    kind="binary",
    value="+",
    children=(
        TQPExprNode(kind="slot_ref", ref=SlotRef(name="b")),
        TQPExprNode(kind="literal", value=1),
    ),
)
```

Aggregate：

```sql
select sum(b) as total from t
```

变成：

```python
TQPExprNode(
    kind="call",
    value="sum_no_overflow",
    children=(TQPExprNode(kind="slot_ref", ref=SlotRef(name="b")),),
)
```

Join condition：

```sql
select a, c from t join u on t.a = u.c
```

变成：

```python
TQPExprNode(
    kind="binary",
    value="=",
    children=(SlotRef(name="a"), SlotRef(name="c")),
)
```

DECIMAL literal：

```sql
select amount + 0.05::decimal(3,2) as adjusted from t
```

变成：

```python
TQPExprNode(
    kind="binary",
    value="+",
    children=(
        TQPExprNode(kind="slot_ref", ref=SlotRef(name="amount")),
        TQPExprNode(kind="literal", value=Decimal("0.05")),
    ),
)
```

这里不是 `float(0.05)`，而是 `Decimal("0.05")`。后续 lowering 到 `TensorRecordBatch` projection plan 时，会按 `int64 + scale` materialize。

## 7. 当前兼容策略

当前 executor 仍主要使用 raw/canonical string 执行，原因是 TPC-H Q1-Q22 已经在字符串 evaluator 上跑通，直接替换风险较大。因此目前策略是：

```text
保留 raw metadata 兼容现有 executor
同时在 graph 上生成 slot-bound expression view
后续逐个算子迁移到 TQPExprNode evaluator
```

也就是说，现在的稳定边界是：

```text
TQPOperatorGraph 语义层：TQPSlot / TQPSlotRef / TQPExprNode
Executor 兼容层：raw DuckDB expression string
```

这比“把 `#0` 字符串替换成列名字符串”更稳，因为 slot_id 不受 alias、同名列、自连接影响。

## 8. 验证覆盖

新增/更新测试覆盖：

- `tests/test_duckdb_frontend_ast.py`
  - DuckDB parser AST alias extraction。
- `tests/test_duckdb_plan_json.py`
  - output schema。
  - scan table schema / DECIMAL slot type。
  - aggregate `#0` 到 child slot ref。
  - projection expression AST。
  - DECIMAL literal AST。
  - join condition expression AST。
- `tests/test_operator_graph.py`
  - graph 携带 output schema、aliases、output slots。

全量验证由 CI/本地 pytest 回归覆盖：

```text
timeout 60 python -m pytest -q
# 378 passed, 2 skipped
```
