# Sirius-grade Frontend 演进说明

本文记录当前仓库向 Sirius 前端质量靠拢的实现状态。目标不是调用 Sirius 的 cuDF/RMM 后端，而是复用同样的数据库前端原则：让 DuckDB 负责 SQL 解析、绑定、优化和 schema 推导，把稳定的 typed metadata 交给 TQP/PyTorch backend。

## 1. 和 Sirius 的对齐点

Sirius 的关键做法是：

```text
SQL
  -> DuckDB parser / binder / optimizer
  -> 捕获 DuckDB LogicalOperator / prepared names / prepared types
  -> Sirius expression AST / physical plan
  -> GPU backend
```

本仓库当前 Python-only frontend 对齐到以下层次：

```text
SQL
  -> DuckDB parser JSON: json_serialize_sql(sql)
  -> DuckDB binder schema: DESCRIBE sql
  -> DuckDB physical JSON: EXPLAIN (FORMAT JSON) sql
  -> TQPOperatorGraph(output_schema, select_aliases, normalized metadata)
  -> PyTorch physical executor
```

区别是：Sirius 在 C++ extension 内直接处理 DuckDB `LogicalOperator` 和 bound expressions；本仓库仍使用 DuckDB Python API 可获得的 parser JSON / DESCRIBE / physical JSON。它已经去掉了 backend 末端 SQL alias regex，但还不是完整 bound-expression exporter。

## 2. 当前实现

### 2.1 输出 schema 来自 DuckDB binder

`tpch_torch/duckdb_plan_json.py` 新增：

```python
def describe_output_schema(con, sql) -> tuple[TQPOutputColumn, ...]:
    rows = con.execute(f"DESCRIBE {sql}").fetchall()
    return tuple(TQPOutputColumn(str(row[0]), str(row[1]), _nullable(row[2])) for row in rows)
```

`TQPOperatorGraph` 现在携带 frontend-bound schema：

```python
@dataclass(frozen=True)
class TQPOutputColumn:
    name: str
    type_name: str
    nullable: bool | None = None

@dataclass(frozen=True)
class TQPOperatorGraph:
    ...
    output_schema: tuple[TQPOutputColumn, ...] = ()
    select_aliases: Mapping[str, str] = field(default_factory=dict)
```

后端输出 rename 优先使用 `graph.output_names`，不再在正常 Sirius-like 路径下重复 `DESCRIBE`。

### 2.2 AS / SELECT alias 来自 DuckDB parser JSON

`tpch_torch/frontend/duckdb_ast.py` 使用 DuckDB 自带 parser：

```python
def serialize_sql_ast(con, sql) -> dict[str, Any]:
    raw = con.execute("select json_serialize_sql(?)", [sql]).fetchone()[0]
    return json.loads(str(raw))


def select_expressions_by_alias(con, sql) -> dict[str, str]:
    aliases = {}
    for select_node in _select_nodes(serialize_sql_ast(con, sql)):
        for expression in select_node.get("select_list") or ():
            alias = str(expression.get("alias") or "")
            if alias:
                aliases[alias] = render_expression(expression)
    return aliases
```

这替代了之前 backend 里的 SQL text regex。示例：

```sql
select a as x, b + 1 as y from t
```

frontend 生成：

```python
{"x": "a", "y": "(b + 1)"}
```

### 2.3 DuckDB JSON metadata 规范化

`lower_duckdb_json_to_operator_graph()` 在 lowering 时保留 DuckDB raw key，同时增加 canonical key：

| DuckDB raw key | Canonical key |
| --- | --- |
| `Table` | `table` |
| `Type` | `scan_type` |
| `Projections` | `projections` |
| `Filters` | `filters` |
| `Aggregates` | `aggregates` |
| `Groups` | `groups` |
| `Order By` | `order_by` |
| `Conditions` | `conditions` |
| `Expression` | `expression` |
| `Estimated Cardinality` | `estimated_cardinality` |

backend 统一通过 `tpch_torch/backend/physical_metadata.py` 读取，优先 canonical key，再兼容 raw key。

## 3. compile_sirius_plan 当前链路

```python
physical_plan_json = export_duckdb_physical_plan_json(con, sql)
operator_graph = lower_duckdb_json_to_operator_graph(
    sql,
    query_id,
    physical_plan_json,
    output_schema=describe_output_schema(con, sql),
    select_aliases=select_expressions_by_alias(con, sql),
)
```

这意味着 `AS`、输出列名、输出类型都在 frontend compile 阶段固定下来，backend 只消费 graph metadata。

## 4. 当前边界

已完成：

- DuckDB parser JSON 驱动的 alias extraction。
- DuckDB DESCRIBE 驱动的 output schema。
- graph 携带 `output_schema` / `select_aliases`。
- physical / batch / partitionable executor 优先消费 graph schema。
- DuckDB physical JSON metadata canonical 化。

仍未声称完成的部分：

- 还没有 C++ DuckDB extension 直接导出 optimized `LogicalOperator`。
- 还没有 DuckDB bound expression / `ColumnBinding` 级 IR。
- backend expression evaluator 仍消费 expression text；只是这些 text 现在来自 DuckDB parser AST renderer，而不是 backend regex。

因此当前状态是 Sirius-grade frontend 的 Python API 第一阶段：frontend 边界更清晰，SQL 字符串解析不再散落在 backend；完整对齐 Sirius 还需要 C++ logical-plan exporter。

## 5. Slot/SlotRef 引用统一

DuckDB physical JSON 里天然会混用两种引用：

- 列名：例如 scan/project 中的 `b`。
- child ordinal：例如 aggregate 中的 `sum_no_overflow(#0)`。

成熟数据库通常允许 physical plan 使用 ordinal，但 ordinal 不应该裸露在字符串里让后端猜。当前实现新增了 graph-level typed slot view：

```python
@dataclass(frozen=True)
class TQPSlot:
    slot_id: str      # 例如 n0_0.s0
    node_id: str
    ordinal: int
    name: str
    type_name: str | None
    aliases: tuple[str, ...]

@dataclass(frozen=True)
class TQPBoundExpression:
    raw: str          # DuckDB raw expression，例如 sum_no_overflow(#0)
    canonical: str    # frontend canonical expression，例如 (b + 1)
    refs: tuple[TQPSlotRef, ...]
    unresolved: tuple[str, ...]
    output_slot: TQPSlot | None
```

lowering 阶段会为每个 node 生成 `node.output_slots`，并在 metadata 里补充结构化字段：

| metadata 字段 | 含义 |
| --- | --- |
| `slot_projections` | projection expression 的输入 `SlotRef` 与输出 `TQPSlot` |
| `slot_aggregates` | aggregate argument 中 `#0/#1` 解析后的 `SlotRef` |
| `slot_groups` | group-by expression 的 `SlotRef` |
| `slot_conditions` | join condition 解析后的左右输入 slot refs |
| `output_slots` | 当前 node 的 typed output slots |

例如：

```sql
select sum(b) as total from t
```

DuckDB raw physical aggregate 仍是：

```text
sum_no_overflow(#0)
```

但 TQP graph 的 canonical slot view 是：

```python
TQPBoundExpression(
    raw="sum_no_overflow(#0)",
    canonical="sum_no_overflow(#0)",
    refs=(TQPSlotRef(slot_id="n0_0.s0", name="b", ordinal=0),),
    output_slot=TQPSlot(slot_id="n0.s0", name="total", type_name="HUGEINT"),
)
```

也就是说，raw DuckDB 字符串会保留作兼容和调试；新的 graph 语义层统一用 `TQPSlot` / `TQPSlotRef`。后续 executor 可以逐步从字符串解释迁移到 slot-bound expression evaluator。
