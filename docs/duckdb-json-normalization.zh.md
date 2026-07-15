# DuckDB JSON Physical Plan Normalization

本分支把 DuckDB `EXPLAIN (FORMAT JSON)` 的 raw `extra_info` 统一为 backend 可稳定读取的 metadata，同时保留 DuckDB 原始字段，避免不同算子间属性形态“乱蹦”。

## 为什么需要统一

DuckDB JSON 的同一个逻辑字段可能因 arity 不同而返回不同形态：

- `Projections`: 可能是字符串，也可能是数组。
- `Aggregates`: 可能是字符串，也可能是数组。
- `Order By` / `Groups` / `Filters` 同理。
- `Estimated Cardinality` 是字符串，但 backend 需要数值语义。

以前 backend 需要到处写 `_metadata_list(node, "Projections")` 之类的适配逻辑。现在 lowering 阶段会额外写入稳定 snake_case 字段：

| DuckDB raw key | Canonical key | 类型 |
| --- | --- | --- |
| `Table` | `table` | `str` |
| `Type` | `scan_type` | `str` |
| `Projections` | `projections` | `tuple[str, ...]` |
| `Filters` | `filters` | `tuple[str, ...]` |
| `Aggregates` | `aggregates` | `tuple[str, ...]` |
| `Groups` | `groups` | `tuple[str, ...]` |
| `Order By` | `order_by` | `tuple[str, ...]` |
| `Conditions` | `conditions` | `tuple[str, ...]` |
| `Expressions` | `expressions` | `tuple[str, ...]` |
| `Expression` | `expression` | `str` |
| `Estimated Cardinality` | `estimated_cardinality` | `int | None` |

兼容性策略：原始 DuckDB key 仍保留，现有 executor 不需要一次性迁移；新增代码通过 `tpch_torch.backend.physical_metadata` 优先读 canonical key，再兼容 raw key。

## AS / 输出 schema 的归属

`AS` / final output names 不应在 physical backend 后期通过 SQL 字符串或重复 `DESCRIBE` 恢复。Sirius-like frontend 现在在 compile 阶段执行：

```text
DuckDB DESCRIBE original SQL -> TQPOperatorGraph.output_names
```

因此 physical executor 最终 rename rows 时优先使用：

```python
graph.output_names
```

只有老测试或手工构造 graph 没有 `output_names` 时，才保留兼容 fallback。

## SELECT alias expression map

目前 DuckDB physical JSON 对某些 final projection 只暴露 alias 名，不保留完整源表达式，例如：

```sql
select a as x, b + 1 as y from t
```

DuckDB JSON 可能只给 final projection `x, y`。为了不破坏当前表达式执行，frontend 仍会构造 `graph.select_aliases`，但这个动作已经从 physical executor 构造阶段前移到了 Sirius frontend compile 阶段。后续更完整的方案应接 DuckDB binder / richer JSON，而不是在 backend 内部解析 SQL。
