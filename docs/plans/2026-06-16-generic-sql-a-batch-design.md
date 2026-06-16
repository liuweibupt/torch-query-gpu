# Generic SQL A Batch Design

**Goal:** 完成 Roadmap A 批次：Basic `HAVING`、Generic `CASE`、`ORDER BY ... LIMIT` tensor top-k path，推进任意 SQL 经 DuckDB physical plan lowering 到 PyTorch operator 的覆盖。

**Scope:** 只修改 DuckDB physical-plan interpreter 与相关测试/文档，不引入 DuckDB query-result fallback，不做大规模压缩存储或 scheduler 重构。

## Design

1. **HAVING**
   - DuckDB 通常会把 `HAVING` lowering 成 aggregate 后的 `FILTER` physical node。
   - 设计目标是让 `physical_expr.evaluate_expression()` 能在 aggregate 输出 `PhysicalTable` 上正确解析 aggregate alias / aggregate expression，例如 `sum(l_quantity) > 50`、`count(*) >= 2`。
   - 实现应复用现有 `matching_aggregate_alias` / alias lookup，不新增 SQL rewrite。

2. **Generic CASE**
   - 现有 `physical_expr.parse_case()` 只覆盖简单 searched CASE 形状。
   - 目标支持 DuckDB physical expression 中常见的 searched CASE：`CASE WHEN predicate THEN value [WHEN ... THEN ...] ELSE value END`，输出 tensor 用 `torch.where` 逐层合成。
   - 第一批保持 correctness-first：支持 numeric/string-dictionary predicate 与 numeric result；遇到未支持 result type 显式失败。

3. **ORDER BY ... LIMIT top-k**
   - 当前 `_execute_limit()` 对带 `Order By` 的 limit 复用 full sort。
   - 目标在单 key `ORDER BY` + `LIMIT` 下使用 `torch.topk` 产生候选 row indices，再 `gather` 输出；多 key 保留稳定 full sort，避免错误排序。
   - 对 ascending 用 `largest=False`，descending 用 `largest=True`。不静默改变 ties 语义：ties 只在测试不依赖稳定 tie order 的场景声明支持。

4. **Docs**
   - 更新 README/Roadmap checkbox，明确本批是 generic physical-plan interpreter 的覆盖增强。

## Validation

- TDD：先写失败测试，再实现。
- Targeted tests：新增/扩展 generic physical SQL tests，验证结果与 DuckDB baseline 一致。
- Full verification：`timeout 60 python -m compileall -q tpch_torch scripts`、`timeout 60 python -m pytest -q`。
