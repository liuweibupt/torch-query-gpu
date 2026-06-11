# 中文文档与架构图设计

**目标：** 把仓库文档整理成中文优先、结构清晰、可验证的入口，明确 SQL → DuckDB/Sirius-like Frontend → TQP IR → PyTorch/CUDA Backend 的真实链路，并补齐 Roadmap 中文版本与 Q1 分层图。

## 设计选择

采用“中文主 README + 中文专题文档 + 保留英文原文”的方案：

1. `README.md` 改为中文主入口，放项目状态、快速命令、架构总图、Q1 分层图、TPC-H 支持矩阵、性能计时入口和文档导航。
2. 新增 `README.zh.md`，作为中文 README 的稳定链接入口，指向根 README，避免内容复制漂移。
3. 保留 `docs/architecture.md` 和 `docs/operator-roadmap.md` 的英文内容；新增 `docs/architecture.zh.md` 与 `docs/operator-roadmap.zh.md`，提供中文架构说明和中文 Roadmap。
4. 用 Mermaid 图替代外部图片，保证 GitHub 可直接渲染，且不引入二进制资产。

## 内容边界

文档必须准确描述当前实现：

- 默认前端是 Sirius-like DuckDB planner admission，不是默认 Substrait。
- strict Substrait 仍保留为显式实验路径，只承诺 DuckDB 原生 exporter 能导出的查询。
- DuckDB validation 只做正确性对照，不作为 fallback 输出。
- PyTorch backend 当前覆盖 TPC-H Q1-Q22 模板与有限 generic SQL subset；不声称任意 SQL 都能执行。
- Q1 是当前 fast path：`PyTorchBackend` 对 `query_id == 1` 调 `fetch_lineitem_tensor_table()` 和 `execute_q1()`，用预编码低基数字典列、dense group id 和 `torch.bincount` 做聚合。

## 验证策略

文档改造后运行：

```bash
timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_packaging.py tests/test_runner_cli.py tests/test_benchmark.py -q
timeout 60 /work/torch-query-gpu/.venv/bin/python -m compileall -q tpch_torch scripts
```

合并回 main 前再执行全量 pytest 和 compileall。若远端 main 变化，先 `git fetch origin` 并 `git merge origin/main`，处理冲突后再提交/推送。
