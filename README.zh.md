# torch-query-gpu 中文入口

中文主 README 见仓库根目录 [`README.md`](README.md)。它包含：

- 当前 DuckDB/Sirius-like → TQP IR → PyTorch/CUDA 架构图。
- Q1 分层实现图与关键代码片段。
- Generic SQL 与 DuckDB physical-plan interpreter 当前支持边界。
- TPC-H Q1-Q22 支持矩阵；其中 Q1-Q22 已迁入 physical-plan interpreter。
- Q1 SF=1 冷/热性能对比与 benchmark 方法。
- 当前算子优化状态：tensor join index、sorted-unique build fast path、SEMI/ANTI membership probe、sorted group-by fast path、static dictionary encoding、membership mask、alias 去重 gather/filter、Q1 resident tensor hot path、RLE aggregate primitives。
- 中文 Roadmap/TODO 链接。

文档导航：

- 中文架构说明：[`docs/architecture.zh.md`](docs/architecture.zh.md)
- 中文 Roadmap：[`docs/operator-roadmap.zh.md`](docs/operator-roadmap.zh.md)
- 英文架构说明：[`docs/architecture.md`](docs/architecture.md)
- 英文 Roadmap：[`docs/operator-roadmap.md`](docs/operator-roadmap.md)
