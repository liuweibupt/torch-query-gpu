# TQP / TQEx / TQP++ / CoddSpeed paper notes

This directory stores papers and source notes for the tensor-query-processing line
that motivates this repository's DuckDB/Sirius-like frontend → TQP IR → PyTorch/GPU
execution path. Public PDFs are committed when available; blocked or unavailable
PDFs are recorded explicitly rather than replaced with fake files.

## Reading guide

Recommended order for this repository:

1. **TQP**: establishes the core idea: SQL query processing on tensor computation runtimes.
2. **TQEx**: analyzes the gap between irregular SQL workloads and uniform tensor operations, then optimizes strings, joins, aggregates, and multi-XPU execution.
3. **GPU compressed SQL**: extends the TQP line to lightweight-compressed data.
4. **CoddSpeed**: moves from research prototype to Microsoft Fabric system integration with coprocessors, fallback, partitionable execution, and data movement.
5. **TQP++**: public note only; points toward ML-compiler-native query processing, fusion, scheduling, and runtime gating.

## Download status

| Short name | Title | Venue / date | Source | Local PDF | Reading note | Status |
| --- | --- | --- | --- | --- | --- | --- |
| TQP | Query Processing on Tensor Computation Runtimes | PVLDB 15(11), July 2022 | DOI: <https://doi.org/10.14778/3551793.3551833>; arXiv PDF: <https://arxiv.org/pdf/2203.01877> | `p2811-he.pdf`; legacy copy: `tqp-query-processing-on-tensor-computation-runtimes.pdf` | [`reading-notes/tqp-pvldb-2022.zh.md`](reading-notes/tqp-pvldb-2022.zh.md) | PDF available. `p2811-he.pdf` was uploaded on 2026-06-23; legacy arXiv copy was downloaded on 2026-06-09. |
| TQEx | TQEx: Tensor-based Query Engine Enhanced by Bridging the Gap | Proc. ACM Manag. Data / SIGMOD, Dec. 2025 | DOI: <https://doi.org/10.1145/3769835> | `3769835.pdf` | [`reading-notes/tqex-sigmod-2025.zh.md`](reading-notes/tqex-sigmod-2025.zh.md) | PDF uploaded on 2026-06-23. Earlier ACM download was blocked in this environment. |
| TQP++ | TQP++: Bridging ML Compilers and Analytical Query Processing on GPUs | VLDB 2026 Industrial Track, Aug. 2026 | Microsoft Research page: <https://www.microsoft.com/en-us/research/publication/tqp-bridging-ml-compilers-and-analytical-query-processing-on-gpus/?lang=fr-ca>; preprint link: <https://cmt3.research.microsoft.com/api/VLDBInd2026/Files/522> | — | — | Not downloaded: public MSR page is reachable, but the preprint endpoint returned HTTP 403 in this environment. |
| CoddSpeed | CoddSpeed: Hardware Accelerated Query Processing in Microsoft Fabric | SIGMOD Companion 2026, May 2026 | DOI: <https://doi.org/10.1145/3788853.3803077>; Microsoft Research page: <https://www.microsoft.com/en-us/research/publication/coddspeed-hardware-accelerated-query-processing-in-microsoft-fabric/> | `3788853.3803077.pdf` | [`reading-notes/coddspeed-sigmod-2026.zh.md`](reading-notes/coddspeed-sigmod-2026.zh.md) | PDF uploaded on 2026-06-23. Earlier ACM download was blocked in this environment. |
| Compressed GPU SQL | GPU Acceleration of SQL Analytics on Compressed Data | arXiv v2, Sept. 2025 | arXiv PDF: <https://arxiv.org/pdf/2506.10092> | `gpu-acceleration-sql-analytics-compressed-data.pdf` | — | Downloaded from arXiv on 2026-06-10. |

## Bibliographic metadata and repository relevance

### TQP

- Authors: Dong He, Supun C. Nakandala, Dalitso Banda, Rathijit Sen,
  Karla Saur, Kwanghyun Park, Carlo Curino, Jesús Camacho-Rodríguez,
  Konstantinos Karanasos, Matteo Interlandi.
- DOI: <https://doi.org/10.14778/3551793.3551833>.
- Local PDFs:
  - `p2811-he.pdf`: uploaded PVLDB copy.
  - `tqp-query-processing-on-tensor-computation-runtimes.pdf`: legacy arXiv copy.
- Key idea: transform SQL queries into tensor programs and execute them on
  tensor computation runtimes such as PyTorch. TQP emphasizes avoiding
  data-dependent Python control flow, using tensor-based columnar data, and
  implementing relational operators with existing tensor routines.
- Repository relevance: this is the main architectural baseline for
  `TQPOperatorGraph`, PyTorch tensor backend, and the goal of lowering SQL
  into reusable tensor primitives instead of query-specific Python scripts.

### TQEx

- Authors: Haitao Zhang, Ran Pang, Yuanyuan Zhu, Hao Zhang, Congli Gao,
  Ming Zhong, Jiawei Jiang, Tieyun Qian, Jeffrey Xu Yu.
- DOI: <https://doi.org/10.1145/3769835>.
- Local PDF: `3769835.pdf`.
- Key idea: TQP is portable but suffers from the gap between irregular SQL
  workloads and uniform tensor operations. TQEx bridges the gap with loop
  unrolling, early padding elimination, compact variable-length string storage,
  improved join and aggregate algorithms, and multi-XPU execution.
- Repository relevance: provides concrete operator-level TODOs for this project:
  variable-length string tensors, batch candidate generation for joins,
  skew-aware grouped aggregation, and backend-aware tensor algorithm selection.

### TQP++

- Authors from the Microsoft Research page: Wei Cui, Peng Cheng, Carlo Curino,
  Rathijit Sen, Matteo Interlandi.
- Source page: <https://www.microsoft.com/en-us/research/publication/tqp-bridging-ml-compilers-and-analytical-query-processing-on-gpus/?lang=fr-ca>.
- Notes from the public MSR abstract: TQP++ positions the design as an
  ML-compiler-native analytical query processor; it integrates Antares,
  tiered GPU resource scheduling, map-reduce-oriented fusion, and a
  multi-gated execution graph. These notes come from the public abstract, not
  the unavailable preprint.
- Repository relevance: points to the next stage after PyTorch eager execution:
  graph/compiler lowering, map-reduce fusion, scheduling, and runtime algorithm
  gating.

### CoddSpeed

- Authors include Matteo Interlandi, Nicolas Bruno, Brandon Haynes, Carlo
  Curino, Rathijit Sen, Yinan Li, Kaushik Rajan, Bailu Ding, Lukas M. Maas,
  Wei Cui, Kevin Gaffney, Mingsheng Hong, and many Microsoft co-authors.
- DOI: <https://doi.org/10.1145/3788853.3803077>.
- Local PDF: `3788853.3803077.pdf`.
- Key idea: CoddSpeed integrates hardware-accelerated analytics into Microsoft
  Fabric through a coprocessor abstraction layer (CAL), partitionable execution,
  fragment-level fallback, and a data movement service layer (DAL). Its most
  mature implementation is a GPU execution engine derived from TQP.
- Repository relevance: provides system-engineering guidance beyond operator
  kernels: capability negotiation, explicit fallback, Substrait/open plan
  boundaries, host/coprocessor API, chunked execution under accelerator memory
  limits, and first-class data movement.

### GPU Acceleration of SQL Analytics on Compressed Data

- Authors from the arXiv PDF: Zezhou Huang, Krystian Sakowski, Hans
  Lehnert, Wei Cui, Carlo Curino, Matteo Interlandi, Marius Dumitru,
  Rathijit Sen.
- arXiv: <https://arxiv.org/abs/2506.10092>; PDF:
  <https://arxiv.org/pdf/2506.10092>.
- Local PDF: `gpu-acceleration-sql-analytics-compressed-data.pdf`.
- Key idea: extends TQP-style PyTorch tensor execution to lightweight-compressed
  data. It covers RLE, Index, Plain+Index, RLE+Index, bit-width reduction,
  dictionary encoding, encoded logical operations, alignment, group-by
  aggregation, joins, and appendix optimization rules.
- Repository relevance: motivates the existing compressed mask/RLE aggregate
  primitives and future compressed storage metadata.

## Cross-paper synthesis

| Theme | TQP | TQEx | CoddSpeed | Repository implication |
| --- | --- | --- | --- | --- |
| Core abstraction | SQL → tensor program on TCRs. | Bridge SQL irregularity and tensor uniformity. | Host engine → coprocessor fragment API. | Keep SQL lowering explicit; avoid query-id scripts. |
| Operator design | Sort/hash join, aggregation, expressions with tensor ops. | Loop unrolling, padding elimination, compact hash/index join, skew-aware aggregate. | Hardened TQP-derived GPU engine plus PK-FK and hash optimizations. | Expand physical nodes into reusable tensor primitives and fusion candidates. |
| Runtime | PyTorch eager, TorchScript, ONNX, TVM. | TCR execution across XPUs. | Fabric host + CAL/DAL + GPU/FPGA/ASIC. | Separate DB graph, tensor backend, and hardware capability registry. |
| Memory/data movement | Basic tensor conversion and device movement. | Reduce padding and intermediate overhead. | Data movement as a service, cache/shuffle/tiering/zero-copy. | Add operator memory accounting, resident layout, H2D/D2H timing, and explicit chunking. |
| Portability | Use existing TCR APIs, no custom tensor ops. | Multi-XPU portability with better SQL/tensor mapping. | Hardware-agnostic coprocessor API. | Support PyTorch-compatible domestic accelerators through capability checks. |

## Maintenance notes

- Keep paper PDFs in this directory.
- Put detailed Chinese reading notes in `docs/papers/reading-notes/`.
- If a previously blocked PDF becomes available, add it to the table above with
  the exact local filename and access/update date.
- Do not add empty placeholder PDFs for blocked sources.
