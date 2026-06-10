# TQP / TQEx / TQP++ / CoddSpeed paper notes

This directory stores papers and source notes for the tensor-query-processing line
that motivates this repository's DuckDB → Substrait → PyTorch/GPU execution path.
Only publicly reachable PDFs are committed here. Closed, blocked, or unavailable
PDFs are recorded explicitly rather than replaced with fake files.

## Download status

| Short name | Title | Venue / date | Source | Local PDF | Status |
| --- | --- | --- | --- | --- | --- |
| TQP | Query Processing on Tensor Computation Runtimes | PVLDB 15(11), July 2022 | DOI: <https://doi.org/10.14778/3551793.3551833>; arXiv PDF: <https://arxiv.org/pdf/2203.01877> | `tqp-query-processing-on-tensor-computation-runtimes.pdf` | Downloaded from arXiv on 2026-06-09. |
| TQEx | TQEx: Tensor-based Query Engine Enhanced by Bridging the Gap | Proc. ACM Manag. Data, Dec. 2025 | DOI: <https://doi.org/10.1145/3769835>; ACM PDF listed by metadata: <https://dl.acm.org/doi/pdf/10.1145/3769835> | — | Not downloaded: ACM endpoint returned HTTP 403/Cloudflare in this environment and OpenAlex marks the work closed OA. |
| TQP++ | TQP++: Bridging ML Compilers and Analytical Query Processing on GPUs | VLDB 2026 Industrial Track, Aug. 2026 | Microsoft Research page: <https://www.microsoft.com/en-us/research/publication/tqp-bridging-ml-compilers-and-analytical-query-processing-on-gpus/?lang=fr-ca>; preprint link: <https://cmt3.research.microsoft.com/api/VLDBInd2026/Files/522> | — | Not downloaded: public MSR page is reachable, but the preprint endpoint returned HTTP 403 in this environment. |
| CoddSpeed | CoddSpeed: Hardware Accelerated Query Processing in Microsoft Fabric | SIGMOD 2026 Industrial Track, May 2026 | Microsoft Research page: <https://www.microsoft.com/en-us/research/publication/coddspeed-hardware-accelerated-query-processing-in-microsoft-fabric/>; DOI: <https://doi.org/10.1145/3788853.3803077> | — | Not downloaded: no public PDF link was reachable from the MSR page; ACM PDF endpoint was HTTP 403/Cloudflare. |
| Compressed GPU SQL | GPU Acceleration of SQL Analytics on Compressed Data | arXiv v2, Sept. 2025 | arXiv PDF: <https://arxiv.org/pdf/2506.10092> | `gpu-acceleration-sql-analytics-compressed-data.pdf` | Downloaded from arXiv on 2026-06-10. |

## Bibliographic metadata

### TQP

- Authors: Dong He, Supun C. Nakandala, Dalitso Banda, Rathijit Sen,
  Karla Saur, Kwanghyun Park, Carlo Curino, Jesús Camacho-Rodríguez,
  Konstantinos Karanasos, Matteo Interlandi.
- DOI: <https://doi.org/10.14778/3551793.3551833>.
- Notes from the downloaded PDF: TQP transforms SQL queries into tensor
  programs and executes them on tensor computation runtimes such as PyTorch.
  It emphasizes avoiding data-dependent Python control flow and implementing
  relational operators as tensor routines.

### TQEx

- Authors from Crossref/OpenAlex metadata: Haitao Zhang, Ran Pang,
  Yuanyuan Zhu, Hao Zhang, Congli Gao, Ming Zhong, Jiawei Jiang, Tieyun Qian,
  Jeffrey Xu Yu.
- DOI: <https://doi.org/10.1145/3769835>.
- Local note: only metadata has been verified. Do not derive implementation
  details from TQEx until a readable PDF or authorized copy is available.

### TQP++

- Authors from the Microsoft Research page: Wei Cui, Peng Cheng, Carlo Curino,
  Rathijit Sen, Matteo Interlandi.
- Source page: <https://www.microsoft.com/en-us/research/publication/tqp-bridging-ml-compilers-and-analytical-query-processing-on-gpus/?lang=fr-ca>.
- Notes from the public MSR abstract: TQP++ positions the design as an
  ML-compiler-native analytical query processor; it integrates Antares,
  tiered GPU resource scheduling, map-reduce-oriented fusion, and a
  multi-gated execution graph. These notes come from the public abstract, not
  the unavailable preprint.

### CoddSpeed

- Authors from Crossref metadata include Matteo Interlandi, Nicolas Bruno,
  Brandon Haynes, Carlo Curino, Rathijit Sen, Yinan Li, Kaushik Rajan,
  Bailu Ding, Lukas M. Maas, Wei Cui, Kevin Gaffney, and Mingsheng Hong.
- DOI: <https://doi.org/10.1145/3788853.3803077>.
- Notes from the public MSR abstract: CoddSpeed describes a Microsoft Fabric
  hardware-accelerated analytics effort, including a GPU execution engine
  derived from TQP and data movement over NVLink and InfiniBand. These notes
  come from the public abstract, not an unavailable full paper.

### GPU Acceleration of SQL Analytics on Compressed Data

- Authors from the arXiv PDF: Zezhou Huang, Krystian Sakowski, Hans
  Lehnert, Wei Cui, Carlo Curino, Matteo Interlandi, Marius Dumitru,
  Rathijit Sen.
- arXiv: <https://arxiv.org/abs/2506.10092>; PDF:
  <https://arxiv.org/pdf/2506.10092>.
- Notes from the downloaded PDF: the paper extends TQP-style PyTorch
  tensor execution to lightweight-compressed data. It covers RLE, Index,
  Plain+Index, RLE+Index, bit-width reduction, dictionary encoding,
  encoded logical operations, alignment, group-by aggregation, joins, and
  appendix optimization rules.
