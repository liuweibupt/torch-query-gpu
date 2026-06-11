# TQP-line Operator and Optimization TODO

中文执行版见 [`docs/operator-roadmap.zh.md`](operator-roadmap.zh.md).

This roadmap tracks the operator work needed to evolve this repository from a
correctness-first TPC-H-on-PyTorch prototype toward the TQP → TQEx → TQP++ →
CoddSpeed line, including direct GPU execution on lightweight-compressed data.

## Source status

| Source | Local status | Extraction confidence |
| --- | --- | --- |
| TQP, *Query Processing on Tensor Computation Runtimes* | `docs/papers/tqp-query-processing-on-tensor-computation-runtimes.pdf` | Full text extracted. The PDF has no separate appendix section. |
| TQEx, *Tensor-based Query Engine Enhanced by Bridging the Gap* | ACM PDF is blocked by HTTP 403/Cloudflare in this environment. Crossref/DOI metadata and abstract are reachable. | Abstract-derived only; full paper and appendices still pending. |
| TQP++, *Bridging ML Compilers and Analytical Query Processing on GPUs* | Microsoft Research page is reachable; preprint endpoint is blocked. | Abstract/page-derived only; full paper and appendices still pending. |
| CoddSpeed, *Hardware Accelerated Query Processing in Microsoft Fabric* | Microsoft Research/DOI metadata are reachable; ACM PDF is blocked. | Abstract/page-derived only; full paper and appendices still pending. |
| *GPU Acceleration of SQL Analytics on Compressed Data* | `docs/papers/gpu-acceleration-sql-analytics-compressed-data.pdf` | Full arXiv v2 text and appendices extracted. |

**Policy:** items below are marked **verified** when they come from local full
text. Items marked **abstract-derived** are intentionally not treated as complete
until a readable full paper/appendix is available.

## Current repository baseline

- Frontend path: original SQL → DuckDB/Sirius-like planner admission → `TQPPlan`.
- Experimental strict frontend: original SQL → DuckDB native Substrait JSON →
  `TQPPlan`; no fabricated JSON and no frontend fallback.
- Backend path: `TQPPlan` → PyTorch operators on CPU/CUDA.
- Current TPC-H status: Q1-Q22 are lowered through DuckDB JSON physical plans into `TQPOperatorGraph` on the Sirius-like path. Q1/Q6 execute with real graph primitives; complex Q2-Q22 subgraphs still use explicit compatibility execution while generic join/subquery/CTE nodes are added. DuckDB native Substrait remains limited by exporter coverage for several queries.
- Current generic SQL status: single-table projection/filter/aggregate subset.

## Operator inventory from TQP (**verified**)

### Tensor operation families to keep exposed

- Creation: `from_numpy`, `zeros`, `ones`, `empty`, `fill`, `arange`,
  `zeros_like`, `ones_like`.
- Indexing/slicing: tensor indexing, `index_select`, `masked_select`, `narrow`.
- Reorganization: `reshape`, `view`, `squeeze`, `gather`, `scatter`, `sort`.
- Comparison: `eq`, `lt`, `gt`, `le`, `ge`, `isnan`, `where`, `bucketize`.
- Arithmetic/logical: `add`, `mul`, `div`, `sub`, `fmod`, `remainder`,
  `logical_and`, `logical_or`, negation, shift operations.
- Tensor joining/stacking: `cat`, `stack`.
- Reductions: `sum`, `max`, `min`, `mean`, `scatter_add`, `scatter_min`,
  `scatter_max`, `scatter_mean`, `all`, `any`, `bincount`, `histc`, `nonzero`,
  `unique`, `unique_consecutive`.

### Relational operators and SQL features

- Selection/filter with bitmap masks and index-based selection.
- Projection and expression evaluation through post-order expression-tree DFS.
- Sort.
- Group-by aggregation, including sort-based grouping.
- Natural joins with both sort-based and hash-based algorithms.
- Non-equi joins.
- Left outer joins.
- Left semi joins.
- Left anti joins.
- Comparison and arithmetic expressions.
- Date functions.
- `IN`, `CASE`, and `LIKE` expressions.
- Aggregate expressions: `SUM`, `AVG`, `MIN`, `MAX`, `COUNT`.
- Distinct and non-distinct aggregate variants.
- NULL handling.
- Scalar, nested, and correlated subqueries.
- Prediction UDFs / ML model operators in mixed SQL+ML plans.

### TQP algorithm TODO

- [x] Columnar tensor table representation for plain columns.
- [x] Bitmap-style filter masks for current TPC-H executors.
- [x] First Batch 2 step: boolean filter tree for `AND`, `OR`, and `NOT` in generic SQL.
- [x] First Batch 2 step: generic bitmap selection for comparisons, `IN`, and `LIKE`.
- [ ] Generic projection operator with expression-tree lowering.
- [x] First Batch 2 step: generic stable multi-key `ORDER BY` with `ASC`/`DESC`.
- [ ] Generic sort-based equi-join using sort, histograms, prefix sums,
      `bucketize`, quotient/remainder output-index generation.
- [ ] Generic hash equi-join using hash buckets, scatter, probe, collision
      iteration, and duplicate accumulation.
- [ ] Join variants: non-equi, left outer, left semi, left anti.
- [ ] Sort-based group-by aggregation using concatenated keys, sort,
      `unique_consecutive`, inverse ids, and scatter reductions.
- [x] Sum/count group reductions for current query templates.
- [x] First batch: min/max/mean group reductions as reusable primitives.
- [ ] Count-distinct aggregation.
- [ ] Scalar/nested/correlated subquery lowering into tensor operators.
- [ ] NULL-aware boolean and aggregate semantics.
- [ ] ML prediction operator boundary for PyTorch/Hummingbird-style models.

### TQP optimization TODO

- [x] First generic grouped aggregate path avoids Python row-group loops and uses
      tensor group ids plus grouped reductions. Non-aggregate projection
      materialization still decodes result rows explicitly.
- [ ] Keep remaining data-dependent loops out of hot paths; loops over
      schema/operators are acceptable, loops over rows are not.
- [ ] Preserve columnar late materialization: joins produce row-index pairs before
      materializing payload columns.
- [ ] Prefer tensor operations over Python control flow for row-level work.
- [ ] Use compiled execution where possible: TorchScript/TVM/torch compile paths,
      common sub-expression elimination, operator fusion, code generation, and
      Python dependency removal.
- [x] Add reusable `LookupIndex` for pre-sorted dimension-key lookup probes.
- [ ] Add optimizer awareness for sorted/unique columns to avoid redundant `sort`,
      `unique`, and `unique_consecutive` across whole query plans.
- [ ] Add join strategy selection between hash and sort joins based on collision
      degree, key cardinality, and device.
- [ ] Track backend-specific bottlenecks: `unique`, indexing, `masked_select`,
      `scatter_add`, `nonzero` synchronization, and sort cost.
- [ ] Pipeline/capture data movement separately from query execution.
- [ ] Cache frontend compilation and tensor operator plans.
- [ ] Add inter-operator parallelism and distributed/data-parallel execution once
      the single-GPU operator graph is explicit.

## Compressed-data operator inventory (**verified**)

### Encodings

- [x] Plain tensor columns: current repository baseline.
- [x] Dictionary-encoded string columns: current repository baseline.
- [ ] RLE columns represented by value, inclusive start, and inclusive end
      tensors sorted by start/end, with non-overlapping ranges.
- [x] First compressed primitive step: Index mask positions represented by sorted
      unique position tensors.
- [ ] Plain + Index composite encoding for outlier separation and bit-width
      reduction.
- [ ] RLE + Index composite encoding for columns with both continuous runs and
      isolated impure segments.
- [ ] Centered bit-width reduction for numeric ranges.
- [ ] Encoding metadata in storage/catalog so the backend can choose plain,
      RLE, index, or composite execution without changing SQL.
- [ ] Heavyweight codecs such as Snappy/zstd/LZ4/gzip are out of scope for this
      line until lightweight encoded execution is complete.

### Fundamental primitives from the paper and appendices

- [x] First batch: `plain_to_rle` for boolean masks.
- [x] First batch: `rle_to_index` for RLE mask expansion to explicit positions.
- [x] First batch: `range_intersect` for RLE/RLE interval intersection.
- [x] `idx_in_rle` for index positions contained in RLE ranges.
- [x] `rle_contain_idx` for choosing RLE ranges that contain index positions.
- [x] `idx_in_idx` for index/index intersection.
- [x] First batch: `range_union` for RLE/RLE union.
- [x] `merge_sorted_idx` for index/index union.
- [ ] `compact_rle` to remove gaps between RLE runs after filtering.
- [ ] `compact_rle_index` for RLE+Index compaction.
- [x] First batch: `complement_rle` from Appendix A.1.
- [x] `complement_index` from Appendix A.1.
- [x] `rle_to_plain`.
- [ ] `plain_to_rle_index` for Plain → RLE+Index.
- [ ] `plain_to_plain_index` for Plain → Plain+Index.
- [ ] `range_arange` helper used to generate positions/runs in range algorithms
      and RLE join-index expansion.

### Logical operators on encoded masks

- [x] AND for Plain/Plain: direct boolean mask `&`.
- [x] AND for RLE/RLE: `range_intersect`.
- [x] AND for RLE/Plain: correctness-first dispatch via explicit Index
      conversion; cost model remains pending.
- [x] Primitive support for RLE/Index AND via `idx_in_rle` and `rle_contain_idx`;
      cost-based selection remains pending.
- [x] AND for Index/Index: `idx_in_idx`.
- [x] OR for RLE/RLE: `range_union`.
- [x] OR for Index/Index: `merge_sorted_idx`.
- [x] OR for mixed RLE/Plain and RLE/Index: correctness-first dispatch via
      explicit Plain/Index conversion; bucketized inclusion remains pending.
- [x] NOT Plain: tensor complement.
- [x] NOT RLE: `complement_rle`.
- [x] NOT Index: `complement_index`, returning RLE because complements of sparse
      masks are usually continuous.
- [ ] Composite mask rewrites with De Morgan expansions for RLE+Index and
      Plain+Index instead of bespoke special-case kernels.

### Alignment, arithmetic, comparison, and selection

- [ ] General alignment operator for point-wise operations over heterogeneous
      encodings.
- [ ] RLE/RLE alignment: intersect positional ranges and reconstruct aligned
      value tensors without expanding to rows.
- [ ] Plain/RLE, Plain/Index, RLE/Index, and composite alignment cases.
- [ ] Scalar arithmetic/comparison on compressed columns by operating only on
      value tensors when no positional alignment is needed.
- [ ] Binary arithmetic: `+`, `-`, `*`, `/`, modulo/remainder across aligned
      columns.
- [ ] Binary comparison: `=`, `!=`, `<`, `<=`, `>`, `>=` across aligned columns.
- [x] First Q6 selection path computes encoded `MaskColumn`, converts to row
      indices, and applies to Plain revenue columns.
- [ ] General selection: compute encoded `MaskColumn`, align it with the selected
      `DataColumn`, and apply output encoding rules for non-Plain targets.
- [ ] Preserve output encoding decisions from the paper's tables instead of
      silently materializing plain tensors.

### Group-by and aggregation on encoded data

- [ ] Grouping phase over aligned group-by columns using unique values and
      inverse ids.
- [ ] Aggregation phase using scatter operations over inverse ids.
- [ ] RLE `COUNT`: sum run lengths.
- [ ] RLE `SUM`: sum value × run length.
- [ ] RLE `MIN`/`MAX`: reduce value tensors only.
- [ ] `AVG`: post-process `SUM / COUNT`.
- [ ] `STD`/`VAR`: use sum of squared values plus sum/count post-processing.
- [ ] Appendix A.2 group-by walkthrough converted into regression tests.
- [ ] Avoid redundant filtering of aggregate columns when RLE group-by columns
      already carry filtered ranges.

### Join operators on encoded data

- [ ] Reuse GPU hash join over value tensors for Plain/RLE/Index join columns.
- [ ] Produce Join Index tensors rather than immediately materializing payload
      columns.
- [ ] RLE join columns: hash join run values, then map run ids back to row ranges.
- [ ] Index join columns: hash join encoded values, then recover row positions
      through the position tensor.
- [ ] RLE/RLE many-to-many join-index expansion using run-length products.
- [ ] Plain/RLE and RLE/Index join-index encodings from Table 6.
- [ ] Apply Join Index to payload columns, including unsorted and duplicate join
      indices.
- [ ] Bucketize the sorted side when applying unsorted RLE/Index join indices.
- [ ] Optimize one-to-one and one-to-many joins when duplicate-free side is known.
- [ ] Support semi-joins and PK/FK joins as first-class join-index patterns.
- [ ] Appendix A.3 join-index example converted into regression tests.

### Compression-aware optimizer rules from Appendix D

- [ ] Apply predicates to RLE columns before Plain columns.
- [ ] Evaluate multiple predicates on the same RLE column as one composite value
      tensor predicate, then apply it to start/end tensors once.
- [ ] Prioritize joins and semi-joins involving RLE columns to avoid fragmenting
      runs with earlier Plain operations.
- [ ] Avoid redundant filter operations in filter → group-by → aggregate plans
      when RLE group-by columns already track the filtered row ranges.
- [ ] Generalize selection pushdown for Plain and compressed execution.
- [ ] Add explicit NULL support for compressed representations rather than using
      the paper's experimental no-NULL shortcut.

### Compression layout and data-ordering TODO

- [ ] Implement encoding choice heuristics: small columns Plain; RLE when
      compression ratio exceeds threshold; RLE+Index for many single-element runs
      with longer-run compression; Plain+Index for outlier-driven bit-width
      reduction; otherwise Plain/centered Plain.
- [ ] Add query-specific ordering experiments for TPC-H Q1, Q2, Q6, Q11, Q14,
      Q15, Q17, and Q19 using Appendix B.1 ordering columns.
- [ ] Add V-order or cardinality-ordering experiments as optional storage layout
      preparation steps, not as SQL rewrites.
- [ ] Track compression ratio, run count, average run length, and HBM footprint in
      validation output.

## Abstract-derived TODO from TQEx

Full TQEx text is not locally available yet. From DOI/Crossref metadata and the
abstract, the follow-up tasks are:

- [ ] Re-download or obtain an authorized TQEx PDF and extract all operators,
      appendices, and implementation details.
- [ ] Analyze and model the gap between irregular SQL workloads and uniform
      tensor operations.
- [ ] Add efficient storage and computation strategies for variable-length data.
- [ ] Revisit tensor join and aggregate algorithms using TQEx's stated efficient
      designs.
- [ ] Extend execution to multi-XPU / multi-device processing for large data.
- [ ] Add TQP-vs-TQEx regression/performance notes once the full details are
      available.

## Abstract-derived TODO from TQP++

Full TQP++ text is not locally available yet. From the Microsoft Research page:

- [ ] Re-download or obtain the TQP++ preprint and extract all operators,
      appendices, and algorithms.
- [ ] Define a ML-compiler-native operator graph that can be lowered beyond
      eager PyTorch.
- [ ] Add Antares-compatible lowering experiments.
- [ ] Add tiered GPU resource scheduling for SQL operator execution.
- [ ] Add map-reduce-oriented fusion to reduce intermediate materialization.
- [ ] Add a multi-gated execution graph that can choose operator algorithms from
      runtime data and encoding/cardinality signals.

## Abstract-derived TODO from CoddSpeed

Full CoddSpeed text is not locally available yet. From public metadata/pages:

- [ ] Re-download or obtain the CoddSpeed paper and extract all operators,
      appendices, and system details.
- [ ] Keep the GPU engine hardware-independent and separate from SQL admission.
- [ ] Treat data movement as a first-class plan property.
- [ ] Model accelerator and interconnect placement: GPUs, FPGAs, ASICs, NVLink,
      and InfiniBand.
- [ ] Add plan annotations for HBM residency, CPU↔GPU transfer, GPU↔GPU transfer,
      and remote/distributed movement.
- [ ] Add execution metrics that distinguish compile time, transfer time, kernel
      time, allocation time, and result materialization time.

## Implementation batches

### Batch 1: foundational reusable primitives — completed

- [x] Document source status, verified operator inventory, and full TODO.
- [x] Download and track the compressed SQL analytics arXiv PDF.
- [x] Plain mask helpers: `logical_and_all`, `logical_or_all`, `gather_by_mask`.
- [x] Plain grouped reductions: `grouped_min`, `grouped_max`, `grouped_mean`.
- [x] Plain top-k helper: `topk_indices`.
- [x] RLE mask container and first primitives: `plain_to_rle`, `rle_to_index`,
      `range_intersect`, `range_union`, `complement_rle`.

### Batch 2: generic SQL expression and aggregate expansion — in progress

- [x] Boolean filter expression tree for comparison/`IN`/`LIKE` with `AND`/`OR`/`NOT`.
- [ ] Full arithmetic/date/string expression tree beyond filters.
- [x] Generic `MIN`, `MAX`, `AVG`, and `COUNT(col)`.
- [ ] Basic `HAVING`.
- [x] Generic `IN`, `LIKE`, `OR`, and `NOT`.
- [ ] Generic `CASE`.
- [x] Multi-key order-by and DESC/ASC handling.
- [ ] Tensor top-k integration for `ORDER BY ... LIMIT`.

### Batch 3: generic joins and subquery lowering

- [ ] PK/FK lookup join as the first generic join.
- [ ] Hash equi-join producing late-materialized index pairs.
- [ ] Semi/anti joins.
- [ ] Mark/delimiter-style subquery patterns needed by TPC-H shapes.

### Batch 4: compressed storage and mask execution

- [ ] Encoding metadata and RLE/Index column storage.
- [x] First encoded logical mask primitives for RLE/Index/Index combinations.
- [x] Correctness-first encoded logical mask dispatch for Plain/RLE/Index.
- [x] First encoded selection path in TPC-H Q6 behind `--compressed-masks`.
- [ ] Full compressed column alignment and output-encoding decisions.
- [ ] General encoded selection and predicate pushdown.

### Batch 5: compressed aggregate and join execution

- [ ] RLE-aware aggregation.
- [ ] Compressed Join Index generation and application.
- [ ] Compression-aware join ordering and filter/group-by optimizations.

### Batch 6: compiler/fusion/scheduling

- [x] First explicit operator graph inside `TQPPlan`.
- [ ] Replace complex Q2-Q22 compatibility execution with generic Join/Subquery/CTE/Distinct graph nodes.
- [ ] Fusion passes for map-reduce and projection/filter/aggregate chains.
- [ ] Device/data-movement scheduler and metrics.
- [ ] Torch compile / Antares / alternative compiler experiments.
