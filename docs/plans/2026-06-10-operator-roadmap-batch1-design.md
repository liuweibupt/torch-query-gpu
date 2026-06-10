# Operator Roadmap Batch 1 Design

## Goal

Create a paper-grounded operator roadmap for the TQP → TQEx → TQP++ → CoddSpeed line and the compressed-data GPU SQL paper, then implement the first correctness-first batch of reusable PyTorch tensor primitives.

## Sources and confidence

- Full-text local sources: TQP PDF and `GPU Acceleration of SQL Analytics on Compressed Data` PDF.
- Metadata/abstract-only sources: TQEx DOI/Crossref metadata, TQP++ Microsoft Research page, CoddSpeed Microsoft Research/DOI metadata.
- The roadmap must separate verified paper/appendix items from abstract-derived items. It must not invent appendix details for inaccessible papers.

## Architecture

Batch 1 adds small, reusable primitives below the existing SQL frontend/backend boundary. Plain tensor primitives extend `tpch_torch/operators.py`. Compression-oriented RLE mask primitives live in a new `tpch_torch/compressed.py` module so future compressed execution can evolve without polluting the generic SQL frontend or TPC-H template executors.

## First batch scope

Plain operators:

- `logical_and_all`, `logical_or_all`, and `gather_by_mask` for selection/filter expression composition.
- `grouped_min`, `grouped_max`, and `grouped_mean` for TQP aggregate coverage beyond sum/count.
- `topk_indices` as a top-k/limit building block.

Compressed primitives:

- `RLERanges` for inclusive RLE mask intervals.
- `plain_to_rle`, `rle_to_index`, `range_intersect`, `range_union`, and `complement_rle` from the compressed-data paper's primitive and appendix NOT-operator sections.

## Error handling

All new helpers should fail explicitly on unsupported or inconsistent inputs: empty mask sequences, non-boolean masks, mismatched shapes, invalid group ids, empty groups for min/max/mean, malformed RLE ranges, and invalid top-k sizes. There is no DuckDB-result fallback and no simulated success path.

## Testing

Use TDD. Add unit tests for the new plain and compressed primitives, run them to observe the expected import failures, then implement the minimal production code. Finish with the full repository test suite and compile verification.
