# Chinese Docs Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship a Chinese-first README, Chinese architecture document, and Chinese operator roadmap that clearly describe the current DuckDB/Sirius-like frontend → TQP IR → PyTorch/CUDA backend path and Q1 layering.

**Architecture:** Documentation-only change. The root README becomes the concise Chinese entrypoint with Mermaid diagrams and command examples. English deep-dive documents remain in place; Chinese `.zh.md` counterparts provide the localized architecture and roadmap.

**Tech Stack:** Markdown, Mermaid diagrams, existing Python package/tests for verification.

---

### Task 1: Create Chinese architecture deep dive

**Files:**
- Create: `docs/architecture.zh.md`
- Reference: `docs/architecture.md`, `tpch_torch/runner.py`, `tpch_torch/backend/pytorch.py`, `tpch_torch/queries/q01.py`, `tpch_torch/duckdb_bridge.py`

**Steps:**
1. Add an end-to-end Mermaid flowchart from SQL input to frontend, TQP IR, PyTorch backend, optional validation.
2. Add a layer/module table mapping CLI, runner, frontend, IR, backend, query kernels, compressed masks, benchmark.
3. Add key code snippets for `run_sql_with_frontend`, `compile_tqp_plan`, `TQPPlan`, `compile_sirius_plan`, `PyTorchBackend.execute`, and Q1.
4. Add Q1-specific Mermaid diagram and explain filter, pre-encoding, dense group ids, `torch.bincount`, decode/sort.
5. Document strict Substrait policy and current SQL/TPC-H support boundaries.

### Task 2: Create Chinese operator roadmap

**Files:**
- Create: `docs/operator-roadmap.zh.md`
- Reference: `docs/operator-roadmap.md`

**Steps:**
1. Mirror the English roadmap status in Chinese.
2. Keep verified-vs-abstract-derived source status explicit.
3. Preserve TODO checkboxes for TQP operators, compressed-data operators, optimizer rules, and implementation batches.
4. Link back to the English full roadmap.

### Task 3: Rewrite README as Chinese-first entrypoint

**Files:**
- Modify: `README.md`
- Create: `README.zh.md`

**Steps:**
1. Start with project positioning and current status badges/checklist.
2. Add a polished architecture Mermaid diagram.
3. Add Q1 layered Mermaid diagram and short implementation explanation.
4. Add setup, data generation, run, validate, all TPC-H, Substrait, compressed mask, benchmark commands.
5. Add support matrix and documentation navigation.
6. Add accurate Roadmap summary with links to English and Chinese docs.
7. Make `README.zh.md` a stable pointer to the Chinese root README to avoid duplicated content drift.

### Task 4: Verify, commit, merge main, push

**Files:**
- All docs changed above.

**Steps:**
1. Run targeted verification:
   ```bash
   timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest tests/test_packaging.py tests/test_runner_cli.py tests/test_benchmark.py -q
   timeout 60 /work/torch-query-gpu/.venv/bin/python -m compileall -q tpch_torch scripts
   ```
2. Commit docs on `docs/zh-docs-architecture`.
3. Fetch origin and merge latest `origin/main` into the docs branch if needed.
4. Switch to main repository, fast-forward merge docs branch.
5. Run full verification:
   ```bash
   timeout 60 /work/torch-query-gpu/.venv/bin/python -m pytest -q
   timeout 60 /work/torch-query-gpu/.venv/bin/python -m compileall -q tpch_torch scripts
   ```
6. Push `main` to origin.
