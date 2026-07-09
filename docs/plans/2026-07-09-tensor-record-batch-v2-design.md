# TensorRecordBatch v2 与表达式 AST/DAG 设计方案

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在已有 TQP physical-plan interpreter 上，把 `TensorRecordBatch` 升级为携带 DuckDB 类型、列式物理存储、chunk 元数据、变长数据与生命周期语义的 typed columnar ABI，并把 filter/projection 从字符串递归求值演进为 typed AST → optimized DAG → tensor primitive plan。

**Architecture:** 先调研并借鉴 DuckDB vector/data chunk、Arrow ArrayData/buffer/offset、libcudf column/column_view/RMM、PyTorch Tensor/storage/caching allocator 的边界设计；再在当前仓库中以 `record_batch.py`、`type_mapping.py`、`physical_scan.py`、`physical_expr.py` 为最小侵入落点逐步替换，而不是一次性重写 TPC-H executor。

**Tech Stack:** Python 3.12, PyTorch Tensor CPU/CUDA, DuckDB physical JSON plans, optional future DLPack/cuDF interop, pytest。

---

## 1. 外部实现调研结论

### 1.1 DuckDB：vector 与 data chunk

DuckDB 的执行模型以 `DataChunk`/`Vector` 为核心：一个 chunk 是多列向量的水平切片，vector 携带类型、长度、有效性和物理数据。这个设计给当前项目两个直接启发：

- `TensorRecordBatch` 需要显式 `row_count/chunk_size/chunk_index/source_offset`，不能只从第一列 tensor 推断行数。
- scan/filter/projection/join/agg 应该以 chunk 为单位调度，后续才能做 pipeline、partitionable execution 和冷热查询计时。

参考：[DuckDB C Data Chunk / Vector 文档](https://duckdb.org/docs/stable/clients/c/data_chunk.html)。

### 1.2 Apache Arrow：type + buffers + children + offsets

Arrow 把逻辑类型、长度、null bitmap、buffers、child arrays 分开表达。固定宽度列通常是 validity + values；变长 binary/string 使用 offsets + values；dictionary encoding 用 indices + dictionary。这个设计对 TQP/Tensor 生态很重要：

- DB 类型不能被 PyTorch dtype 吞掉，必须保存 `duckdb_type_repr`，例如 `DECIMAL(15,2)`。
- 变长数据要以 `offsets + chars/values` 或 dictionary ids 表达，不能让 Python string object 进入 hot path。
- validity 应是列的一等属性；第一阶段可用 bool mask，后续可考虑 bitmask/packed mask。

参考：[Apache Arrow Columnar Format 文档](https://arrow.apache.org/docs/format/Columnar.html)。

### 1.3 cuDF/libcudf：owning column、non-owning view、children 与 RMM

libcudf 区分 owning `column` 与 non-owning `column_view`。`column` 管理 data buffer、null mask 和 child columns；`column_view` 只描述已有内存。字符串列是复合列，通常包含 offsets child 与 chars child。RMM 负责 RAPIDS 生态的 device memory resource / pooling。

对当前项目的结论：

- 现阶段不需要手写 GPU memory manager，PyTorch Tensor 的生命周期和 CUDA caching allocator 已够用。
- 但 ABI 必须预留 `owner/is_view/parent/stream/memory_resource`，否则后续接 DLPack、cuDF buffer 或国产卡 runtime 会破坏类型结构。
- filter/gather/project 产生的是 owning copy 还是 view 必须显式记录；对于 `offsets + chars` 字符串尤其关键。

参考：[libcudf `column`](https://docs.rapids.ai/api/libcudf/stable/classcudf_1_1column/)、[`column_view`](https://docs.rapids.ai/api/libcudf/stable/classcudf_1_1column__view/)、[column factories / strings column](https://docs.rapids.ai/api/libcudf/stable/group__column__factories/)、[RMM](https://docs.rapids.ai/api/rmm/stable/)。

### 1.4 PyTorch：Tensor/storage/device/stream 是当前生命周期基础

PyTorch Tensor 已有 device、dtype、storage、view 与 CUDA caching allocator。当前项目主要复用 PyTorch 的 tensor storage/lifetime、device dispatch、CUDA kernel launch 和算子库。设计上应避免在 Python 层重复实现内存池；生命周期扩展只做元数据标记和外部 buffer owner 句柄。

结论：

- 默认 `AllocationOwner.kind = TORCH`，由 Tensor 引用计数管理生命周期。
- 只有接入外部 CUDA/DLPack/cuDF buffer 时才引入显式 owner。
- 不增加 silent CPU fallback；任何 device move 都应是显式 primitive。

参考：[PyTorch CUDA semantics / memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html)。

---

## 2. 当前 TQP 实现的最合适落点

| 模块 | 当前职责 | 问题 | 适合的演进位置 |
| --- | --- | --- | --- |
| `tpch_torch/record_batch.py` | 第一版 `LogicalDType`、`ColumnMeta`、`TensorRecordBatch`；列为 `name -> Tensor`。 | 缺 DuckDB 原始类型、chunk metadata、storage kind、children、lifecycle。 | 升级为 v2 typed columnar ABI；保持兼容 constructor 或提供 adapter。 |
| `tpch_torch/backend/type_mapping.py` | DuckDB type string → `ColumnMeta`；decimal encode；string dictionary helper。 | `ColumnMeta` 同时承载 logical type 和 storage，职责混合。 | 新增 `ColumnType` / `ColumnStorage` 后，成为 DuckDB schema binding 入口。 |
| `tpch_torch/backend/physical_scan.py` | 从 DuckDB `fetchnumpy()` 读列并编码为 `PhysicalValue`。 | scan 是类型信息最完整的位置，但目前丢失 chunk 信息。 | 生成 `TensorRecordBatch` 的最佳入口；再适配成 `PhysicalTable` 兼容旧 executor。 |
| `tpch_torch/backend/physical_types.py` | `PhysicalValue` / `PhysicalTable` 是当前 physical executor 的运行时数据结构。 | 与 `TensorRecordBatch` 并存；alias/value dedup、metadata 传播在这里。 | 中期保留为兼容 view；新增 `PhysicalTable.batch` 或 adapter，避免一次性重写。 |
| `tpch_torch/backend/physical_expr.py` | 字符串递归解析 DuckDB expression 并即时执行 torch op。 | 难做 CSE、constant folding、decimal scale hoisting、multi-projection batch execution。 | 抽出 `expression_ast.py`、`expression_optimizer.py`、`expression_lowering.py`，旧函数成为 compatibility facade。 |
| `tpch_torch/backend/physical_join.py` / `physical_aggregate.py` | 当前 join/agg primitive。 | 对 batch schema、key storage、validity、varlen 支持不统一。 | 等 filter/projection v2 稳定后，把 join/agg 输入边界改成 `TensorRecordBatch`。 |

判断：**最合适的路线不是推倒重写，而是 “ABI 先行 + adapter 兼容 + filter/projection 先迁移 + join/agg 后迁移”。** 原因是 TPC-H Q1-Q22 当前依赖 physical interpreter 已经能跑，贸然替换所有 `PhysicalTable` 会增加正确性风险；而 filter/projection 是表达式优化收益最大、接口最清晰的切入点。

---

## 3. TensorRecordBatch v2 目标结构

### 3.1 数据类型层

```python
@dataclass(frozen=True)
class ColumnType:
    name: str
    duckdb_type_id: str           # BIGINT, DECIMAL, VARCHAR, DATE, ...
    duckdb_type_repr: str         # DECIMAL(15,2), VARCHAR, DATE
    logical_dtype: LogicalDType   # INT64, FP32, FP64, DECIMAL, STRING, BOOL, DATE
    nullable: bool
    precision: int | None = None
    scale: int | None = None
```

要求：

- `ColumnType` 表达 SQL/DuckDB 语义。
- `LogicalDType` 表达 TQP tensor lowering 的逻辑分类。
- `torch.dtype` 不属于 `ColumnType`，避免把 `DECIMAL(15,2)` 和普通 `INT64` 混淆。

### 3.2 物理存储层

```python
class StorageKind(str, Enum):
    FIXED = "fixed"
    DECIMAL64 = "decimal64"
    DICTIONARY = "dictionary"
    UTF8_OFFSETS = "utf8_offsets"

@dataclass(frozen=True)
class ColumnStorage:
    kind: StorageKind
    data: torch.Tensor
    torch_dtype: torch.dtype
    validity: torch.Tensor | None = None
    children: Mapping[str, torch.Tensor] = field(default_factory=dict)
    dictionary: tuple[str, ...] | None = None
    owner: AllocationOwner | None = None
    is_view: bool = False
    parent_id: str | None = None
```

存储规则：

- `FIXED`: `data.shape == [row_count]`，用于 INT64/FP32/FP64/BOOL/DATE。
- `DECIMAL64`: `data.dtype == torch.int64`，scale/precision 来自 `ColumnType`。
- `DICTIONARY`: `data.dtype == torch.int64` ids，`dictionary` 保存 vocabulary；适合 TPC-H 低基数字符串。
- `UTF8_OFFSETS`: `children["offsets"].shape == [row_count + 1]`，`children["chars"]` 或 `data` 保存 packed uint8 bytes。

### 3.3 Batch metadata 层

```python
@dataclass(frozen=True)
class BatchMeta:
    row_count: int
    chunk_size: int
    chunk_index: int
    source_offset: int
    device: torch.device
    schema_version: int = 2
```

`TensorRecordBatch`：

```python
@dataclass(frozen=True)
class TensorRecordBatch:
    columns: Mapping[str, ColumnStorage]
    types: Mapping[str, ColumnType]
    meta: BatchMeta
```

不变量：

- 所有 row-level storage 的逻辑行数等于 `meta.row_count`。
- 所有 tensor device 等于 `meta.device`，除非字段被显式标注为 host-only metadata。
- filter/gather/project 返回新 batch；是否 view 由 `ColumnStorage.is_view` 标记。
- old `ColumnMeta` 可作为 v1 兼容 alias，但新实现不继续扩大其职责。

---

## 4. 生命周期与内存管理方案

### 4.1 第一阶段：PyTorch owner

默认策略：

```python
@dataclass(frozen=True)
class AllocationOwner:
    kind: Literal["torch", "external", "dlpack", "cudf"]
    handle: object | None = None
    stream: object | None = None
    memory_resource: str | None = None
```

- `kind="torch"` 时不做手工释放，由 Tensor 引用计数和 PyTorch allocator 处理。
- 不增加 `close()`，避免 Python 用户误释放仍被其他 tensor/view 使用的内存。
- `is_view=True` 时必须保留 `parent_id` 或 parent owner，防止外部 buffer 悬垂。

### 4.2 后续阶段：外部 buffer

外部 buffer 只在明确 interop 时进入：

- DLPack tensor capsule。
- cuDF/Arrow GPU buffer。
- 国产卡 runtime 的 tensor buffer。

要求：

- owner 必须显式。
- device/stream 语义必须显式。
- 不允许无声 copy 到 CPU。

---

## 5. 变长数据方案

### 5.1 低基数字符串：dictionary ids

第一阶段继续优先 dictionary：

- TPC-H 中 flag/status/nation/region/brand/container 等低基数字符串非常适合。
- equality/IN/group/join 都可转成 int64 ids 操作。
- 需要补齐 dictionary merge、unknown policy、gather/filter 后 vocabulary 稳定性。

### 5.2 通用 VARCHAR：offsets + chars

第二阶段实现 `UTF8_OFFSETS`：

```text
row i bytes = chars[offsets[i] : offsets[i+1]]
validity[i] = false 表示 NULL，不等于 empty string
```

filter/gather 策略：

- correctness-first：先 compact offsets/chars，生成 owning column。
- optimization：后续可支持 view 模式，但必须显式记录 parent 和 logical row mapping。

字符串函数：

- `=` / `IN`: dictionary fast path；UTF8 fallback 后续实现。
- `prefix/contains/suffix`: dictionary fast path 可直接在 vocabulary 上预计算 accepted ids。
- `LIKE/substring`: 后续考虑 Triton/CUDA extension/cuDF interop，不作为 v2 P0。

---

## 6. Filter / Projection AST 与优化方案

当前 `physical_expr.evaluate_expression()` 是字符串递归解释器，适合作为兼容入口，但不适合长期优化。目标链路：

```text
DuckDB expression string / JSON expression
        ↓ parse
TypedExpr AST
        ↓ bind DuckDB type + nullable + storage kind
Expression DAG
        ↓ optimize
TensorPrimitivePlan
        ↓ execute(batch)
TensorRecordBatch
```

### 6.1 AST 节点

```python
@dataclass(frozen=True)
class TypedExpr:
    op: ExprOp
    children: tuple["TypedExpr", ...]
    literal: object | None
    column: str | None
    output_type: ColumnType
    nullable: bool
```

### 6.2 优化 passes

| Pass | P0/P1 | 说明 |
| --- | --- | --- |
| Type binding | P0 | 每个 node 标注 DuckDB type、logical dtype、nullable、scale。 |
| Constant folding | P0 | 纯 literal 子树提前求值。 |
| Decimal scale hoisting | P0 | 对同一表达式统一 scale alignment，减少重复乘/除 10。 |
| CSE | P0 | 多 projection 共享子表达式，例如 Q1 中 `l_extendedprice * (1-l_discount)`。 |
| Predicate normalization | P1 | `AND/OR/NOT/BETWEEN/IN` 统一成 mask DAG。 |
| Validity propagation | P1 | NULL mask 与表达式结果一起 lower。 |
| Numeric fusion | P2 | 对纯数值 DAG 评估 `torch.compile`/Triton/custom CUDA。 |

### 6.3 Primitive plan

Primitive plan 是 schema/operator 级循环，不是 row-level Python loop：

```python
@dataclass(frozen=True)
class TensorPrimitive:
    op: PrimitiveOp
    inputs: tuple[str, ...]
    output: str
    attrs: Mapping[str, object]

@dataclass(frozen=True)
class TensorPrimitivePlan:
    primitives: tuple[TensorPrimitive, ...]
    outputs: tuple[str, ...]
```

执行规则：

- primitive 只能调用 tensor op 或明确的 GPU kernel。
- 输出 tensor device 必须与 batch device 一致。
- CUDA 不可用时测试 skip；运行时不静默 fallback。
- DECIMAL overflow/rounding 未定义时显式错误，不自动转 fp64；只有文档化的 `/` 或 AVG 可以输出 fp64。

---

## 7. 分阶段 TODO 与验收

### P0：ABI 与 filter/projection 优化入口

1. 扩展 `record_batch.py`：加入 `ColumnType`、`ColumnStorage`、`BatchMeta`、`StorageKind`、`AllocationOwner`。
2. 保留 v1 API 兼容：现有 `ColumnMeta` / `TensorRecordBatch(columns, meta, validity)` 测试不破。
3. 更新 `type_mapping.py`：DuckDB type string 绑定到 `ColumnType`，DECIMAL/VARCHAR/DATE 可 round-trip。
4. 更新 `physical_scan.py`：scan 时生成 batch metadata，包括 `chunk_size/chunk_index/source_offset/device`。
5. 新增 `record_batch_ops.py`：filter/project/gather 以 `TensorRecordBatch -> TensorRecordBatch` 执行。
6. 新增 AST 模块：`expression_ast.py`、`expression_optimizer.py`、`expression_lowering.py`。
7. Projection 测试覆盖：多精度、多表达式、嵌套深度、CSE、decimal scale hoisting、CPU/CUDA device。

完成标准：

- 所有现有测试通过。
- 新增测试能证明 filter/projection 不依赖 row-level Python loop。
- Q1 projection/filter 至少可通过 AST/DAG primitive plan 执行，并与现有结果一致。

### P1：变长数据与生命周期

1. Dictionary string metadata 标准化：vocabulary、unknown policy、nullable、merge。
2. `UTF8_OFFSETS` prototype：offsets + chars + validity。
3. filter/gather/project 支持 dictionary 与 UTF8 offsets。
4. lifecycle 标记：owning vs view、parent id、external owner placeholder。
5. string predicate fast path：equality/IN/prefix/contains/suffix dictionary ids。

完成标准：

- empty string 与 NULL 可区分。
- dictionary ids 在 filter/gather/project 后稳定。
- UTF8 offsets 在 CPU/CUDA tensor 上 shape/device 正确。
- 没有无声 CPU fallback。

### P2：join/agg 全面接入 typed batch

1. sort join 输入改成 `TensorRecordBatch` key columns。
2. group-by SUM 输入改成 typed storage。
3. DECIMAL key/value 按 `ColumnType.scale` 处理。
4. string dictionary key 支持 join/group。
5. hash join API 接受 typed key storage，输出 row indices。

完成标准：

- inner join 支持指定 key/payload 列数与类型。
- single group-by SUM 支持指定 group key 列数/type 与 SUM 个数/type。
- CPU/GPU 输出一致，device 断言完整。

### P3：性能与编译

1. 多 projection batch lowering + intermediate cache。
2. CSE 后的 primitive fusion。
3. `torch.compile`/Triton/custom CUDA 评估。
4. sort/hash join strategy selector。
5. 与 partitionable execution/chunk scheduler 合并。

完成标准：

- Q1 热查询 projection/filter 部分减少重复中间 tensor materialization。
- benchmark 区分 compile time、H2D/D2H、kernel time、end-to-end time。

---

## 8. 不做什么

- 不在 P0 引入自研 GPU memory manager。
- 不在 P0 直接依赖 cuDF 作为 fallback。
- 不为“能跑”添加静默 CPU fallback 或 mock 成功路径。
- 不一次性把 Q1-Q22 executor 全部替换成新 batch ABI。
- 不把 DECIMAL 隐式转 fp64 来掩盖 overflow/rounding 问题。
