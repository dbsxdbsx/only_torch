---
status: active
created: 2026-07-01
updated: 2026-07-01
owners: []
reviewers: []
---

# 非连续张量契约：`src/data` 变换/加载 与 `src/tensor/image.rs` 图像 I/O

> **状态**：active —— autograd/raw_node 主线 + Tensor 数值公共原语 + 参数序列化的**连续性（contiguity）健壮性已闭环**（见下文「已闭环范围」）。本条目仅追踪**刻意留作后续**的两块边界：数据管线变换/加载、图像 I/O。它们契约不同、风险面独立，非 autograd 数值主风险面。
> **关联**：广播/shape 见 [广播机制设计](../../.doc/design/broadcast_mechanism_design.md)；数据加载见 [数据加载设计](../../.doc/design/data_loader_design.md)。

---

## 一、背景：什么是"连续性"问题

张量可能是**非连续**内存布局的零拷贝视图（`permute` / `transpose` 用 stride 重排产生；`narrow` 等亦可能）。代码里"按行主序读原始缓冲"的写法分两类：

- **布局无关（安全）**：ndarray 元素索引 `t[[i,j]]`、`.iter()`、`.to_vec()`、广播算术、`ArrayView::dot()`。
- **布局相关（要求连续，否则出错）**：`data_as_slice()`、`flatten_view()`（内部 `into_shape().unwrap()`）、`as_slice().unwrap()`、手写平铺偏移 `slice[base + i*W + j]`。

在**非连续**张量上用第二类会：(1) **panic**（`.unwrap()`）；或 (2) **静默算错**——按物理内存序把逻辑转置矩阵误读（已在 conv2d 前向实测：loss 78 vs 正确 66，比 panic 更危险）。

**统一修法**：新增的 `Tensor::contiguous() -> Cow`（连续零拷贝借用、非连续单次物化）在布局相关读取的**消费点**做局部守卫；或改用布局无关写法（`to_vec()` / `iter()`）。**不做**全局强制连续（会废掉零拷贝视图）。

## 二、已闭环范围（本条目**不含**，仅供对照，勿重复处理）

- **autograd 节点**：`minimum` / `maximum` / `amax` / `amin` / `clip` / `atan2` / `bce` / `huber` / `layer_norm` / `rms_norm` / `batch_norm` / `repeat` / `conv2d` / `conv_transpose2d` / `deformable_conv2d`。
- **Tensor 数值公共原语**：`flatten` / `flatten_mut`（对齐 `reshape`）、`pad` / `slice_ranges` / `repeat` / `topk` / `diag` / `diag_mut` / `one_hot` / `order` / `order_mut` / `shuffle` 系列。
- **其它**：`group_norm` / `embedding` / onnx 导出、参数序列化保存（`model_save.rs` / `serialization.rs`）、`AbsDiffEq`（`assert_abs_diff_eq!`）。
- 均补了带 `permute` 上游的回归测试（`*noncontiguous*`），全 lib 测试 0 失败，经三轮独立 review 闭环。

## 三、本条目追踪的未闭环边界

### 3.1 `src/data`（变换 / DataLoader）
transforms / dataloader 仍大量使用 `flatten_view()` + 手写 CHW / HW 平铺索引。当前**假设**输入是 dataset / transform 自建的**连续**图像/批张量，实践中通常成立，故未纳入本次数值主线修复。

**风险**：若未来有人把 `permute`/`transpose` 视图直接喂给某个 transform，会 panic 或静默算错。

### 3.2 `src/tensor/image.rs`（图像 I/O）
图像转换仍有 `as_slice().unwrap()` 等连续性假设，属图像 I/O 边界。

## 四、决策 / 待办（二选一，需人工拍板）

对上述两块，任选其一收口：

1. **文档化契约**：在 `Transform` / `SampleTransform` 与 `image.rs` 的 API 文档明确"输入必须连续"，并在入口 `debug_assert!(x.is_contiguous())` 或返回错误，把契约显式化（成本低）。
2. **统一守卫**：在各布局相关读取消费点套 `Tensor::contiguous()`（与数值主线一致，成本中等，面较大）。

建议默认走 **方案 1**（成本低、契约清晰），仅当出现真实非连续输入需求时再逐点升级为方案 2。

## 五、下次恢复条件

- 出现"把 `permute`/`transpose` 视图喂进 data transform / image I/O 导致 panic 或数值异常"的真实报告；或
- 决定统一"全仓所有公共 Tensor 输入都接受非连续布局"的口径时。

## 六、复现（示意）

```rust
// 数值主线已修：以下不再 panic（本条目范围外，仅示意同类构造）
let x = Tensor::new(&[1.,2.,3.,4.,5.,6.], &[2,3]);
let nc = x.permute(&[1,0]); // [3,2] 非连续视图
// 若把 nc 喂给 src/data 的某个 flatten_view + 手写平铺索引的 transform，
// 或 src/tensor/image.rs 的 as_slice().unwrap()，可能 panic / 静默算错。
```
