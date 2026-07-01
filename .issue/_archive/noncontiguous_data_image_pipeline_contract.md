---
status: resolved
created: 2026-07-01
updated: 2026-07-01
resolved: 2026-07-01
owners: []
reviewers: []
---

# 非连续张量契约：`src/data` 变换/加载 与 `src/tensor/image.rs` 图像 I/O

> **状态**：resolved（2026-07-01 闭环）—— 两块边界均按**方案 2·统一守卫**收口，全仓公共 Tensor 输入现统一接受非连续布局。原始分析保留作历史参考。

## 解决方案（2026-07-01 闭环）

采纳 [四、决策] 中的**方案 2（统一守卫）**，把 `src/data` 与 `src/tensor/image.rs` 的布局相关读取全部改为布局无关或加 `contiguous()` 守卫，让变换/加载/图像 I/O 与 autograd 主线口径一致——**接受任意布局输入**：

- **`X.flatten_view().to_vec()` → `X.to_vec()`**（结果本就是拷贝，`to_vec` 按逻辑行主序、布局无关）：`random_rotation` / `affine_kernel` / `random_resized_crop` / `random_erasing`（`erase_region`）/ `detection`（`ImageBatch`）/ `dataloader`（`apply_transform_to_batch` 两处）。
- **`let flat = X.flatten_view();` → `let src = X.contiguous(); let flat = src.flatten_view();`**（连续零拷贝借用、非连续物化一份，flat 逻辑序与手写偏移对齐）：`normalize` / `gaussian_noise` / `color_jitter`（对比度/饱和度两处）/ `random_flip`（`flip_horizontal`）/ `dataloader`（`apply_transform_to_batch` 迭代、`extract_tensor_batch` 特征/标签）。
- **`image.rs::is_image` 的 `as_slice().unwrap()` → `iter()`**：逐元素范围校验与遍历顺序无关，直接用布局无关的 `iter()`；`to_image_buff_*` 本就用 `self[[y,x,c]]` 逻辑索引，无需改动。
- **回归测试**：新增 `src/data/tests/transform_noncontiguous.rs`（Normalize / 水平翻转与「permute 视图物化为连续」的等价计算逐元素对比，既抓 panic 也抓静默错序；GaussianNoise 非连续 no-panic 冒烟）与 `src/tensor/tests/image.rs`（`is_image` 非连续 no-panic、`to_image` 与连续等价张量产出一致）。

**验证**：全 lib **3271 测试 0 失败**；`cargo fmt` 已过；改动文件无新增 clippy 告警。**关键校正**：排查中一度怀疑 `CenterCrop`/`RandomCrop`（内部 `narrow`）会产出非连续张量、接 `Compose(...→Normalize)` 时 panic；实测发现本项目 `narrow` 实现为 `slice.to_owned()`（**立即物化为连续**），故 crop 链路本就安全——即在此次统一守卫前，库内也**无**可达 panic，这块纯属把「用户手动喂 permute/transpose 视图」的潜伏 footgun 一并焊死。

---

## 原始内容（保留作为历史参考）


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
