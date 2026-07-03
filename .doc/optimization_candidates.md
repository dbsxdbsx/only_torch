# 性能优化候选项

> 本文档记录性能优化的候选项、已实施项和已否决项。
> 最后更新: 2026-07-03

---

## 性能验证标准流程

任何声称"优化"或"重构"的改动，必须按以下流程验证：

1. **保存 baseline**：`just bench-save before-change`（在改动前的 commit / 工作树上）
2. **执行改动**
3. **正确性验证**：优先 `just test-filter <pattern>`，影响面大时跑 `just test`
4. **快速性能回归检查**：`just bench-smoke`（目标整组约 30 秒内）
5. **对比关键路径**：`just bench-compare before-change`（查看 Criterion 输出与 HTML report）
6. **Macro 验证**：`just bench-macro`（缺少本地 chess ONNX 模型时先跑 `just bench-macro-core`）
7. **回填本文档**：在对应优化段写明 before / after 数据、命令与环境

宏基准用于真实用户路径参考，不作为默认门禁；不要把不同 CPU、不同 BLAS 后端或不同模型文件的百分比直接混为同一条趋势。

### 当前 baseline

| baseline | 日期 | 命令 | 后端 | 用途 |
|---|---|---|---|---|
| `pre-execution-context` | 2026-04-29 | `just bench-save pre-execution-context` | `blas-mkl` | `Mode` 重构前的 Criterion 对照基线 |
| `post-mode-refactor` | 2026-04-29 | `just bench-save post-mode-refactor` | `blas-mkl` | `Mode` 重构完成后的新命名完整基线，后续性能回归从这里继续比较 |
| `pre-hotpath-opt` | 2026-07-03 | `just bench-save pre-hotpath-opt` | `blas-mkl` | 热路径分配/拷贝消除批量优化（见优化 J）前的完整基线；已快照入仓 `.bench/history/20260703-000547-pre-hotpath-opt-before-hotpath/`（104 case） |
| `post-hotpath-opt` | 2026-07-03 | `just bench-save post-hotpath-opt` | `blas-mkl` | 优化 J 全部落地（含 Add 分流修复 `3027ac0`）后的新基线，**后续性能回归从这里比较**；已快照入仓 `.bench/history/20260703-093545-post-hotpath-opt-after-hotpath/`（104 case，conv2d 前向 / max_pool 前向组经安静复测覆盖首跑受扰值）。注意与 pre 基线跑于不同时段，存在 ~10% 环境偏移（见优化 J 噪声鉴定），跨基线百分比勿直接混读 |

说明：baseline 保存在 Criterion 的 `target/criterion` 报告目录中，属于本地构建产物，不入仓。`pre-execution-context` 仅用于解释 `Mode` 重构前后差异；`smoke_conv2d_eval_1x1_b1` 已随语义改名为 `smoke_conv2d_inference_1x1_b1`，后续不再为重构前 baseline 兼容名称，统一从 `post-mode-refactor` 继续对比。

> **`post-hotpath-opt` 对新依赖栈仍有效（2026-07-03 裁定）**：ndarray 0.17.2 + pyo3/numpy
> 0.29 升级经三层归因（噪声金丝雀 / 复测翻转 / worktree 控制实验，详见 §4）判定性能中性，
> 该基线无需因依赖升级重录；下次优化战役开跑时照常 `just bench-save pre-<名>` 即自然获得
> 新栈起点。唯一纪律不变：**基线录制与对比跑尽量同为安静窗口**，白天对比只做方向参考。

### Mode 重构对比结果

- `just bench-compare pre-execution-context` 跑到 `smoke` 时因 benchmark case 改名中断；中断前 `tensor_ops`、`conv2d_forward`、`backward`、`end_to_end` 多个分组相对重构前 baseline 已显示明显改善，但该 baseline 不再作为后续门禁基线。
- 已复跑 `cargo bench --bench smoke --features blas-mkl -- --baseline pre-execution-context`；6 个 smoke 项全部显著改善，关键链路未复现回归。
- 已保存 `post-mode-refactor` 完整新基线，并补跑 `smoke` 到 `attention` 后续分组；新命名 benchmark 全部跑通。
- 行为回归已用 invariants、梯度流、模型加载、BatchNorm / Conv2d 节点测试，以及 MNIST / MNIST GAN / CartPole SAC / chess YOLO ONNX example 验证；CartPole SAC 训练达到单回合 200，但三次测试平均 185.7，低于示例目标 190，仍为随机训练波动范围内。

---

## 待优化项

### 1. RNN 场景 `select` + `set_value` 二次复制问题

**位置**：`src/nn/layer/rnn.rs` → `forward`

**现状**：每个时间步两次数据复制。当前规模开销可忽略（~1.2 MB 总冗余）。

**推荐方案**：方案 A（使用已有的 `set_value_owned` 方法），最小改动。

**状态**：暂缓（YAGNI），等 RNN 处理大规模数据时再实施

---

### 2. RNN 场景更多优化

参见 #1，等 RNN 处理大规模数据时统一评估

---

### 3. 热路径审计留档项（2026-07-03，批量优化 J 时降级未做）

2026-07-02/03 全库分配/拷贝热路径审计（优化 J）中识别、经 Reviewer 评估后**刻意不随批实施**的项。共同特点：真实存在但缺 profiler 证据排进热点前列，或改动会触碰图结构/行为（需单独 3-seed 重验），不值得为其扩大批量改动的验证面。

| 项 | 位置 | 问题 | 优先级与备注 |
|---|---|---|---|
| A3 | `node_inner.rs` `propagate_grad_to_parents` | 用 `msg.contains("不应该")` 字符串匹配做控制流（loss 对 target 不传梯度）；实际因 target 是叶子会被 `branch_needs_gradient()` 提前跳过，无热路径开销 | P2，纯代码卫生：正解是 `GradResult::NoGrad` 变体 |
| B2 | `softmax.rs` / CE 反向 | 逐 batch IxDyn 逐元素索引，可改行 slice 直取 | P1，等 profiler 证据 |
| B7 | `tensor/property.rs` `sum_to_shape` | 多趟归约（逐轴循环 sum_axis），可单趟 | P2 |
| B8 | `loss/mse.rs` 前向 | 平方和走张量表达式链，可单趟 fold | P2 |
| D4 | `mcts/search.rs` `select` | 每次 simulation 重建全量 `ChildStat` Vec | P2，等 tree-reuse 改造时一起做 |
| E1 | `graph/inner` CSE | CSE key 用 String 拼接，可换紧凑 hash key | P2 |
| C2 | `my_zero/network.rs` `min_max_normalize` | 两个 repeat 节点冗余（Subtract/Divide 本就支持 `[B,dim] ÷ [B,1]` 广播），代数成立 | 改图结构 → 必须单独一项 + 3-seed 哨兵重验，勿与其他改动混批 |
| C1 | `layer/attention.rs` | 逐 head 循环建图 | 正解是 3D batched MatMul，归属演化阶段 D（刻意暂缓），不单独开口子 |

---

### 4. ndarray / pyo3 / numpy 升级（2026-07-03 全部完成；ndarray 0.17.2 + pyo3 0.29 + numpy 0.29）

**背景**：本体曾用 `ndarray 0.15.6`（2022），而 `numpy 0.27`（pyo3 桥）与 `ndarray-npy 0.9.1`
依赖 `0.16.1` → 依赖树里两个 ndarray 版本共存、重复编译；且 `ndarray-npy` 在 `src/`
零使用（死依赖 + `"*"` 通配版本）。上游最新 `0.17.2`（2026-01；`0.17.0` 因引用类型
use-after-free 被 yank）。

**进展**：

1. ✅ **清死依赖**：`ndarray-npy` 已删。
2. ✅ **升 0.16.1**（2026-07-03）：`Cargo.toml` `^0.15` → `^0.16`，并
   `cargo update --package ndarray@0.17.2 --precise 0.16.1` 把 `numpy 0.27` 拉起的
   0.17 统一到 0.16.1（rust-numpy 声明宽版本区间，cargo 默认取最高，需手动统一）。
   迁移：`into_shape` → `into_shape_with_order`（12 处），`into_raw_vec` →
   `into_raw_vec_and_offset().0`（2 处）。
   **实际踩坑（比预期多一条）**：#1419 BLAS 全布局派发后，TN 形式 `A^T @ B` 的 GEMM
   输出直接以 **F 序**返回（0.15 会物化 C 序），破坏本框架「Tensor 产物皆标准布局」
   约定（`flatten_view`/`data_as_slice` 依赖），曾致 2 个梯度累积测试 panic；已在
   `mat_mul` 系加 `into_standard_dyn` 归一化守卫（标准布局零开销直通）。
   同时 `flatten_view` 沿用 0.16 收紧后的布局报错（0.15 旧 `into_shape` 会静默按
   内存序展平 F 序数组，属正确性隐患，收紧是好事）。
3. ✅ **升 0.17.2**（2026-07-03，与 0.16.1 同日连升）：**「需 numpy 0.29 / pyo3 联动」的
   旧预判被推翻**——`numpy 0.27.1`（Cargo.lock 已锁定版本）的 ndarray 区间即为
   `>=0.15,<=0.17`，升级零涉及 pyo3 / RL 桥。实际动作 = `Cargo.toml` `^0.16` → `^0.17.1`
   （0.17.0 因 ArrayRef use-after-free 被 yank）+ `cargo update --package ndarray@0.16.1
   --precise 0.17.2` 统一 numpy 侧；**代码零改动**（0.17 对 0.16 纯增量：`ArrayRef` 引用
   类型、IxDyn 直接 `dot`、数组级 `tanh` 等数学函数、原地 `permute_axes`；仅删了我们
   未用的 `serde-1`/`test`/`docs` feature 别名）。BLAS 路径 0.16→0.17 零变更 →
   `into_standard_dyn` F 序守卫仍必要，浮点数值与 0.16.1 一致（不再引入新一轮轨迹漂移）。
   验证：blas-mkl + 默认双口径全量测试 3317 全绿，clippy 无新增。
   可选后续（不阻塞）：用 IxDyn `dot` 删 `mat_mul` 系的 `into_dimensionality::<Ix2>`
   样板、公共 API 逐步换 `ArrayRef` 签名。
4. ✅ **pyo3 / numpy 0.27 → 0.29 同日跟进**（2026-07-03）：pyo3 与 rust-numpy 同属
   PyO3 组织、0.28 起版本号同步发版，成对升级。0.29 含两个 RustSec 安全修复
   （`PyCFunction::new_closure` 缺 `Sync` 约束、`Bound{Tuple,List}Iterator::nth_back`
   越界读）+ 多项 soundness 收紧；0.27→0.29 的破坏性变更（`PyErr` 的 `From<Utf8Error>`
   等实现移除、`pyo3_build_config` 需直接依赖、pyclass Clone 自动 FromPyObject 弃用、
   free-threaded 默认支持等）全部集中在「写 Python 扩展模块」场景——本项目纯嵌入方向
   （`auto-initialize` + `Python::attach` + `py.import`），**代码零改动**。numpy 0.29
   的 ndarray 区间仍为 `>=0.15,<=0.17`，与 0.17.2 兼容。验证：blas-mkl + 默认双口径
   全量测试 3317 全绿（含 RL / pyo3 桥测试）、examples + benches 编译通过、clippy
   180 warnings 无新增。至此依赖栈定格：**ndarray 0.17.2 + pyo3 0.29.0 + numpy 0.29.0**。

**升级后 bench 对比结论（2026-07-03 白天全量跑，104 case vs `post-hotpath-opt`）**：
名义结果 80 回归 / 3 改善 / 21 持平，**经三层归因全部判定为白天环境噪声，无一项可归因于
依赖升级**：① 噪声金丝雀齐涨——`tensor_clone`（~400 ps 空操作，代码零涉及）+8~21%、
纯 MKL `tensor_matmul` +7~18%，与全场 +5~20% 均匀漂移同带；② 最大离群项
`smoke_add_chain_backward_8` +226% 立即复测翻转为 No change；③ **worktree 控制实验**
（检出基线录制 commit `6365d85`，连 ndarray 0.15.6 一并还原，与基线状态逐 bit 相同）
当前环境复跑 pool2d backward「回归」+43~76%，**比 HEAD 新栈（+17~45%）还差**——同代码
同依赖只换时段即回归，锤死环境归因；且旧栈（0.16.1 + pyo3 0.27）中间对照 +27~48% 居中。
结论：**依赖升级在测量能力范围内性能中性**；干净的新栈基线待安静窗口录制（见下）。

**冷水**：优化 J 踩的 `&a + &b` 广播慢路径上游 0.16/0.17 均未修（最后一次广播性能优化
是 0.15.2），Add 三路分流修复升级后仍必要。

---

### 5. 推理模式中间节点 value 及早释放（liveness）

**来源**：arXiv 2308.13898（内存感知调度）的"穷人版"思想——执行顺序与张量生命周期决定峰值内存；论文本体（ILP 调度器）对我们不适用，见[阅读日志](./paper/reading_log.md)。

**现状**：`Mode::Inference` 已跳过 backward 缓存，但前向期间所有节点的 `value` 存活到结束。改动量约十几行：前向拓扑序执行时统计每个节点的剩余消费者数，归零即 `clear_value()`。

**状态**：暂缓（YAGNI）。桌面 CPU + 当前网络规模峰值内存远非瓶颈；触发条件 = 图像线大 batch CNN 或受限部署目标真实撞到内存墙。届时先做这个，再考虑任何调度方案。

---

## 已否决项

### 内核级优化路线：Winograd / 手写 GEMM / 三值内核（2026-07-02 论文清账盖棺）

**结论**：卷积与矩阵乘的**内核层**优化整体不做，含"未来图像线网络变大"场景。7 篇论文判定明细见[论文阅读日志](./paper/reading_log.md)。

| 路线 | 否决理由（各自独立成立） |
|------|------|
| **Winograd 卷积** | OneDNN 实测（arXiv 2509.26217）：单算子占优但整网推理被数据搬运反噬，隐式 GEMM 胜出；f32 数值稳定性差（条件数问题）；我们 batch=1 小矩阵场景比论文更不利 |
| **手写/魔改 GEMM 内核**（LP-GEMM 布局传播、三值掩码内核等） | GEMM 已外包 `matrixmultiply` / MKL（`.dot()` 一行）；上游若实用化可免费继承；自持内核违背跨平台与维护成本约束 |
| **Strassen 系快速矩阵乘** | 实测交叉点方阵 n≈2000–4000；DL 卷积 GEMM 高长方形、维度几百~两千；MKL/oneDNN 从未为 DL 负载启用 |

**保留的唯一方向**：若图像线 profiling 证明 conv 是热点，做**隐式 lowering**（分 tile 现场 im2row 降低搬运，2509.26217 的实践赢家），而非任何上述路线。

### `&[&NodeHandle]` vs `&[NodeHandle]`

**结论**：`&[&NodeHandle]` **不推荐**。双重间接引用增加指针追踪开销、缓存局部性差。

### 前向传播 clone NodeHandle 问题

**原始问题**：`NodeHandle.clone()` 深拷贝内部 `Tensor`。

**结论**：v2 动态图架构已彻底消除此问题。`NodeInner` 由 `Rc` 管理，前向传播通过 `borrow()` 零拷贝借用父节点值，无需 clone `NodeHandle`。

---

## 已实施的优化

### A. 赋值算子减少 clone（早期）

| 位置 | 改动 | 效果 |
|------|------|------|
| `GradientAccumulator::get_average_gradient` | `gradient.clone() / scalar` → `gradient / scalar` | 少一次 Tensor clone |
| `graph.rs` 梯度累加 | `current + contribution` → `current += &contribution` | 避免临时张量分配 |
| 反向传播 | clone `upstream_grad` → 借用引用 | 避免大 Tensor 拷贝 |

### B. Conv2d im2col + GEMM 优化（2026-02-14）

| 改动 | 效果 |
|------|------|
| 前向/反向卷积从嵌套循环改为 im2col + ndarray `.dot()` | 完整训练步 2.6-4.4x 加速 |
| 利用 ndarray 底层 matrixmultiply 库的 AVX2 自动向量化 | 无需引入外部 BLAS |

### C. 反向传播全局优化（2026-02-14）

| 改动 | 效果 |
|------|------|
| 全部 59 个节点实现 `grad_mut()`，梯度累加改为原地 `+=` | 消除每次累加的临时 Tensor 分配 |
| ReLU 反向融合 mask + multiply 为单次 `where_with_tensor` | 2 次 Tensor 分配 → 1 次 |
| MaxPool2d 反向用 `par_chunks_mut` 预分配 buffer | 消除 Vec<Vec> + flatten 双重分配 |

**综合效果（vs 优化前 baseline，release benchmark）**：

| 场景 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 完整训练步 单样本 | 493 us | 110 us | 4.5x |
| 完整训练步 batch32 | 12.3 ms | 4.4 ms | 2.8x |
| 完整训练步 batch64 | 26.3 ms | 5.8 ms | 4.5x |

### D. GradResult 零拷贝梯度传递（2026-02-15）

**原始问题**：`calc_grad_to_parent` 返回 `Result<Tensor>`，Add/Identity 等节点被迫 clone `upstream_grad`。Profile 显示 Add 反向占 12%，其中大部分是 clone。

**解决方案**：引入 `GradResult` 枚举替代裸 `Tensor` 返回：

```rust
pub(in crate::nn) enum GradResult {
    PassThrough,       // 零拷贝，直接用 upstream_grad 累加
    Negated,           // 零分配，累加时原地 -=
    Computed(Tensor),  // 新计算的梯度
}
```

| 节点 | 变体 | 效果 |
|------|------|------|
| Add（无广播）、Identity、Subtract（第一父节点）、Dropout（eval） | `PassThrough` | 零 clone |
| Negate、Subtract（第二父节点） | `Negated` | 零分配（`accumulate_grad_negated` 原地 `-=`） |
| 其余 53 个节点 | `Computed` | 行为不变 |

**影响范围**：全部 59 个节点 + trait 签名 + `propagate_grad_to_parents` 调用方

### E. Conv2d 反向传播并行策略（2026-02-15）

**原始问题**：反向传播中每个 batch 样本独立做 `im2col + GEMM`，N 次小矩阵乘法。

**尝试方案**：`batch_im2col` 将所有样本拼成大矩阵做单次 GEMM。

**最终结论**：批量化方案被**撤回**。因为 MKL 配置为 `seq`（单线程，避免与 Rayon 冲突），
单次大 GEMM 只用一个核心，反而不如 per-sample Rayon 并行（多核各跑一个小 GEMM）。
参考 Burn/Candle/PyTorch 的做法后统一为 per-sample Rayon 路径。
启用 BLAS 后 `dot()` 内部自动用 MKL 加速，代码无需区分。

### F. 优化器 set_value_owned + Adam 中间变量优化（2026-02-15）

| 改动 | 效果 |
|------|------|
| `TraitNode` 新增 `set_value_owned(Tensor)` | 优化器更新参数时零拷贝（消除 `set_value(Some(&val))` 的 clone） |
| SGD/Adam `step()` 改用 `set_value_owned` | 每个参数更新省一次完整 Tensor clone |
| Adam 偏差修正因子外提到循环外 | 所有参数共享 `bc1`/`bc2`，省去重复计算 |
| Adam `grad_sq *= (1-β2)` 原地操作 | 省去 `scaled_grad_squared` 临时 Tensor |
| Adam `denom += ε` 原地操作 | 省去 `&v_sqrt + eps` 临时 Tensor |

### G. BLAS 可选支持 — Intel MKL / OpenBLAS（2026-02-15）

**位置**：`Cargo.toml` features + `lib.rs`

**解决方案**：通过 feature flag 启用 BLAS 后端，`ndarray::dot()` 自动路由到 MKL/OpenBLAS：

```toml
[features]
blas-mkl     = ["ndarray/blas", "dep:intel-mkl-src"]     # Intel CPU 推荐
blas-openblas = ["ndarray/blas", "dep:openblas-src"]      # 跨平台备选
```

配置选择 `mkl-static-lp64-seq`（lp64 = 与 cblas-sys 兼容；seq = 避免与 Rayon 线程冲突）。

**实测效果**（Chinese Chess CNN，debug 模式，50 epoch）：

| 指标 | 无 BLAS | 有 MKL | 提升 |
|------|---------|--------|------|
| 训练总耗时 | 43.1s | **36.7s** | **14.9%** |
| 推理 batch=1 | 1.5ms | 1.4ms | 6.7% |
| 推理 batch=256 | 70.2ms | 68.9ms | 1.9% |

**设计决策**：无需条件编译不同代码路径。MKL 加速完全透明（在 `dot()` 内部），
per-sample Rayon 并行策略在有无 BLAS 时完全一致。

### H. 节点 cache clone 消除（2026-02-15）

| 节点 | 优化前 | 优化后 |
|------|--------|--------|
| Conv2d `padded_input` | `Some(padded.clone())` 缓存 | `Some(self.pad_input(input))` 直接 move |
| LeakyReLU | 缓存完整 `parent_value` | 不再缓存，反向时用 `value`（输出）判断区域，数学等价 |
| ChannelBiasAdd | `let mut result = input.clone()` | 节点已删除，由通用 Add + 广播替代 |

### I. Conv2d Inference 推理快路径（2026-04-29）

**原始问题**：YOLOv5 TinyChess 在 Debug 模式下单图检测约 2.0s，其中 forward 约 1.87s；decode + NMS 仅约 0.4ms，不是瓶颈。`Conv2d` 原本不区分训练 / 推理，`1x1` 卷积也走通用 `im2col + GEMM`，并保存 backward 需要的 `im2col_cache`。

**解决方案**：

| 改动 | 效果 |
|------|------|
| `Conv2d` 感知 `eval` 模式 | 推理时可跳过 backward 缓存 |
| `1x1 stride=1 padding=0 dilation=1` 卷积走直接 GEMM 快路径 | 避免为每个空间位置构造等价的 `im2col` 矩阵 |
| padding / `im2col` 热循环改用连续 slice 索引 | 避免 Debug 模式下多维动态索引开销 |

**实测效果**（`chinese_chess_yolov5_onnx_recognize_fen`，Debug + MKL，单图）：

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| forward | 1871 ms | 596 ms | 约 3.1x |
| 总耗时 | 2030 ms | 745 ms | 约 2.7x |

**设计结论**：卷积的数学前向在 Train / Inference 下相同，但执行引擎需要区分“是否要为 backward 保存缓存”。后续新增重算代价高、缓存占用大的节点时，应同时设计训练路径和推理路径，避免推理承担训练负担。

### J. 热路径分配/拷贝消除批量优化（2026-07-03，commit `3fd498a`）

**来源**：全库分配/拷贝审计（自审 + Reviewer 压力测试三轮收敛），动机是 MCTS 推理热路径
（每次 recurrent 约 25~40 节点 × 50 sims，全是 [1,N] 小张量，分配开销占主导）与训练路径的无谓整块拷贝。

**方法论**（与以往单项优化不同，本次为批量实施）：

- **正确性逐项锁死**：所有行为敏感改动配「fused vs 旧链」逐 bit 金测试（共 9 个），
  逐元素运算顺序刻意保持与旧张量表达式链一致 → 批量叠加不放大正确性风险；
- **性能统一测量**：单项收益多在 Criterion 噪声底（2~5%）以下，仅聚合端到端 bench 可测；
  bench 组与改动面近似对角映射（`adam_step`→优化器融合、`conv2d_*`→Conv2d 直写、
  `my_zero_forward`→解码路径…），若某组回归可用定向 bench + 局部 revert 快速归因；
- **行为改变类一律不混批**（如 min_max_normalize 去 repeat，见「待优化项 #3」C2）。

| 类别 | 改动 | 关键点 |
|------|------|--------|
| MCTS 推理 | expand 去除子节点 latent 预克隆（值从未被读且语义错误） | 大动作空间免「候选数 × latent」白拷 |
| MCTS 推理 | `softmax_row` / `decode_categorical_slice` 无 Tensor 解码 | 替代 `amax→sub→exp→sum→div` 链约 6 次小分配，逐 bit 一致 |
| MCTS 推理 | 输出 `with_value` 借用单拷贝；输入 `Var::set_value_owned` move | 消双拷贝 |
| 优化器 | Adam/SGD 单趟 Zip 融合原地更新（`apply_param_update` 单借用接口） | 每参数每步省 grad/value 整块 clone + ~6 个全尺寸临时 |
| backward | 拓扑序建一次，清梯度与传播复用 | 省一轮 DFS + HashSet + Vec\<Rc\> |
| MatMul 反向 | `mat_mul_nt` / `mat_mul_tn` 转置视图直入 GEMM | 免 transpose 整块物化，Linear backward 双热点 |
| Conv2d | 前向/反向预分配 + `par_chunks_mut` 直写 + `general_mat_mul` 写 chunk 视图 | 消 Vec\<Vec\> collect/flatten/extend 多级拷贝 |
| 张量层 | `Tensor::from_vec` 零拷贝构造替换热点；`zip_map` 融合 sigmoid/tanh/where_cond 反向；keepdims 归约 `insert_axis` | 全库「build Vec → `Tensor::new`」各省一次 memcpy |
| bug 修复 | Add 广播顺序：被广播方在前（`bias + x`）旧实现 panic | 顺带补反转/三父单测 |

**验证状态**：

- 全量测试 3317 + doctest 通过；9 个逐 bit 金测试全绿；
- 3-seed CartPole 哨兵：中位 **9826 env-steps**（3/3 达标），与改动前 ~9.8k 基线一致，无回归；
- Criterion 对比基线 `pre-hotpath-opt`（104 case，已入仓 `.bench/history/`）。

**bench 对比结果**（`just bench-compare pre-hotpath-opt`，2026-07-03 上午，blas-mkl；
76 组有结论：41 改善 / 14 持平 / 21 名义回归）：

| bench 组 | 中位变化 | 归因 |
|---|---|---|
| `adam_step` / `sgd_step` | **-73% ~ -88%** | 优化器融合 |
| `my_zero_forward`（initial/recurrent） | **-66%** | 推理无 Tensor 解码 + set_value_owned |
| `my_zero_train_batch` | -29% ~ -47% | 优化器 + backward + MatMul 叠加 |
| `conv2d_full_step` / `mlp_train_step` | -13% ~ -34% | Conv2d 直写 + 拓扑序复用 |
| `*_chain_backward` / `loss_backward` / `attention` / `rnn_backward` | -12% ~ -40% | backward 通用路径红利 |

**名义回归的噪声鉴定**：基线录制于深夜安静环境，对比跑于次日上午，存在系统性
~+10% 环境偏移。证据：① `tensor_clone` 实测 ~420 **ps**（空循环，代码影响为零）也
"回归" +10~16%；② 未触碰路径（纯 MKL GEMM 的 `tensor_matmul`、`rnn_forward`、
`mse/bce loss_forward` 等）整齐落在 +6~17% 同一噪声带；③ 初跑异常项复测全部翻转
（`avg_pool2d_forward` +123% → **-24%**、`subtract_chain_backward/small` +64% →
**-44%**、`conv2d_forward/1x1` +32% → **-25%**）。结论：改善数字实际被低估约 10%，
噪声带内的名义回归不构成代码回归证据；跨环境百分比不混为同一趋势（见本文档开头原则）。

**GroupNorm 回归定向二分（已闭环）**：初测 `group_norm_backward` +28%、forward +14%，
超出噪声带。按对角映射逐项还原二分，**一轮命中**：Add 节点前向重写把「clone 首父 + 原地
广播 `+=`」全量换成引用加法后，「大张量 + 小广播量」场景（GroupNorm 的
`(x_hat*gamma) + beta`，beta 为 `[1,C,1,1]`）踩中 ndarray 引用加法广播分配路径的慢区——
它明显慢于 memcpy + 原地广播 `+=`。修复：按形状分流三路（同形 → 引用加法单趟；
大 + 小广播 → clone + 原地 `+=`；小在前 → 引用加法扩形，即原 bug 修复路径），
三路与旧实现逐 bit 一致。修复后复测：`group_norm` forward **-7%**、backward 持平，
`add_chain_backward` 三档全部改善（-4% ~ -22%），无新回归。
教训：**「消 clone」不能想当然——memcpy + 原地向量化两趟可以快于一趟广播分配**，
广播场景的等价改写必须分形状档实测。

### K. Pool2d 平铺直写 + BatchNorm 反向单趟融合（2026-07-03，bench 绝对值审计驱动）

**来源**：依赖升级 bench 对比的绝对值审计（不看百分比看耗时）发现两处结构性浪费：
`max/avg_pool2d_backward`（b32 规格 2.7/3.5ms）比含 GEMM 的 `conv2d_full_step`（1.9ms）
还慢（~27ns/位置的 scatter）；`batch_norm_backward` 18.9ms 中 dx 表达式链贡献 ~6 个
全尺寸临时 + `[1,C,1,1]` 广播减/除慢路径。

| 改动 | 内容 |
|------|------|
| `avg_pool2d` 前向/反向 | 「IxDyn 逐元素索引 + `Vec<Vec>` flatten」（优化 C/J 未同步的旧模式）→ `contiguous()` 守卫 + 平铺 slice + `par_chunks_mut` 直写；窗口行内 slice 迭代，`upstream × grad_val` 外提 |
| `max_pool2d` 前向 | per-sample `Vec<(Vec,Vec)>` + extend 拷贝 → 双输出 buffer `par_chunks_mut` zip 直写 + 平铺读；padding 行检查外提到 kh 层 |
| `max_pool2d` 反向 | 保留 J 的直写结构，`upstream_grad[[b,c,oh,ow]]` / `max_indices[[...]]` IxDyn 读 → 平铺 slice 读 |
| `BatchNorm` 反向（训练） | dx 表达式链 → 单趟融合 `dx[i] = (n·up[i] − sum_up[ch] − xh[i]·sum_up_xh[ch]) / (std[ch]·n)`，样本维 `par_chunks_mut` 并行；输出 `Tensor::from_vec` 零拷贝 |

**正确性**（照 J 纪律）：3 个新逐 bit 金测试（全库金测试家族 9 → 12）——pool 双算子以
旧实现（IxDyn 串行）为参考覆盖默认 stride/非方核/重叠窗口/padding/ceil_mode/全常数平局
首胜共 7 case；BatchNorm 以「与前向逐 bit 相同的统计量重建 + 旧 dx 链」为参考覆盖
2D/4D 共 3 case。双口径全量 3320 测试全绿，clippy 无新增。

**实测**（同环境背靠背 `pre-pool-bn` 基线，白天窗口）：

| bench | 中位变化 | 备注 |
|---|---|---|
| `avg_pool2d_backward/b32` | **-36%**（3.5ms→2.65ms，**优于夜间基线绝对值** 3.51ms） | 三轮复测稳定（含污染轮也 -36%），最硬的一条 |
| `avg_pool2d_backward/b8` | -9~-10% | 稳定 |
| `max_pool2d_backward/b32` | -9%（复测），首轮持平 | IxDyn 读占比小于 avg 的写扇出 |
| `batch_norm_backward`（隔离对照） | **dx 计算 2.35x**（4.23ms→1.80ms，release+MKL 手动档 `bn_backward_micro`，30 次中位） | 全链路 bench 组含 MSE/SGD/前向稀释后 -14%，两口径吻合 |
| `avg_pool2d_forward/b8`、k3 档 | 确证净代码收益：旧实现同窗对照 avg b8 **+0.6% → 新 -17%**、k3 avg **+13% → +3.7%** | 用 stash 旧实现同窗复跑对照法归因 |
| `max/avg_pool2d_forward/b32` | 与旧实现同窗对照持平（旧 -21% vs 新 -16%，同噪声带） | 大 batch 下 rayon 摊薄了 IxDyn 开销，vs 基线的改善数字主要是环境恢复成分 |

**测量插曲（教训）**：首轮对比跑在「隔壁 agent 编译任务」污染窗口，全场 +30~85% 假回归
（含未改动的 layer/rms_norm）；金丝雀（未触碰的 `layer_norm_backward`）复测归零后重跑，
数字才可用。**白天窗口跑 bench 前先跑一个未触碰组做金丝雀**已验证是必要程序。
`avg/max_pool2d_forward/b16_c32_14x14_k3`（~9.5µs 小 case）在新旧实现间无显著差异
（k=3 小窗口下平铺收益与 par 调度开销相抵），非回归。

---

## Benchmark 基础设施（✅ 已搭建）

已引入 `criterion` 框架，11 个 benchmark 文件覆盖各层面：

| 文件 | 覆盖范围 | 场景 |
|------|---------|------|
| `benches/tensor_ops.rs` | Tensor 底层操作 | clone/add/mul/negate/matmul/where（7 组） |
| `benches/backward.rs` | 节点反向传播 | Add/Negate/Subtract 链路 + MLP backward（4 组） |
| `benches/conv2d.rs` | Conv2d 卷积 | forward/full_step/two_layer_cnn（3 组） |
| `benches/end_to_end.rs` | 端到端训练步 | MLP(XOR/MNIST)/CNN(MNIST) × 多 batch_size（2 组） |
| `benches/smoke.rs` | 快速性能回归 | Tensor / Conv2d / MLP / CNN / Add backward 主链路 |
| `benches/pool2d.rs` | Pool2d | MaxPool2d / AvgPool2d forward + backward |
| `benches/optimizer.rs` | 优化器 | SGD / Adam step |
| `benches/normalization.rs` | 归一化层 | BatchNorm / LayerNorm / RMSNorm / GroupNorm |
| `benches/loss.rs` | Loss | MSE / CrossEntropy / BCE / Huber forward + backward |
| `benches/rnn.rs` | 循环层 | RNN / LSTM / GRU 小规模序列 forward + backward |
| `benches/attention.rs` | Attention | MultiHeadAttention self-attention / cross-attention forward + backward |

```bash
# 运行所有 benchmark
just bench

# 运行特定 benchmark
just bench-backward

# 保存基准线并对比
just bench-save before
just bench-compare before
```

报告输出在 `target/criterion/` 目录。

### I. Mode 执行上下文统一（2026-04-29）

**原始问题**：旧 `is_eval_mode` 同时承担层行为与 backward 缓存控制；中间过渡设计 `ExecutionContext { training, grad_enabled }` 把两者拆成两个正交字段。对 only_torch 当前训练 / 验证 / 推理 / 演化评估目标来说，"训练分支 + 不缓存"或"推理分支 + 仍缓存"不是核心用例，继续保留会让节点接入和测试覆盖度爆炸。

**解决方案**：用单枚举 `Mode { Train, Inference }` 统一三件事：层行为切换、backward 缓存策略、`backward()` 是否被允许（详见 [`mode_design.md`](design/mode_design.md)）。`Graph::load_model()` / `Graph::from_onnx()` 默认进入 `Mode::Inference`，即推理分支 + 不缓存 backward + `backward()` 直接报错。已接入 mode 的重缓存节点：Dropout、BatchNorm、Conv2d、Softmax、LogSoftmax、LayerNorm、RMSNorm、Abs、Square、Pow、Clip、Reciprocal、Ln、Log2、Log10。重构后的后续性能回归统一沿用 `post-mode-refactor` baseline 对比，验证 `Mode::Train` 路径无回归 + `Mode::Inference` 路径节省内存。
