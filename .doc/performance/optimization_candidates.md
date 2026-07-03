# 性能优化候选项（决策面）

> 本文档只回答两件事：**还有什么可做**（待优化项，每项必须带触发条件）与**什么已被否决**（防止重复发明）。
> 条目实施完成后整体移入[实施战报](./optimization_log.md)，不在本文留尸体；验证流程与 baseline 台账见 [benchmark_workflow.md](./benchmark_workflow.md)。
> 最后更新: 2026-07-03

---

## 待优化项

### 1. RNN 场景 `select` + `set_value` 二次复制问题

**位置**：`src/nn/layer/rnn.rs` → `forward`

**现状**：每个时间步两次数据复制。当前规模开销可忽略（~1.2 MB 总冗余）。
注：Tensor 存储 Arc/CoW 化（战报 O）后 `set_value` 的 clone 已是浅拷贝，
本项残余问题只剩 `select` 一侧的物化，价值进一步缩水。

**状态**：暂缓（YAGNI），等 RNN 处理大规模数据时再实施

---

### 2. RNN 场景更多优化

参见 #1，等 RNN 处理大规模数据时统一评估

---

### 3. 热路径审计留档项（2026-07-03，批量优化 J 时降级未做）

2026-07-02/03 全库分配/拷贝热路径审计（[战报 J](./optimization_log.md)）中识别、经 Reviewer 评估后**刻意不随批实施**的项。共同特点：真实存在但缺 profiler 证据排进热点前列，或改动会触碰图结构/行为（需单独 3-seed 重验），不值得为其扩大批量改动的验证面。

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

### 4. 演化 SupervisedTask mini-batch 路径 per-epoch 整集 clone-shuffle

**位置**：`src/nn/evolution/task.rs` → `SupervisedTask::train` mini-batch 分支

**现状**：每个 epoch 把整个训练集（train_x + 全部 head 的 train_y）`clone()` 一份再原地 `shuffle_mut_seeded`。MNIST 规模下不构成热点，但数据集变大后是 O(数据集大小 × epoch 数) 的纯冗余拷贝。

**推荐方案**：改为索引 shuffle——shuffle 一个 `Vec<usize>` 索引数组，batch 切片时按索引 gather；不需要引入 DataLoader 即可实施。

**状态**：暂缓（YAGNI）。触发条件 = 演化任务上大规模数据集（如图像演化线）且 profiling 显示 shuffle/clone 进入热点。届时若同时撞到内存墙，与 [future_enhancements.md 演化对接项](../_archive/future_enhancements.md#9-演化系统对接项2026-07-03-清账) 的 DataLoader 接入合并评估。

---

### 5. 推理模式中间节点 value 及早释放（liveness）

**来源**：arXiv 2308.13898（内存感知调度）的"穷人版"思想——执行顺序与张量生命周期决定峰值内存；论文本体（ILP 调度器）对我们不适用，见[阅读日志](../paper/reading_log.md)。

**现状**：`Mode::Inference` 已跳过 backward 缓存，但前向期间所有节点的 `value` 存活到结束。核心机制：前向拓扑序执行时统计每个节点的剩余消费者数，归零即 `clear_value()`。

**⚠️ 实施陷阱（2026-07-03 评估补记，改动量远超早先"十几行"预估）**：
朴素的「消费者归零即清值」会踩碎 MyZero 持久化推理子图的 **sink 模式**——多头输出用
`concat(latent, policy, value_logits)` 拼 sink 作唯一 forward 目标，forward 后再从
**中间节点**用 `with_value` 读值（`network.rs` RootInfer / RecurrentInfer，MCTS 每次
simulation 都走）：sink 算完时这三个节点消费者恰好归零 → 被清 → 后续读值 panic。
正解至少需要：① **外部引用豁免**——节点被 `Var` 持有时 `Rc::strong_count` 大于
子节点引用数，只清"纯中间节点"（隐式契约，需不变量测试兜底）；② 消费者计数按
本次 forward 可达子图统计（`NodeInner` 只知 parents，需拓扑序构建时顺带算）；
③ 验证与 `pass_id` 跳算、`inference_scope` 的交互。

**状态**：暂缓（YAGNI）。桌面 CPU + 当前网络规模峰值内存远非瓶颈（现役负载全是
`[1,N]` 小张量）；触发条件 = 图像线大 batch CNN 或受限部署目标真实撞到内存墙。
届时先做这个，再考虑任何调度方案。

---

### 6. 小张量高频创建路径的 Arc 分配开销（2026-07-03，战报 O 遗留观察项）

**来源**：Tensor 存储 Arc/CoW 化（[战报 O](./optimization_log.md)）唯一持续回归——
每次 Tensor 产出多付一次 Arc 控制块分配，`my_zero_forward`（[1,64] 级小张量、
MCTS 推理密集）+6%/+7%。大张量路径被 clone 类收益远远盖过（cnn_train_step -12.7%）。

**状态**：暂缓（YAGNI，Candle/Burn 同价结构）。触发条件 = profiler 证明小张量
Arc 分配进入热点前列；届时候选方向 = 小张量池化 / 输出缓冲复用，不走"退回深拷贝"。

---

### 7. 视图类算子（narrow / select / split / slice / get）改 Arc 共享视图（2026-07-03，战报 O 后续候选）

**来源**：Tensor 存储 Arc/CoW 化（战报 O）后 Reviewer 复查发现：这些算子仍走
`to_owned()` 物化拷贝，而 `ArcArray` 原生支持 `slice_move` / `index_axis_move`
等 O(1) 共享视图（保留整块缓冲 + offset/strides）。热路径消费者：演化 mini-batch
切片（`task.rs` narrow）、attention 逐 head 切片（`attention.rs` narrow）、RNN
时间步 select（候选 #1）。

**为什么不随战报 O 一起做**：共享视图会让产物变成带 offset / 可能非连续的张量，
下游 `flatten_view` / `data_as_slice` 等连续性假设路径行为改变（panic 面扩大）+
整块缓冲存活期延长（内存驻留语义变化），破坏战报 O「语义恒等、验证便宜」纪律。

**状态**：暂缓。触发条件 = profiler 证明上述任一 narrow/select 调用点进入热点前列；
届时单独战役实施（需全量测试 + 连续性假设审计，不需 3-seed）。

---

## 已否决项

### 借用型张量视图（零拷贝到底）（2026-07-03 判死）

**原设想**：训练张量以借用 strided view 直指 replay buffer 内存（类 PyTorch
`from_numpy` 共享存储），消除组装批次的全部拷贝。

**否决理由**（各自独立成立，评估含 Reviewer 压测）：
1. 图像路径 3 次数据经手中，① 是 **u8→f32 dtype 转换**（视图本质消不掉，除非计算层
   支持 u8，违反 f32-only 契约）；② 是**随机样本 gather**（batch 来自随机 episode/
   时间步 + 帧堆叠边界重复填充，单 strided view 表达不了，PyTorch collate 同样物化）；
   仅 ③ 入图 clone 是纯冗余——已由战报 N 的 owned 路径消除。
2. 借用生命周期穿不过持久 Rc 计算图（节点 `Option<Tensor>` 跨 step 存活），逼 unsafe
   裸指针会让 buffer 变成训练图的隐式所有者，回放淘汰/重分配全是雷。
3. 成熟 Rust 库（Candle/Burn）均走共享所有权而非借用方案；若未来真需要，走上方
   候选 #6（Arc/CoW），不走借用视图。



### 内核级优化路线：Winograd / 手写 GEMM / 三值内核（2026-07-02 论文清账盖棺）

**结论**：卷积与矩阵乘的**内核层**优化整体不做，含"未来图像线网络变大"场景。7 篇论文判定明细见[论文阅读日志](../paper/reading_log.md)。

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
