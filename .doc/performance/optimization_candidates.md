# 性能优化候选项（决策面）

> 本文档只回答两件事：**还有什么可做**（待优化项，每项必须带触发条件）与**什么已被否决**（防止重复发明）。
> 条目实施完成后整体移入[实施战报](./optimization_log.md)，不在本文留尸体；验证流程与 baseline 台账见 [benchmark_workflow.md](./benchmark_workflow.md)。
> 最后更新: 2026-07-05

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

参见 #1，等 RNN 处理大规模数据时统一评估。

**已出表项（2026-07-04，战报 R）**：RNN 输入投影批量化（`x_t @ W_ih` 逐时间步
小 GEMM 合并为一次 `[N*T, in]` 大 GEMM）已实施，forward -22%；**LSTM/GRU 同款
改造实测负收益已回滚**（4/3 路输入投影的 select/reshape 节点开销吃掉 GEMM 合并
收益，LSTM forward +8%、backward +20%），若未来权重合并为 `[in, 4H]` 单矩阵
（PyTorch 布局，参数结构破坏性变更）可重新评估。

---

### 3. 热路径审计留档项（2026-07-03 立项；同日战报 P 清账收口）

2026-07-02/03 全库分配/拷贝热路径审计（[战报 J](./optimization_log.md)）中识别、经 Reviewer 评估后**刻意不随批实施**的项。2026-07-03 复审裁决：哨兵红灯 + 系数复裁待跑 = 数值扰动类改动的唯一低成本窗口，除 C1 外全部实施（见[战报 P](./optimization_log.md)）。

| 项 | 去向 |
|---|---|
| A3 / B2 / B7 / B8 / D4 / E1 / C2 | ✅ 已实施（战报 P，2026-07-03） |
| B1（sigmoid 反向多趟临时） | ✅ 更早已随激活反向融合完成（`zip_map` 单趟，见 `sigmoid.rs` 注释），本表曾误留 |
| C1（`layer/attention.rs` 逐 head 循环建图） | ✅ 已实施（2026-07-04）：`Tensor::batched_mat_mul(_nt/_tn)` + `MatMul` 节点 3D@3D + attention 常数节点数重写，forward -75~77% / backward -70~80%（baseline `pre_c1`）；留档表就此**全部清账** |

---

### 4. 演化 SupervisedTask mini-batch 路径 per-epoch 整集 clone-shuffle

✅ **已实施（2026-07-04，战报 Q）**：索引 shuffle（`Tensor::shuffled_row_indices_seeded`
同置换契约 + `select_rows` 行 gather），批次构成与旧路径逐 bit 一致（契约金测试锁定），
每 epoch 整集 CoW 物化拷贝消除。

---

### 4b. 小分配残留观察池（2026-07-04 Reviewer 补充立项，统一等分配画像证据）

热路径审计 Reviewer 复审识别的候选，共同触发条件 = `alloc-profile` 画像（战报 Q 工具）
证明该项进入分配热点前列，逐项单独实施：

| 项 | 位置 | 现状 |
|---|---|---|
| 节点前向/反向父值 borrow `Vec` | `node_inner.rs` 执行路径 | 每节点每次执行分配 `Vec`，绝大多数节点 arity 1/2，可做 small-arity 快路（SmallVec/栈数组） |
| builder 形状 `Vec` 簇 | `graph/inner/node_builders.rs` | 每次建节点构造 `parent_shapes` / `parent_dynamic_shapes` 等多个 Vec，图构建密集路径（演化/MCTS 持久图一次性，权重低） |
| backward 种子张量 | `graph` backward 入口 `Tensor::ones(&[1,1])` | 每次 backward 一次小分配，可缓存 |
| `gather` 逐元素索引 | `tensor/ops/others.rs` | 前后向逐元素动态索引 Vec 构造；SAC/categorical 若成热点再做 2D 专用快路 |
| CE 前向 par `sum()` 确定性 | `softmax_cross_entropy.rs` | rayon sum 合并序随运行漂移（loss 标量 run-to-run 可能 ulp 抖动，梯度路径不经过）；**数值冻结解除后**改「map 收集 + 串行序累加」（同 dK 修复，战报 Q） |

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
等 O(1) 共享视图（保留整块缓冲 + offset/strides）。热路径消费者变迁：attention
逐 head 切片已被 C1（3D batched MatMul）整体消除、演化 mini-batch narrow 已被
索引 shuffle（候选 #4，战报 Q）替换为 `select_rows`（gather 本就必须物化）——
剩余消费者只剩 RNN 时间步 select（候选 #1）等低频点，触发条件更难满足，但条目
保留不删（未来新增视图类调用点时仍是正解方向）。

**为什么不随战报 O 一起做**：共享视图会让产物变成带 offset / 可能非连续的张量，
下游 `flatten_view` / `data_as_slice` 等连续性假设路径行为改变（panic 面扩大）+
整块缓冲存活期延长（内存驻留语义变化），破坏战报 O「语义恒等、验证便宜」纪律。

**状态**：暂缓。触发条件 = profiler 证明上述任一 narrow/select 调用点进入热点前列；
届时单独战役实施（需全量测试 + 连续性假设审计，不需 3-seed）。

---

### 8. CPU 训练效率远期簇（2026-07-05 RL 讨论沉淀，全部 YAGNI 留触发条件）

背景共识（讨论结论，防重复推导）：我们的 regime = 小网络（<1 MB，权重常驻单核私有
L2）+ 串行小活（MCTS batch=1 推理）+ MKL 单线程 GEMM——CPU 赢延迟、GPU 赢吞吐；
batch 的效率红利主要来自**框架 per-node 固定开销摊薄**（调用次数与 batch 无关）+
权重缓存复用摊薄，缓存效应平缓（MKL 分块自动打理），**batch 选择不需要显式建模
缓存**。各项按触发条件独立实施：

| 项 | 内容 | 触发条件 |
|---|---|---|
| **并行 self-play（治本档，seed 内）** | 每核一局、各持网络副本（权重常驻各核私有 L2，不抢 L3/带宽，预期近线性扩展）；AlphaZero 系标准架构的 CPU 多核版。⚠️ 改变数据新鲜度语义（对弈权重比训练线程略旧），须预注册 A/B。**工程形态注记（2026-07-07）**：Gomoku env 走 pyo3 + Python gymnasium，GIL 会串行化进程内多线程的 env 调用——届时选型 = 多进程 actor / 棋类线 Rust 直连 env（规则已在 `gomoku.rs`，金测试锁等价，哲学口径下棋类域合法）/ free-threaded Python | 便宜档落地后压力已消化，真正触发点 = **象棋线立项**或单 seed 墙钟再上一个量级；实施时与 seed 级并行叠乘（3 进程 × 每进程 N actor，核预算分好） |
| **✅ seed 级进程并行（便宜档，已落地 2026-07-07）** | 3-seed 臂的各 seed 完全独立（各自 RNG 流），每 seed 一进程同时跑：臂墙钟 ÷3（2.5h→<1h），**各 seed 数值轨迹与串行逐 bit 一致**（纯实验排程，零算法语义变化）。落地物 = 载体 `M3_SEEDS` 环境变量过滤（`gomoku_m3_bench.rs`，不设则全量串行向后兼容）+ `scripts/bench_m3_seedpar.sh`（预编译一次、分 seed 日志、全绿后自动拼接单一臂级日志）+ `just bench-m3-seedpar <臂名>`。口径：并行跑 t=/wall= 计时含资源竞争略胖，env_steps 与学习指标不变（wall-clock 本非评价指标）。实测背景 = ⑫⑬ 臂单进程仅用 32 线程中的 3–4 个（11% CPU，i9-13900HX 24 核） | 已实施（触发实录：⑫⑬ 臂 40–55 min/seed、2.5–3.5h/臂，2026-07-05 满足「self-play 墙钟成为迭代瓶颈」） |
| **MCTS 评估缓存 / 置换表** | 内存换计算：局面哈希 → 网络评估结果缓存（树内 + 跨局）。⚠️ 改变搜索语义（命中 = 不同的树），须按一次一臂单独消融 | sims 显著加大或象棋线（分支因子/局面重复率更高）落地时 |
| **自动 batch size** | `auto_batch = min(效率拐点, 统计临界)`：效率侧 = 训练前微基准扫 batch 实测 samples/sec 取拐点（缓存/SIMD/MKL 分块效应自动包含）；统计侧 = 梯度噪声尺度（McCandlish et al. 2018 critical batch size）。**首份实测锚点**（2026-07-05 Gomoku 探针，`.bench/gomoku_batch_scale_probe_20260705.log`）：每 env-step 成本 b16=5.0ms / b256=**4.4ms**（拐点）/ b512=5.3ms / b1024=7.3ms——摊薄红利 256 附近见顶，之后训练计算量盖过红利单调上爬，无悬崖、内存可忽略；**学习侧同 lr 同更新数下大 batch 单调劣化**（vs random 1.000→0.750，教科书「大 batch 欠训练」形态），佐证效率拐点 ≠ 最优 batch，统计侧上限必须参与裁决 | Gomoku batch 臂正信号 + 第二个域复现后，才值得做成框架机制（④臂已裁决梯度噪声非瓶颈，本项优先级进一步降低） |
| **KL 自适应 lr** | 参照实现（AlphaZero_Gomoku）机制：每次更新测 policy KL 位移，超标降 lr / 不足升——「用户不调 lr」的现成方案；linear scaling rule 可把 lr 与 batch 绑定再砍一个自由度 | naive0 issue 假设 5 裁决入口，或 batch 臂正信号后的 lr 联动臂 |
| **网络规模 × batch 重标联动** | 演化让网络大小变成运行时变量 → batch 吞吐拐点漂移。惰性规则：参数量变化 ≥ 一个数量级才触发微基准重标（结构变异是天然触发点）；缓存大小永不出现在接口里 | 演化线与 RL 线汇合（演化 RL 网络拓扑）时 |

---

## 已否决项

### `mat_mul` 系 IxDyn 直接 `dot` 卫生改写（2026-07-04 Reviewer 否决，同日复核修正论据）

**原设想**：ndarray 0.17 支持 IxDyn 直接 `dot`，可删 `mat_mul` 系的
`into_dimensionality::<Ix2>` 样板（依赖升级战报曾列为"可选后续"）。

**否决理由**（复核修正：初版「数值风险」论据已证伪，改按收益≈0 维持否决）：

- ~~初版论据「IxDyn 路径是否走完全相同 BLAS 派发无法担保」~~ **不成立**：读 ndarray
  0.17.2 源码（`impl_linalg.rs` 的 `Dot<ArrayRef<A, IxDyn>>`）确认 IxDyn `dot` 是
  薄包装——2D 分支内部同样 `into_dimensionality::<Ix2>` 后调同一条 Ix2 `dot`，
  逐 bit 等价、同一 BLAS 派发，数值风险为零。
- 维持否决的真实理由：可删样板仅 `mat_mul`/`_nt`/`_tn` 三处 ×6 行，集中在
  `src/tensor/ops/mat_mul.rs` 单文件、不扩散（新算子走 `general_mat_mul` /
  batched `Ix3` 路径，IxDyn dot 覆盖不到）；`Ix2` 显式写法把二维契约锁在类型层，
  比 IxDyn 运行时 match 更自文档化；`into_standard_dyn` F 序守卫需跟着改写成
  `ArrayD` 版，净省行数≈0。纯卫生收益不抵表达力损失；放着无利息、不阻塞任何
  后续（ndarray 升级/BLAS 切换/新增 op 均不依赖此处写法），不构成技术债。

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

**保留的唯一方向已清账**：隐式 lowering（分 tile 现场 im2row，2509.26217 的实践赢家）
✅ 已实施（2026-07-05，[战报 S](./optimization_log.md)）——col 驻留预算化（≤16 MiB 整批
物化驻留、逐 bit 冻结；超预算流式分 tile + dK 反向重算），大形状前向 -22~-40%、超预算
col 驻留内存归零；`benches/conv2d.rs` 真形状裁决器组（board15/19、atari84/42、b128）随批
入库。内核层「不自写 GEMM/Winograd」纪律不变。

### `&[&NodeHandle]` vs `&[NodeHandle]`

**结论**：`&[&NodeHandle]` **不推荐**。双重间接引用增加指针追踪开销、缓存局部性差。

### 前向传播 clone NodeHandle 问题

**原始问题**：`NodeHandle.clone()` 深拷贝内部 `Tensor`。

**结论**：v2 动态图架构已彻底消除此问题。`NodeInner` 由 `Rc` 管理，前向传播通过 `borrow()` 零拷贝借用父节点值，无需 clone `NodeHandle`。
