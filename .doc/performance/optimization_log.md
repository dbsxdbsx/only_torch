# 性能优化实施战报（append-only）

> 已实施优化的完整测量现场与教训，**倒序排列**（最新在上），新战役直接在顶部追加条目。
> 与 CHANGELOG 的分工：CHANGELOG 记一行结论，本文保留 before/after 数据、命令、归因过程与教训。
> 修剪规则：当某条战报的 baseline 被完全取代、且教训已沉淀进设计文档时，整条移入 `_archive/`（带归档说明头）。
> 候选与否决见 [optimization_candidates.md](./optimization_candidates.md)；流程与 baseline 台账见 [benchmark_workflow.md](./benchmark_workflow.md)。

---

## N. 训练 batch 组装融合 + owned 入图路径（2026-07-03，组装/入图流量 2.2×，全框架受益）

**动机**：「零拷贝到底（借用 strided view 指向 buffer）」评估判死（论证见
[candidates 已否决项](./optimization_candidates.md#借用型张量视图零拷贝到底2026-07-03-判死)）后，
Reviewer 识别出两刀廉价替代：图像训练路径每样本 obs 数据经手 3 次——
① `assemble_stacked_obs` 反量化堆叠出 per-sample 中间 `Vec<f32>`；② `stack_obs`
把 G 个样本 Vec 复制拼进 batch flat；③ `graph.input(&Tensor)` 深拷贝进 BasicInput
节点。①② 可融合为一次物化，③ 是纯冗余。

**方案**（一次一项纪律下同属"拷贝消除"单主题）：
- **融合组装**：`UnrollItem.obs_t/next_obs` 由 `Vec<f32>` 改为 `ObsSource<'a>` 枚举
  （`Owned` / `Single(&StoredObs)` / `Stacked{steps,t,stack}`），batch 组装时
  `append_into` 从 buffer 借用帧**直写最终 flat**（边反量化边堆叠，零中间 Vec）；
  `assemble_stacked_obs` 降级为 `#[cfg(test)]` 语义参照，新增守门测试锤逐 bit 等价。
- **owned 入图**：新增 `Graph::input_owned(Tensor)`（move 语义）；`IntoVar for Tensor`
  与 `LossTarget for Tensor`（含标量广播）改走 owned 路径。`train_unroll_batch` 内
  obs/action onehot/各目标张量全部 move 入图；recon/consistency 需同一数据两份时
  仅 clone 一次（旧路径两处各 clone）。
- **顺手清账**（同一根因、机械替换）：DataLoader `extract_tensor_batch` /
  `VarLenDataset::get_batch`、SAC `transitions_to_batch`、PPO `rollout_to_batch` 的
  `Tensor::new(&vec)` 改 owned move；`ema_update` 与演化 mini-batch `set_value` 改
  `set_value_owned`。

**实测**（新增 `benches/obs_batch_assembly.rs`，Pong 口径 frame=84²/stack=4/G=16，
release+MKL；两路径输出先断言逐 bit 相等再计时）：

| case | 旧 | 新 | 加速 |
|---|---|---|---|
| 组装（per-sample→stack vs 融合直写） | 978 µs | 426 µs | **2.3×** |
| 入图（`input` clone vs `input_owned`） | 784 µs | 419 µs | **1.9×** |
| 端到端（组装+建 Tensor+入图） | 993 µs | 447 µs | **2.2×** |

绝对量级：单次 batch 组装省 ~550 µs × 每局 64 次训练 × 多 group ≈ 图像线每局省
零点几秒——符合评估时「收益确定但不巨大」的预期；真正的价值是**消除结构性冗余**
且全框架受益（DataLoader / SAC / PPO / 演化同享 owned 路径），API 零心智负担
（`IntoVar`/`LossTarget` 对用户透明，owned 参数天然表达 move）。

**验证**：`just test` 全量 0 失败（RL 271 含新增 `ObsSource` 逐 bit 守门测试，
锤融合组装与参照实现逐 bit 等价）+ `smoke-rl` 7 目标全过。语义恒等论证：Flat 路径
`ObsSource::Single` 与旧 `to_f32_vec` 值序恒等、RNG 消耗不变、图结构与节点值不变
（owned 入图只改所有权转移方式）。3-seed CartPole 哨兵**未复跑**——ndarray 0.16
升级后哨兵基线重定在册待办（有 seed 满额不收敛），当前跑数字不可解读，等基线
重定后一并回归。

---

## M. 图像 obs u8 量化帧存储 `StoredObs`（2026-07-03，Pong 单局 wall 3~7× + 增长斜率归零）

**动机**：Phase 1 Pong 实测「单局耗时随 buffer 占用增长、buffer 满（Ep32）后进入平台」
（Run A：Ep1 2.4s → Ep26 65s → 平台 65~110s）。零克隆采样（战报 J 前置修复）已消掉
clone 流量后，残余病灶 = **工作集本身**：单帧以 f32 存（84²=28KB/帧），32 局 buffer
≈ 800MB，每局 64 次训练 × 随机抽 16 局 × 组装 ~11MB 堆叠批次——随机读几乎每读必冷，
内存带宽成瓶颈。曾候选的「借用型张量视图」救不了这个（省的是最后一次 memcpy，
省不了 800MB 工作集的冷读），已判死。

**方案**：`src/rl/buffer/obs.rs` 新增 `StoredObs` 枚举——`F32(Vec<f32>)` 直通（向量
obs，与旧字段逐 bit 同语义）/ `U8(Vec<u8>)` 像素量化帧；`SelfPlayStep.obs` 换该类型
（`From<Vec<f32>>` 保扩展者人体工学）。量化决策在 **`ObsAdapter` 源头**按 env 观察
空间声明显式做出（不做数据猜测，`ReplayBuffer<T>` 保持内容无感知）；acting 滑窗
（`ImagePipe.frames`）与训练组装（`assemble_stacked_obs`）吃**同一份**量化语义。
**f32-only 计算契约不变**：u8 是存储休眠编码（类比磁盘 PNG），进 `Tensor` 前已反量化。

**量化口径**（图像域行为改变，一次一项）：resize 输出（0–255 像素域）round 为 u8，
读取反量化 `u8/255` → [0,1]；相对旧「f32 直存 v/255」每像素误差 ≤ 0.5/255
（DQN→MuZero 系标准做法）。buffer 800MB → 200MB，批次组装读写量同步 ¼。

**实测**（`MAX_EP=60 PROFILE=1` release+MKL，与 Run A 同命令同 seed）：

| 口径 | 旧（f32 帧） | 新（u8 帧） |
|---|---|---|
| 逐局 wall 曲线 | Ep1 2.4s → Ep26 65s → 平台 65~110s | **Ep5~Ep60 全程平坦 ~10s，无增长斜率** |
| 健康环境同期对照（重启后 3-seed 跑 Ep11~13） | 25~33s 且仍在爬 | 9~10s 平 |
| 60 局总 wall | —（Run A 为降级环境，不可比总量） | 716.8s（含 eval） |
| PROFILE 分解 | batch_prepare 曾 67.7s/6局（J 前） | self_play 139.8s / train_step 481.3s / eval 113.9s / batch_prepare **0.01s** |

平台期对比 ~7×、健康环境同期对比 ~3×，且**增长斜率本身消失**（Ep5 与 Ep55 同为
~10s，跨过 buffer 满点无拐点）——锤死「工作集 × 随机读」归因；排除 swap（旧进程
RSS 仅 ~1GB）与「agent 变强局变长」（局长钉在 800~1000 步、计算量结构固定）。
60 局零崩溃（Ep53 被动陷阱干净通过）。

**验证**：RL 单测 270 全绿（新增量化往返/F32 直通/量化堆叠组装 3 个单测）+
`smoke-rl` 7 目标全过；Flat 路径语义恒等（值与 RNG 消耗序均不变）。CartPole 哨兵
复测被「ndarray 0.16 BLAS 派发漂移 → 哨兵基线重定」在册待办遮蔽（见 CHANGELOG
deps 条目），与本改动无关；Pong 账本尚无历史数字，量化无「数字失效」问题，
口径已写入 pong README。

---

## L. Tensor 构造接口统一：`from_vec` 并入泛型 `new`（2026-07-03，API 收口，性能中性）

**动机**：优化 J 引入的 `Tensor::from_vec(Vec, shape)` 与 `Tensor::new(&[f32], shape)`
签名几乎一致、仅所有权语义不同，双入口造成"该用哪个"的用户成本。参考对照：Burn
`TensorData::new` 单入口收 owned `Vec`；PyTorch 的 `tensor` / `from_numpy` 分名是因为
**可观察语义不同**（别名共享），而我们两者产物完全等价（独立所有权、连续布局），
只有构造成本差异——该场景 Rust 惯例是单入口 + 参数类型静态分派。

**方案**：新增 `IntoTensorData` trait（`Vec<f32>` 零拷贝 move / `&Vec` / `&[f32]` /
const-generic `&[f32; N]` 复制一次），`new` 改 `impl IntoTensorData` 入参，**删除
`from_vec`**（未进任何发布版本，无兼容包袱）。关键细节：不能用 `impl Into<Vec<f32>>`
——泛型参数不触发 unsize coercion，`&[1.0, 2.0]` 数组字面量（全库测试数百处调用形状）
会全部编译失败，const-generic impl 是保住零改动兼容的必要件。全库 28 处 `from_vec`
调用点机械迁移；`random`/`normal`/`arange`/`eyes` 构造器内部顺手改 owned 传参
（各免一次 memcpy）；ONNX 导入 `Cow` 权重改 `into_owned()` 传入（Owned 分支由复制转 move）。

**验证**：双口径全量 3320 测试全绿；clippy 179（低于改动前 181，顺手清了 5 处
needless-borrow）；同窗基线 `pre-unify-new`（smoke + pool2d + my_zero_forward 共 18 case）
对比——初跑 4 项名义回归中 3 项复测翻转 No change；唯一持续项 `max_pool2d_backward/b32`
+5~7% 经 **stash 旧代码同窗对照**复跑 +11.7%（旧代码反而更差）→ 锤死环境漂移归因，
非代码回归。泛型单态化后 `new(Vec)` / `new(&slice)` 生成代码与旧 `from_vec` / 旧 `new`
逐指令等价，符合"零运行时开销"预期。**全量 bench 与基线重录均无必要**，
`post-hotpath-opt` 基线继续有效（本文 J/K 表格中的 `from_vec` 字样即现今
`new(owned Vec)` 路径）。

---

## K. Pool2d 平铺直写 + BatchNorm 反向单趟融合（2026-07-03，bench 绝对值审计驱动）

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

## 依赖升级：ndarray / pyo3 / numpy（2026-07-03 全部完成；ndarray 0.17.2 + pyo3 0.29 + numpy 0.29）

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
结论：**依赖升级在测量能力范围内性能中性**；干净的新栈基线待安静窗口录制。

**冷水**：优化 J 踩的 `&a + &b` 广播慢路径上游 0.16/0.17 均未修（最后一次广播性能优化
是 0.15.2），Add 三路分流修复升级后仍必要。

---

## J. 热路径分配/拷贝消除批量优化（2026-07-03，commit `3fd498a`）

**来源**：全库分配/拷贝审计（自审 + Reviewer 压力测试三轮收敛），动机是 MCTS 推理热路径
（每次 recurrent 约 25~40 节点 × 50 sims，全是 [1,N] 小张量，分配开销占主导）与训练路径的无谓整块拷贝。

**方法论**（与以往单项优化不同，本次为批量实施）：

- **正确性逐项锁死**：所有行为敏感改动配「fused vs 旧链」逐 bit 金测试（共 9 个），
  逐元素运算顺序刻意保持与旧张量表达式链一致 → 批量叠加不放大正确性风险；
- **性能统一测量**：单项收益多在 Criterion 噪声底（2~5%）以下，仅聚合端到端 bench 可测；
  bench 组与改动面近似对角映射（`adam_step`→优化器融合、`conv2d_*`→Conv2d 直写、
  `my_zero_forward`→解码路径…），若某组回归可用定向 bench + 局部 revert 快速归因；
- **行为改变类一律不混批**（如 min_max_normalize 去 repeat，见[候选项 #3](./optimization_candidates.md#3-热路径审计留档项2026-07-03批量优化-j-时降级未做) C2）。

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
噪声带内的名义回归不构成代码回归证据；跨环境百分比不混为同一趋势。

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

---

## I. Conv2d Inference 推理快路径（2026-04-29）

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

**设计结论**：卷积的数学前向在 Train / Inference 下相同，但执行引擎需要区分"是否要为 backward 保存缓存"。后续新增重算代价高、缓存占用大的节点时，应同时设计训练路径和推理路径，避免推理承担训练负担。

---

## Mode 执行上下文统一（2026-04-29）

**原始问题**：旧 `is_eval_mode` 同时承担层行为与 backward 缓存控制；中间过渡设计 `ExecutionContext { training, grad_enabled }` 把两者拆成两个正交字段。对 only_torch 当前训练 / 验证 / 推理 / 演化评估目标来说，"训练分支 + 不缓存"或"推理分支 + 仍缓存"不是核心用例，继续保留会让节点接入和测试覆盖度爆炸。

**解决方案**：用单枚举 `Mode { Train, Inference }` 统一三件事：层行为切换、backward 缓存策略、`backward()` 是否被允许（详见 [`mode_design.md`](../design/mode_design.md)）。`Graph::load_model()` / `Graph::from_onnx()` 默认进入 `Mode::Inference`，即推理分支 + 不缓存 backward + `backward()` 直接报错。已接入 mode 的重缓存节点：Dropout、BatchNorm、Conv2d、Softmax、LogSoftmax、LayerNorm、RMSNorm、Abs、Square、Pow、Clip、Reciprocal、Ln、Log2、Log10。重构后的后续性能回归统一沿用 `post-mode-refactor` baseline 对比，验证 `Mode::Train` 路径无回归 + `Mode::Inference` 路径节省内存。

**Mode 重构对比结果**：

- `just bench-compare pre-execution-context` 跑到 `smoke` 时因 benchmark case 改名中断；中断前 `tensor_ops`、`conv2d_forward`、`backward`、`end_to_end` 多个分组相对重构前 baseline 已显示明显改善，但该 baseline 不再作为后续门禁基线。
- 已复跑 `cargo bench --bench smoke --features blas-mkl -- --baseline pre-execution-context`；6 个 smoke 项全部显著改善，关键链路未复现回归。
- 已保存 `post-mode-refactor` 完整新基线，并补跑 `smoke` 到 `attention` 后续分组；新命名 benchmark 全部跑通。
- 行为回归已用 invariants、梯度流、模型加载、BatchNorm / Conv2d 节点测试，以及 MNIST / MNIST GAN / CartPole SAC / chess YOLO ONNX example 验证；CartPole SAC 训练达到单回合 200，但三次测试平均 185.7，低于示例目标 190，仍为随机训练波动范围内。

---

## H. 节点 cache clone 消除（2026-02-15）

| 节点 | 优化前 | 优化后 |
|------|--------|--------|
| Conv2d `padded_input` | `Some(padded.clone())` 缓存 | `Some(self.pad_input(input))` 直接 move |
| LeakyReLU | 缓存完整 `parent_value` | 不再缓存，反向时用 `value`（输出）判断区域，数学等价 |
| ChannelBiasAdd | `let mut result = input.clone()` | 节点已删除，由通用 Add + 广播替代 |

---

## G. BLAS 可选支持 — Intel MKL / OpenBLAS（2026-02-15）

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

---

## F. 优化器 set_value_owned + Adam 中间变量优化（2026-02-15）

| 改动 | 效果 |
|------|------|
| `TraitNode` 新增 `set_value_owned(Tensor)` | 优化器更新参数时零拷贝（消除 `set_value(Some(&val))` 的 clone） |
| SGD/Adam `step()` 改用 `set_value_owned` | 每个参数更新省一次完整 Tensor clone |
| Adam 偏差修正因子外提到循环外 | 所有参数共享 `bc1`/`bc2`，省去重复计算 |
| Adam `grad_sq *= (1-β2)` 原地操作 | 省去 `scaled_grad_squared` 临时 Tensor |
| Adam `denom += ε` 原地操作 | 省去 `&v_sqrt + eps` 临时 Tensor |

---

## E. Conv2d 反向传播并行策略（2026-02-15）

**原始问题**：反向传播中每个 batch 样本独立做 `im2col + GEMM`，N 次小矩阵乘法。

**尝试方案**：`batch_im2col` 将所有样本拼成大矩阵做单次 GEMM。

**最终结论**：批量化方案被**撤回**。因为 MKL 配置为 `seq`（单线程，避免与 Rayon 冲突），
单次大 GEMM 只用一个核心，反而不如 per-sample Rayon 并行（多核各跑一个小 GEMM）。
参考 Burn/Candle/PyTorch 的做法后统一为 per-sample Rayon 路径。
启用 BLAS 后 `dot()` 内部自动用 MKL 加速，代码无需区分。

---

## D. GradResult 零拷贝梯度传递（2026-02-15）

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

---

## C. 反向传播全局优化（2026-02-14）

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

---

## B. Conv2d im2col + GEMM 优化（2026-02-14）

| 改动 | 效果 |
|------|------|
| 前向/反向卷积从嵌套循环改为 im2col + ndarray `.dot()` | 完整训练步 2.6-4.4x 加速 |
| 利用 ndarray 底层 matrixmultiply 库的 AVX2 自动向量化 | 无需引入外部 BLAS |

---

## A. 赋值算子减少 clone（早期）

| 位置 | 改动 | 效果 |
|------|------|------|
| `GradientAccumulator::get_average_gradient` | `gradient.clone() / scalar` → `gradient / scalar` | 少一次 Tensor clone |
| `graph.rs` 梯度累加 | `current + contribution` → `current += &contribution` | 避免临时张量分配 |
| 反向传播 | clone `upstream_grad` → 借用引用 | 避免大 Tensor 拷贝 |
