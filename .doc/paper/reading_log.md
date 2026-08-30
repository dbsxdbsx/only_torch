# 论文阅读日志（累积）

> **用途**：记录"哪些论文已经读过、判定是什么、结论落在哪"，避免重复阅读与重复论证。
> 判定口径：**采纳**（产出行动项）· **证据归档**（不行动但作为决策依据）· **否决**（对本项目不适用，写明理由）· **跳过**（读前言/摘要即可定性）。
> PDF 一律本机留存不入库；需要复读时按 arXiv 号重新下载。

---

## 2026-07-02 · CPU 优化 + RL 样本效率批次（7 篇清账）

背景：v0.25 收口后讨论"RL 主线要不要做底层算力优化"，产生一份论文清单（A/B/C/可跳过四档）。本批次把清单全部读完清账。核心结论先行：

1. **内核级优化路线整体否决**（Winograd / GEMM 布局传播 / 三值内核 / 快速矩阵乘）——我们的 GEMM 刻意外包给 `matrixmultiply` / MKL，这条线若某天实用化会经由上游 BLAS 库免费继承，不需要也不应该自己实现。详见 [optimization_candidates.md 已否决项](../performance/optimization_candidates.md#已否决项)。
2. **唯一采纳项是 Simulus（RL 样本效率）**——三个组件拆解挂靠 v0.26 P0/P1，见 [RL 状态总览 · 处方表](../../.doc/design/rl_myzero_status.md#34-处方表跨域路由规则)。
3. 方法论沉淀：值得精读的只有"必须改我们自己代码才能兑现"的思想；kernel/编译器层的成果等上游。

| arXiv | 论文（缩写） | 原档位 | 判定 | 一句话理由 | 结论落点 |
|---|---|---|---|---|---|
| 2509.26217 | CPU 卷积算法能耗横评（OneDNN 实测 direct/im2row/gemm/wino） | A | ✅ 证据归档 | 单算子 wino 最快，但**整网推理数据搬运反噬，隐式 GEMM 全面胜出**；1×1 快路径与 batch=1 场景假设与我们吻合 | Winograd 盖棺 → [已否决项](../performance/optimization_candidates.md#已否决项)；隐式 lowering 记为图像线唯一潜在方向 |
| 2502.11537 | Simulus（世界模型版 Rainbow，planning-free SOTA） | A | ✅ **采纳** | RaC/HL-Gauss、loss 优先回放、sg 解耦三组件与 MyZero 兼容且便宜；planning-free 路线是 CPU-only 风险的实证退路 | [RL 状态总览 · 处方表](../../.doc/design/rl_myzero_status.md#34-处方表跨域路由规则) + [CPU 风险 issue](../../.issue/items/cpu_only_mcts_image_realtime_risk.md) |
| 2604.04599 | LP-GEMM（打包布局跨 GEMM 传播） | B | ❌ 否决 | 要求自持 GEMM 内核源码 + 框架级布局描述符；我们 GEMM 一行外包。其"矩阵越小内核外开销占比越高"的洞察，对我们的同构物是框架层分配/拷贝，已另行处理 | 本日志 |
| 2604.20913 | FairyFuse（三值 LLM 无乘法 CPU 推理） | B | ❌ 否决 | 前提三缺三：无量化基础设施（f32-only）、网络小到塞进 L2（无带宽瓶颈）、AVX-512 手写内核违背跨平台。Roofline"极限量化天然属于 CPU"的论证值得记住 | 本日志 |
| 2308.13898 | 复杂拓扑 DNN 内存感知调度（ILP + 最优性保持融合） | B | ❌ 否决·留一思想 | 解决"边缘设备峰值内存放不下"，我们桌面 CPU 无此问题；其穷人版思想（推理期按 liveness 及早释放中间节点 value；实施陷阱见候选项正文）值得留档 | [待优化项 #4](../performance/optimization_candidates.md#待优化项) |
| 2606.13408 | 快速矩阵乘算法目录（frontier-closure 搜索） | 可跳过 | ❌ 确认跳过 | 双线性复杂度社区的文献数据库，不是优化方案；4×4=47 次乘法等结果在 F2 有限域，对 f32 无意义 | 本日志 |
| 2602.11041 | 张量分解结构利用（6×6 指数 2.8075→2.8016） | 可跳过 | ❌ 确认跳过 | 纯理论；作者自认 n≥1000 才赢操作数且"不代表实现中有加速"；DL 场景行业（MKL/oneDNN）从未启用 Strassen 系 | 本日志 |
| 2512.18453 / 2411.16152 / 2209.12982 | C 档 Winograd 专档三篇（NOVA / ARMv8 / Tap-Wise） | C | 📦 未读·整档划掉 | 随 2509.26217 的 Winograd 盖棺失去阅读前提（含"未来图像线网络变大"场景，理由见已否决项） | — |

**"网络变大后是否翻案"已检验**：即便 ResNet 级 CNN，im2col 后 GEMM 维度停留在几百~两千（如 256×2304×196 高长方形），低于 Strassen 系实测交叉点（方阵 n≈2000–4000），且 f32 数值稳定性劣化、行业无先例——结论对可见未来的大网络同样成立。

---

## 2026-07-22 · 百度 XWORLD 语言接地（2 篇精读）

背景：从 README 引用追溯到百度 XWORLD 论文，结合 myAGI 方向（显式推理 + 可解释 AI）精读两篇。核心结论：**架构原则（显式解耦 + 共享表示）比具体 CNN/RL 实现更有长期价值**，与 myAGI "松耦合 + 人类可检视中间表示"理念高度一致。

| arXiv | 论文（缩写） | 判定 | 一句话理由 | 结论落点 |
|---|---|---|---|---|
| 1703.09831 | XWORLD 语言接地框架（2D 迷宫 + 自然语言导航，Baidu Research） | ✅ 证据归档 | 显式接地（解耦语言理解与动作决策）+ 共享词嵌入实现 zero-shot 组合泛化；架构原则有参考价值 | [XWORLD 笔记](./XWORLD_Baidu_2017/summary.md) |
| 1802.01433 | XWORLD 完整版（ICLR 2018，增加交互式 QA 对话 + teacher 提问 + 课程学习） | ✅ 证据归档 | 同上扩展版；新增交互式对话证明"双向提问"可行；课程学习对未来复杂环境有参考 | 同上 |

**关键启示摘要**（详见笔记 §7）：

1. **显式接地 = 松耦合**：把"理解含义"和"做决策"拆成两步 + 人类可读中间表示（注意力图），与 myAGI 的 InputAdapter → 候选逻辑公式 → 推理引擎 路线一致。
2. **共享表示 → 跨任务迁移**：同一张词嵌入表服务多任务，小规模证明了"积累性学习"可行。
3. **局限性反面教训**：模板化语言（无容错）、无路径规划、单次指令（无修正）、agent 不能主动提问——恰好是 myAGI 需要解决的四个核心问题。
4. **与 only_torch MyZero 的切点**：双 loss 联合训练架构类似、课程学习可参考、latent state 的可解释性是未来差异化方向。

---

## 2026-08-14 · 最大熵树搜 vs 官方 Zero 搜索（温度收口）

背景：温度收成评测开关（非评测 \(\tau=1\) / 评测 \(\tau=0\)）后，核对 MENTS 家族与官方 Gumbel 线是否还能并进 MyZero。核心结论：**走 PUCT / Sampled / Gumbel，不走最大熵树搜；玻尔兹曼选边无运行时零件可拆。** 论证见 [搜索派系与温度](../design/rl_myzero_search_lineage.md)。

| 出处 | 论文（缩写） | 判定 | 一句话理由 | 结论落点 |
|---|---|---|---|---|
| NeurIPS 2019 | MENTS（Shannon MaxEnt + 软 Bellman） | ❌ 否决（路径） | 软最优可以和回报最优不是同一条边；温度是正确性参数 | [search_lineage](../design/rl_myzero_search_lineage.md) |
| ICML 2021 | RENTS / TENTS（凸正则化 MCTS） | ❌ 否决（路径） | 换 KL / Tsallis，仍是树内 backup，不是出手 \(\tau\) | 同上 |
| JAIR 2024 / 2202.07071 | Dam 统一 backup / 探索 | ❌ 否决（路径） | 上篇期刊版；作者写明不和 MuZero 比 | 同上 |
| 2102.06808 | ANTS（目标熵反解树内 \(\tau\)） | ❌ 否决（路径） | 仍是 MaxEnt 目标；接 learned \(g\) 要改软 Bellman；引用少 | 同上 |
| NeurIPS 2023 / 2404.07732 | BTS / DENTS（玻尔兹曼选边） | ❌ 否决（纳入）· 证据归档 | 钉死软目标会推错；围棋实装退回平均回报 + \(\alpha(N)\to 0\)，赢的是吞吐 | 同上 |
| ICML 2020 / 2007.12509 | Grill（MCTS = 正则化政策优化） | ✅ 证据归档 | 官方线理论前身；不单独换实现 | 同上 |
| ICLR 2022 | Gumbel（少 sim 政策改进） | ✅ 采纳（组件在库，recipe 默认关） | 访不完所有边时的 correctness；promote 只认状态总览 | 状态总览 §3 + 同上 |
| ICML 2021 / 2104.06303 | Sampled MuZero | ✅ 采纳 | 大 \(\|A\|\) 抽 K；与 MaxEnt 无对打 | 状态总览 §3 + 同上 |

---

## 2026-08-19 · 主动推理两篇（Deep AI + pymdp）

背景：桌面清账对照 MyZero。表面像「学世界模型的前沿智能体」，疑问是能不能治「学得太慢」、能不能塞进 MuZero。核心结论先行：

1. **不是 MuZero 组件论文**，是另一条母算法：用变分自由能同时做感知、学习和行动，目标不是最大化回报。
2. **整包不接入 MyZero**。Deep Active Inference 用 Evolution Strategies 估不可微梯度，不走 autograd；pymdp 是离散 POMDP 的 Python 工具箱，对标 MATLAB DEM，不是树搜索增量。
3. **治不了「学得慢」**。Deep AI 只在 Mountain Car 上演示，优化器更吃样本；pymdp 面向小离散符号模型。MinAtar 便宜旋钮已三刀皆负，下一刀仍是等步数加训或继续暂缓，不是换范式。
4. **同族想法项目里已经试过并停掉**：主动数据 3A0 负裁、recurrent posterior 负裁、BetaZero / DreamerV3 整包明确不做。reconstruction 已是最近的可用亲戚（学着预测观察），不必再开一条自由能损失。
5. PDF 本机放到 `AI论文/Post-AI/`，按 arXiv / JOSS 号可重下。

| 出处 | 论文（缩写） | 判定 | 一句话理由 | 结论落点 |
|---|---|---|---|---|
| 1709.02341 / *Biol. Cybern.* 2018 | Deep Active Inference（深度网络 + ES 的主动推理） | ❌ 否决（路径）· 证据归档 | 换目标函数和优化器；Mountain Car 玩具实验，不提高 MyZero 样本效率 | 状态总览 §5.3 + 本日志 |
| JOSS 2022 / 10.21105/joss.04098 | pymdp（离散状态主动推理 Python 库） | ❌ 否决（路径）· 证据归档 | 软件短文，不是算法增量； discrete POMDP API，接不进 latent MCTS | 同上 |
