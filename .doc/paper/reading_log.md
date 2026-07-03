# 论文阅读日志（累积）

> **用途**：记录"哪些论文已经读过、判定是什么、结论落在哪"，避免重复阅读与重复论证。
> 判定口径：**采纳**（产出行动项）· **证据归档**（不行动但作为决策依据）· **否决**（对本项目不适用，写明理由）· **跳过**（读前言/摘要即可定性）。
> PDF 一律本机留存不入库；需要复读时按 arXiv 号重新下载。

---

## 2026-07-02 · CPU 优化 + RL 样本效率批次（7 篇清账）

背景：v0.25 收口后讨论"RL 主线要不要做底层算力优化"，产生一份论文清单（A/B/C/可跳过四档）。本批次把清单全部读完清账。核心结论先行：

1. **内核级优化路线整体否决**（Winograd / GEMM 布局传播 / 三值内核 / 快速矩阵乘）——我们的 GEMM 刻意外包给 `matrixmultiply` / MKL，这条线若某天实用化会经由上游 BLAS 库免费继承，不需要也不应该自己实现。详见 [optimization_candidates.md 已否决项](../performance/optimization_candidates.md#已否决项)。
2. **唯一采纳项是 Simulus（RL 样本效率）**——三个组件拆解挂靠 v0.26 P0/P1，见 [Simulus 组件采纳消融计划](../design/my_zero_simulus_ablation_plan.md)。
3. 方法论沉淀：值得精读的只有"必须改我们自己代码才能兑现"的思想；kernel/编译器层的成果等上游。

| arXiv | 论文（缩写） | 原档位 | 判定 | 一句话理由 | 结论落点 |
|---|---|---|---|---|---|
| 2509.26217 | CPU 卷积算法能耗横评（OneDNN 实测 direct/im2row/gemm/wino） | A | ✅ 证据归档 | 单算子 wino 最快，但**整网推理数据搬运反噬，隐式 GEMM 全面胜出**；1×1 快路径与 batch=1 场景假设与我们吻合 | Winograd 盖棺 → [已否决项](../performance/optimization_candidates.md#已否决项)；隐式 lowering 记为图像线唯一潜在方向 |
| 2502.11537 | Simulus（世界模型版 Rainbow，planning-free SOTA） | A | ✅ **采纳** | RaC/HL-Gauss、loss 优先回放、sg 解耦三组件与 MyZero 兼容且便宜；planning-free 路线是 CPU-only 风险的实证退路 | [消融计划](../design/my_zero_simulus_ablation_plan.md) + [CPU 风险 issue](../../.issue/items/cpu_only_mcts_image_realtime_risk.md) |
| 2604.04599 | LP-GEMM（打包布局跨 GEMM 传播） | B | ❌ 否决 | 要求自持 GEMM 内核源码 + 框架级布局描述符；我们 GEMM 一行外包。其"矩阵越小内核外开销占比越高"的洞察，对我们的同构物是框架层分配/拷贝，已另行处理 | 本日志 |
| 2604.20913 | FairyFuse（三值 LLM 无乘法 CPU 推理） | B | ❌ 否决 | 前提三缺三：无量化基础设施（f32-only）、网络小到塞进 L2（无带宽瓶颈）、AVX-512 手写内核违背跨平台。Roofline"极限量化天然属于 CPU"的论证值得记住 | 本日志 |
| 2308.13898 | 复杂拓扑 DNN 内存感知调度（ILP + 最优性保持融合） | B | ❌ 否决·留一思想 | 解决"边缘设备峰值内存放不下"，我们桌面 CPU 无此问题；其穷人版思想（推理期按 liveness 及早释放中间节点 value；实施陷阱见候选项正文）值得留档 | [待优化项 #4](../performance/optimization_candidates.md#待优化项) |
| 2606.13408 | 快速矩阵乘算法目录（frontier-closure 搜索） | 可跳过 | ❌ 确认跳过 | 双线性复杂度社区的文献数据库，不是优化方案；4×4=47 次乘法等结果在 F2 有限域，对 f32 无意义 | 本日志 |
| 2602.11041 | 张量分解结构利用（6×6 指数 2.8075→2.8016） | 可跳过 | ❌ 确认跳过 | 纯理论；作者自认 n≥1000 才赢操作数且"不代表实现中有加速"；DL 场景行业（MKL/oneDNN）从未启用 Strassen 系 | 本日志 |
| 2512.18453 / 2411.16152 / 2209.12982 | C 档 Winograd 专档三篇（NOVA / ARMv8 / Tap-Wise） | C | 📦 未读·整档划掉 | 随 2509.26217 的 Winograd 盖棺失去阅读前提（含"未来图像线网络变大"场景，理由见已否决项） | — |

**"网络变大后是否翻案"已检验**：即便 ResNet 级 CNN，im2col 后 GEMM 维度停留在几百~两千（如 256×2304×196 高长方形），低于 Strassen 系实测交叉点（方阵 n≈2000–4000），且 f32 数值稳定性劣化、行业无先例——结论对可见未来的大网络同样成立。
