---
status: active
created: 2026-07-02
updated: 2026-07-02
owners: []
reviewers: []
---

# 商业实时图像游戏目标画像 × MyZero 能力缺口登记

> **性质**:需求登记 + 差距清单,非 bug。纲领 [§2.3](../../.doc/design/my_zero_algorithm_vision.md#23-战略目标与优先轴2026-07-01-定稿) 的「商业图像游戏」战略目标已有**真实标的**(某商业 2D 横版格斗类游戏,已完成 Gym 环境封装与观测/动作/奖励设计;细节属私有项目,本 issue 只保留匿名化技术画像)。
> **用途**:图像线(Phase 1 起)每步推进时对照本画像,避免把管线做成「只服务 Atari 方图」的窄通道。

## 一、目标画像(匿名化,2026-07-02 实录)

| 维度 | 目标口径 | 与当前 MyZero 图像管线的差异 |
|------|----------|------------------------------|
| 图像 obs | **16:9 非方形**(如 256×144)灰度;横版卷轴裁方形会丢 ~44% 横向视野,不可接受 | 当前 `ConvRepresentationNet` 假设 side×side 方图(84²) |
| 帧堆叠 | 滑窗 **3 帧**(60Hz 抓帧,50ms 跨度;二阶差分够算加速度) | 当前常量 STACK=4(Atari 惯例);堆叠数应可配 |
| 复合观测 | **Dict obs** = 图像 + 按键状态 MultiBinary(~17 维) + 时间标量(POMDP 部分恢复 Markov 性的刻意设计) | 当前表征只吃单一图像;需「CNN + 辅助向量 MLP」双分支融合表征 |
| 动作空间 | **MultiDiscrete**(如 [4,4,16] = 256 联合动作) | `ActionAdapter` 仅单维离散/单维连续,多维 assert;且 \|A\|=256 ≫ sims → Sampled/Gumbel 的 native 场景(Phase 2 Gomoku \|A\|=225 复裁正好覆盖此域) |
| 实时预算 | 决策周期 ~**50ms**(20Hz),环境不暂停;步内像素检测(reward 来源)自身占 ~30ms | spike 外推:CNN root(≈4× 84² 像素量)~8–13ms + sims=50 想象 ~2ms ≈ **10–15ms,放得下且非瓶颈** |
| POMDP | 重度:对手离屏/隐身/遮挡、随机地图、外观干扰 | **真正未决项**——learned latent dynamics 在此能否成立是研究级问题,帧堆叠只部分恢复;见 §三 |
| 样本成本 | 真实在线匹配,一局约数分钟,单机日采 ~300–500 局 | 样本效率是 MyZero 谱系(EfficientZero 立身之本)的主场,反而是选型优势 |

## 二、关键洞察:「MCTS 实时延迟硬伤」的旧否决已被本框架实测推翻

该商业标的的私有设计文档曾**否决 MuZero 系**,算术是:Python/PyTorch 栈单次推理 ~5–6ms × 50 sims ≈ 300ms,80ms 周期内只能跑 2–4 sims。
本框架 Phase 1 spike([数据与协议](cpu_only_mcts_image_realtime_risk.md))实测:被 sims 放大的单元(flat-latent MLP recurrent)仅 **0.03ms**,50 sims 想象共 ~1.5ms;完整 acting 单步(含树簿记)sims=50 仅 3.9ms(84² 口径)。**旧否决否决的是「Python 栈上的 MuZero」,不是 MuZero 本身**——wall-clock 维度上该标的对 MyZero 重新开放。私有项目侧的复议记录由私有仓库自行维护,本 issue 不承载其内容。

## 三、能力缺口清单(按依赖序;均为表征/适配层,dynamics/prediction 零改动)

- [ ] **矩形输入**:`ConvRepresentationNet` 从 `side` 参数化为 `(h, w)`(卷积栈本身宽高比无关);`obs_pipeline` 预处理目标尺寸可配、支持非方形。**触发**:图像基准跑通后、或该标的接入前。
- [ ] **堆叠数可配**:STACK 从常量提升为 recipe 注入(3/4 皆常用)。
- [ ] **Dict obs 双分支表征**:图像 CNN + 辅助向量 MLP → 融合 latent;`ObsAdapter` 相应扩展(env 侧 Gymnasium Dict space 解析 GymEnv 已部分支持,Tuple 路径在)。
- [ ] **MultiDiscrete 动作适配**:`ActionAdapter` 支持多维离散(联合枚举或 factored);与 Sampled K 公式对接(\|A\|≫sims 域)。**依赖**:Phase 2 Gumbel/completedQ 复裁结论。
- [ ] **POMDP 表征验证**(研究项,非工程项):帧堆叠 + learned dynamics 在遮挡/隐身域的学习能力——纲领 §6 已留 history/帧堆叠插件位(**非** BetaZero,§5.3 已裁决)。**依赖**:Phase 1 图像基准先证明干净域(Pong)管线在学。

## 四、触发与不做的边界

- 本清单**不进 v0.26 关键路径**:Phase 1 仍按预注册协议以 Pong(方图 84²)收口;上表缺口在「图像支柱成立」之后、按标的接入需求逐项兑现,每项照旧一次一项 + 消融纪律。
- 公开仓库**不落**该商业标的的具体游戏名、私有路径、环境实现细节;本画像的数字口径(分辨率/帧率/动作维度)为通用技术参数,不构成识别信息。
