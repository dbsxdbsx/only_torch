---
status: active
created: 2026-06-18
updated: 2026-07-02
owners: []
reviewers: []
---

# MyZero · Pendulum-v1 失败诊断（进行中，求专家审阅）

> **状态**：active —— 诊断实验跑了一半（base + 臂 A 完成，臂 B/C 未跑）。
> **优先级提示（2026-07-02，纲领 §2.3 战略转向）**：Pendulum 已降级为**非关键路径**（真实目标为象棋 + 图像游戏，均不在连续动作轴上）；本 issue 保持诊断态、不作发版门禁，value-head 坍缩根因待有余力或连续需求出现时再推进。文中 CartPole 数字为 pre-autograd-fix 旧口径（新基线见[账本](../../examples/my_zero/cartpole/README.md)）。
> **审阅结论（2026-06-18）**：方法论成立——失败区间内不下组件裁决。已据此把两份 README 的 Pendulum `➖` 改回 `⏳`（无信号），保留诊断旋钮，删除原始日志。
> **诊断更新（2026-06-18，dynamics 探针）**：base 150ep 跑 dynamics 诊断（对比 model 想象 vs 真实）——**reward head 健康**（预测 std 1.86 ≈ 真实 1.87），**value head 坍缩成常数**（预测 std 26 vs 真实 MC return std 175；episode 末尾真实剩余 −10 仍预测 −500）。**病根精确锁定在 value 学习**，不在 reward / 搜索 / 离散粒度——这也再次**证伪**「reward 分辨率头号嫌疑」。下一步：先区分 value target 本身坍缩（`td_steps`/`gamma`）vs head 学不动（网络/loss/lr），再对症，不盲改。
> **背景对话**：本 issue 源于一次对话中关于"consistency 在 Pendulum 上是否为中性"的争论。原作者认为：当前 Pendulum 所有配置都在"失败区间"（−353 ~ −1445，门禁 −200），在模型根本没学会的任务上做组件消融，➖ 裁决全是噪声，不足以回答中性问题。故转向"先查清为什么学不会"。
> **关联文档**：[MyZero 总览](../../examples/my_zero/README.md) · [Pendulum 详情](../../examples/my_zero/pendulum/README.md) · [RL 路线图](../../.doc/design/rl_roadmap.md)
> **代码位置更新（2026-06-18，v0.25 Phase 0/1 重构）**：MyZero 已统一进库 `src/rl/algo/my_zero/`——模型在 `network.rs`（原 `examples/my_zero/cartpole/model.rs`，consistency 块在其 `train_unroll`），训练循环 / dynamics 诊断在 `runner.rs`，旋钮在 `config.rs::apply_env_overrides`（原 `pendulum/main.rs` 的 `Hyperparams` / env 解析），动作离散化在 `action.rs`。组件（consistency / value_prefix / n_step / support 等）已从 muzero/ez **吸收进 my_zero**。本 issue 下文旧的 `model.rs:NNN` / `main.rs:NNN` 行号据此失效，对应逻辑见上述新文件。
> **value-head 容量诊断（2026-06-18，决定性新结果）**：库内单测 `my_zero::tests::value_head_capacity` 喂高方差、obs 可分的 value 目标，只训 repr+pred 的 value head——高/低价值组预测间隔从 1.77 训练到 **14.00**（真实间隔 14.0，**精确拟合**）→ §一 的分叉 **(b)「head 学不动」证伪**。坍缩来自**上游**（n-step target 构造 / 搜索），非 head 表达力。下一步探针：在 Pendulum 跑里实测 n-step target 的 std，与全程 MC std(≈175)、模型预测 std(≈26) 三者对比，定位是 target 截断（td_steps/gamma）还是搜索。
> **压测更新（2026-06-21）**：已接入与 CartPole 相同的 **consistency + reconstruction + Sampled** 栈（B=7 · K_eff=5 · sims=20 · γ=0.997 · r_scale=0.1）；修复 Sampled `policy_target` 投射 full action_dim（ep10 训练崩溃）。600ep / 120k env-steps：**best greedy −942.2**（门禁 −200 未达标）。详见 §十。
> **事实源锁定（2026-06-24）**：`recipe.rs` 中 Pendulum 仍复用 `consistency + reconstruction + Sampled`，但代码命名与文档口径统一为**诊断栈**，不是已验收 promote；`DIAG=1` 诊断将输出 MC return / n-step target / search root / predicted value 的分布，作为下一步 P0 证据。
> **论文口径审计（2026-06-25）**：修复 Sampled MuZero `π̂_β` 公式错误：应为 `(β̂/β)·π`，不是 `β̂/(β·π)`。复核 consistency 后确认 `negative_cosine_similarity()` 内部已对 target branch `detach()`，原实现已有 stop-gradient；剩余差距是没有独立 EMA target encoder / target projector。旧 §十 压测结果基于修复前 Sampled 公式，后续需重跑。
> **transition discount 语义重构（2026-06-25）**：MyZero 将 `terminated / truncated / continuation` 作为基础 backbone 语义接入：真终止 `continuation=0`，time-limit truncation 仍 `continuation=1` 并 bootstrap；Dynamics 学习 continuation，MCTS imagined edge 使用 `gamma * predicted_continuation` backup。该改动修正 terminal/truncation 闭环，但不宣称 Pendulum 已解决，仍需以 TD=5 诊断和 greedy eval 判读。
> **TD=5 + continuation 实测（2026-06-25）**：600ep / 120k env-steps 仍未达标；best greedy **−1085.2 @ ep200**，final greedy **−1252.2**。DIAG：MC return std **232.1**，n-step(td=5) std **28.1**，search root std **23.7**，network root std **21.7**；value 链路仍被压扁，continuation 语义闭环不是 Pendulum 的充分修复。
> **continuation 二值门修复实测（2026-06-25）**：search discount 由 soft `γ·c` 改 binary `γ·(1−done)` 后复跑（td=5 · 600ep / 120k env-steps）：best greedy **−959.3 @ ep250**、final −1256.5，仍远未达标（门禁 −200）。解读：二值门移除了软折扣对健康边 value 的虚假压低（best 较 soft 的 −1085 略好），但 **Pendulum 主瓶颈是上游 value-head 坍缩**，非 search discount，故此修复在 Pendulum 上仅中性偏正、不构成充分修复。注：CartPole 同一修复把样本效率从 30.2k 修回 ~13.1k（确认软折扣是 CartPole 回归主因）。

---

## 一、核心问题（待专家定夺）

**MyZero 的"纯离散化 MCTS"方案，在 Pendulum-v1 上、`num_simulations ≤ 50` 的合理算力约束下，到底能不能学会？**

这个问题之所以关键：`pendulum/main.rs:5` 的设计者注释早已埋下伏笔——
> "忠实 Gumbel 连续搜索留作后续——**仅当离散化触顶（达不到 SAC 水平）时才需要**"

现在的实测在喊：离散化方案连 −200 的门禁都摸不到（最好 −929），**这不是"触顶"，是"没学会"**。答案直接决定 MyZero 的技术路线：

- **如果能学会**（调参/加大离散档数后能到 −200 甚至 SAC 的 ~−150）：那 base vs +consistency 的对照才有意义，consistency 的 ✅/➖/❌ 才是可信裁决。
- **如果根本学不会**：结论就不是"某组件中性"，而是"**纯离散化 MCTS 这条路在连续控制上走不通，必须提前上 Gumbel-root / 连续采样候选**"——矩阵里 Gumbel-root 那一行要从"可选增强 ⏳"升级为"关键路径"。

**请专家重点审**：方法论（§三）、判读门限（§四）、已得负面结果（§五）是否支持上面的分叉判断。

---

## 二、已完成的代码改动（纯 plumbing，未改算法语义）

为了让 A/B 干净（不每次重编译），给 `examples/my_zero/pendulum/main.rs` 加了 3 个环境变量旋钮。**这些改动不动任何算法逻辑**，只是把原本硬编码的 `const` 提升为可配置：

| 旋钮 | 原 const | 默认值 | 改动点 |
|------|---------|--------|--------|
| `NUM_ACTIONS` | `const NUM_ACTIONS: usize = 9` | 9 | `main.rs:31-79` 新增 `Hyperparams` 结构体 + `from_env()` |
| `RSCALE` | `const REWARD_SCALE: f32 = 0.1` | 0.1 | 同上 |
| `LR` | 无（固定 `cfg.base.lr=0.02`）| 0.02 | `main.rs:644-648` 加 env 覆盖（与 GAMMA/SIMS 同模式）|

附带：
- `idx_to_torque` 加 `num_actions` 参数；`self_play_one_episode` 加 `reward_scale` 参数；`run_one_training` 加 `hyper: &Hyperparams` 参数（纯参数透传，签名已 `#[allow(clippy::too_many_arguments)]`）。
- `justfile` 加 `smoke-my-zero-pendulum` recipe（SMOKE 管线验证，已通过）。
- 两份 README 的环境变量支持列表已同步。

**改动已 `cargo check` + SMOKE 验证通过**，未提交（`git status` 见工作区）。专家若认为方向不对，可整体 revert；若认可，可作为后续 sweep 的基础。

---

## 三、诊断方法论（请专家审）

### 3.1 关键论点：消融必须在"学会的任务"上做

原作者的核心判断（请重点审这个逻辑是否成立）：

> CartPole 上 consistency 判 ✅，是因为它的价值是**真实的**：loss 从 9.6 掉到 0.7（一个数量级），avg_R 从 80 拉到 97。那是在"模型正在学习"的曲线上测出来的增益。
>
> Pendulum 上现在判的 `consistency ➖`，它的**真实含义**不是"consistency 在 Pendulum 上无效"，而是**"我连这个任务都没跑通，所以我看不出 consistency 有没有用"**。这两个结论差别巨大。

类比：10 分类任务，随机水平 10% 准确率。现在测 base/+dropout/+BN，准确率全在 9%~11% 之间晃——不能下结论说"这些 trick 都中性"，只能说"模型根本没学到东西，消融毫无信息量"。

### 3.2 判读口径（与项目 README 一致）

- **唯一成功判据**：greedy(temp=0) eval 10 局均值。
- **排除**：`avg_R`（self-play 带 Dirichlet 探索噪声，永远偏低）；`loss`（仅辅助看是否 NaN/爆炸）。
- **看 trajectory**（轨迹形态：上扬/平直/振荡），不看单点终点。

### 3.3 已排除的怀疑点（读代码 + 数值校验，见 §四脚注）

| 怀疑点 | 状态 | 依据 |
|--------|------|------|
| value support 溢出 | ✅ **排除** | `support.rs:69` `clamp(-20,20)`，变换域 [−20,20] 覆盖原始 value ±440；Pendulum 最差 value target ≈ −316（`h=−17.0`），不溢出 |
| sims 不够 | ⏸ **硬约束** | MuZero/EZ 上限就是 50 sims；用户领域知识："不靠加算力解决"。9 动作 × 50 sims → 每根动作平均 ~5.5 次访问 |
| reward head 分辨率（RSCALE）| ❌ **实测证伪** | 见 §五臂 A，RSCALE 0.1→0.5 不仅没解锁学习，反而略差 |
| `CONSISTENCY` 实现 | ⚠️ **存疑待查** | 见 §六，偏离 SimSiam 设计，但 CartPole 开 cons 反而学得好 → 非致命 bug，须 base 对照定性 |

**新增确认根因（2026-06-18，dynamics 诊断）**：`value head 坍缩`——base 150ep 跑 dynamics 探针，value 预测 std=26 vs 真实 MC return std=175（坍缩成近似常数；episode 末尾真实剩余 −10 仍预测 −500）；reward head 反而健康（预测 std 1.86 ≈ 真实 1.87）。**病根锁定在 value 学习**，上表 reward 分辨率「证伪」由此再获印证。待区分：value target 本身坍缩（`td_steps`/`gamma`）vs head 学不动（网络/loss/lr）。

---

## 四、判读门限（用户拍板）

看 greedy eval 轨迹**最高点**（排除 avg_R/loss）：

| 轨迹最高点 | 含义 | 后续动作 |
|-----------|------|---------|
| **−800 ≈ "质的飞跃"** | 该旋钮**解锁学习**，方法通 | 进 A/B 正式定性 |
| **−600** | 接近门禁一半，强解锁 | 同上 |
| **持续在 −1000 附近徘徊** | 失败区间，该旋钮无效 | 排除该旋钮 |

> 用户原话："分数至少要比前代高出 200 分左右。现在一直在 −1000 左右徘徊，能达到 −800 就是质的飞跃，−600 更好。"

**决策树**：
```
所有臂都在 [-1000 附近徘徊]?
├─ 是 → 确认"纯离散化 MCTS 在 Pendulum、sims≤50 约束下走不通"
│       → Gumbel-root 升级为关键路径
└─ 否，某臂触达 [-800, -600]：
        该臂 vs base 的唯一差异旋钮 = 头号根因
        → 在该旋钮上做 base vs 该旋钮 A/B，正式定性 consistency / CQ
```

---

## 五、已跑实测（seed=42）

> 日志：原始训练日志不入库，关键 eval 数值已摘录在下方各臂表中。
> ⚠️ **方法学警示**：均为**单 seed**，且 base 因命令传递问题实际跑了 ~300ep（非预设 300，见 §七坑点）；臂 A 跑满 300ep。仅作 trajectory 形态判断，不作精确数值裁决。

### 臂 base（全关，RSCALE=0.1，NUM_ACTIONS=9，sims=50）

18 个 eval 点（每 25ep 一次），greedy eval 轨迹：

```
−1714.7 → −1340.3 → −1366.9 → −1108.1 → −1074.9 → −1138.5
→ −1380.2 → −1214.0 → −1493.2 → −1321.1 → −1149.9 → −1433.7
→ −1367.9 → −1430.8 → −1253.0 → −929.6 → −1470.5 → −1421.6
```

- **最高点 −929.6**（env_steps=80k）；整体在 **−929 ~ −1715** 区间振荡。
- self-play avg_R 稳死在 −1100~−1213（平直）；loss 稳死在 ~16~17（**没下降**）。
- **裁决：失败区间确认**（−1000 附近徘徊，远离 −800 门槛）。

### 臂 A（+RSCALE=0.5，其余同 base）—— 跑满 300ep，有 benchmark

12 个 eval 点：

```
−1327.7 → −1271.8 → −1251.6 → −1236.1 → −1138.6 → −1284.2
→ −1173.9 → −1348.1 → −1440.7 → −1271.5 → −1134.7 → −1312.9(终)
```

`[benchmark] seed=42 env_steps=60000 wall_clock=755.5s greedy_eval=-1312.9`

- **最高点 −1134.7**（env_steps=55k）；整体在 **−1134 ~ −1440** 区间。
- **裁决：失败区间，且比 base 略差**（base 最高 −929，臂 A 最高 −1134）。

> 🔑 **关键负面发现**：原作者押注的"reward 分辨率是头号嫌疑"**被证伪**。RSCALE 0.1→0.5（reward head 有效原子从 ~1.6 格 → ~3 格）不仅没解锁学习，反而略差。这**排除**了"reward 表示域分辨率不足"作为主因——瓶颈在别处（dynamics 误差累积？模型容量？离散化结构本身？）。

### 臂 B（+NUM_ACTIONS=25） / 臂 C（+CONSISTENCY=1）—— **未跑**

原作者暂停在此，转而写本 issue 求审。

---

## 六、`CONSISTENCY` 实现的存疑点（请专家重点判）

`examples/my_zero/cartpole/model.rs:432-443`（Pendulum 通过 `#[path]` 复用）：

```rust
if consistency_coef > 0.0 {
    if let Some(next_obs) = next_obs_list.and_then(|list| list.get(i)) {
        let repr_target = self.repr.forward(&Tensor::new(next_obs, ...))?;  // 对 next_obs 正常前向
        let proj_target = self.projector.forward(&repr_target);              // ← 同一个 projector
        let proj_online = self.projector.forward(&next_latent);              // ← 同一个 projector
        let pred_online = self.predictor.forward(&proj_online);
        let cons_loss = negative_cosine_similarity(&pred_online, &proj_target)?;
        ...
    }
}
```

**与 SimSiam（Chen & He 2021）/ EfficientZero 标准实现的差异**：

1. **`proj_target` 与 `proj_online` 走同一个 `self.projector`**。标准 SimSiam 是 online encoder + online predictor vs **target encoder（stop-gradient，且 target encoder 是 online 的 EMA）**。这里 target 路径没有独立的 target projector / EMA。
2. **target branch 已有 stop-gradient**：`negative_cosine_similarity()` 内部会对 `proj_target` 调用 `detach()`；因此这里不是“无 stop-gradient” bug，而是“无 EMA / 独立 target encoder”的简化实现。

**但**：CartPole 上开 `CONSISTENCY` 反而学得显著更好（loss 9.6→0.7，一个数量级；avg_R 80→97），说明它**不是致命 bug**，而是**环境敏感**。

**请专家判**：
- 这个实现是"简化版 SimSiam（可接受）"还是"实现错误（应修）"？
- 它的"CartPole 有效 / Pendulum 中性"的分裂，是因为实现问题，还是因为 Pendulum 本身在失败区间（任何消融都无信号）？

---

## 七、已踩的坑（接手者必读）

1. **`cmd` 的 `set VAR=x && cargo run` 未必把环境变量传给子进程**。原作者第一次跑 base 用 `set MAX_EP=300 && cargo run ...`，结果 MAX_EP 没生效，程序按默认跑了 ~300ep（`unwrap_or(600)`，但因 timeout 在更高 ep 被砍）。**可靠做法**：用 PowerShell `$env:MAX_EP='300'; cargo run ...`，或 `just` recipe。
2. **后台任务 timeout**：默认 1200s（20min）。base 全关 300ep ≈ 7-12min，但 release 首次编译 + pyo3 初始化会吃掉前 ~1min。若一次跑多个臂要串行（pyo3 有导入竞态 + CPU 抢占会让单臂 wall 翻倍）。
3. **日志编码坑**：用 PowerShell 重定向训练输出到 `.log` 时，中文会因 GBK/UTF-8 混杂出现乱码（原始日志已不入库）。下次要留存日志建议显式指定 UTF-8，或直接读终端输出。

---

## 八、建议的下一步（供专家拍板，原作者不预设）

1. **方法论层面**：§3.1"消融必须在学会的任务上做"是否成立？如果成立，Pendulum 的 consistency ➖ 裁决应否**降级为"无信号/待定"**而非"中性"？
2. **是否继续跑臂 B/C**？还是先回答 §一的核心分叉（离散化方案能否学会）？
3. **`CONSISTENCY` 实现（§六）是否需要修正**为标准 SimSiam（独立 target encoder + EMA + stop-grad）后再做消融？
4. **是否承认"纯离散化 MCTS 走不通"并提前启动 Gumbel-root**？还是再试更多离散化变体（非均匀离散化、更大 NUM_ACTIONS）？

---

## 九、原作者的倾向（仅供参考，非结论）

基于 §五的负面结果（RSCALE 证伪 + base 失败区间），原作者倾向认为：
- 当前数据**已倾向支持**"纯离散化 MCTS 在 Pendulum sims≤50 下走不通"这个分叉。
- 但因单 seed + 只跑了 2 个臂，**证据尚不充分**，故 suspension 求审而非直接下结论。
- 若专家认可方法论，最低成本的收尾是：补跑臂 B/C（确认无逃脱）+ 多 seed 复现 base，然后据此决定 Gumbel-root 是否升级为关键路径。

---

## 十、cons+recon+Sampled 压测（2026-06-21）

> **配置**：与 CartPole promote 相同组件栈（consistency + reconstruction + Sampled）；`recipe.rs` 注入 B=7 bin 中点；`main.rs` 仅 `reward_scale(0.1)` + 门禁 −200。训练默认 **sims=20 · γ=0.997**（与 CartPole 对齐，非旧 README 的 sims=50/γ=0.99）。
> **修复**：Sampled 子集 policy_target 须投射回完整 action_dim（`target.rs::scatter_policy_target`）；否则 ep10 训练起点 Tensor shape 崩溃。

### 实测（seed=42 · release · 600 ep · 120k env-steps · wall ~495s）

| 指标 | 值 |
|------|-----|
| best greedy eval | **−942.2** @ ep575 |
| 训末 greedy | −1118.9 |
| load best → eval×10 | **−942.2** |
| run×1 | −867.4 |
| loss 轨迹 | 48 → **~5–7**（较 §五 base ~17 卡死有进步） |
| self-play avg_R | ~−1239 → ~−1032 |
| 门禁 −200 | ❌ 未达标 |

greedy eval 里程碑：ep25 **−1338** → ep50 **−1041** → ep100 **−1037** → ep200 **−1033** → ep500 **−994** → best **−942** @ ep575。

### 与 §五 base 对照

| 配置 | best greedy | loss | 备注 |
|------|-------------|------|------|
| §五 base（9 档 · 无 cons/recon · sims=50） | ~−929 @ 80k | ~17 不降 | 旧栈 |
| **§十 cons+recon+Sampled（B=7 · sims=20）** | **−942** @ 115k | ~5–7 | 本次 |

### 解读

- **仍处失败区间**（远未触达 −800「质的飞跃」线），**不能**对 cons/recon/Sampled 下 ✅/➖/❌ 裁决。
- cons+recon+Sampled **改变了训练动力学**（loss 降、单局 R 偶发 −660~−750），但 **greedy 策略未入门** → 与 §一 value 上游 / 搜索弱信号假设仍一致。
- Sampled K=5<B=7 + sims=20 是否额外伤害，**未隔离**；待 P1：`sims=50` / `γ=0.99` A/B。

### 下一步（P0→P2）

1. **P0**：n-step target std 探针（`.diagnose()` 或 runner 日志），区分 target 截断 vs 搜索。
2. **P1**：`main.rs` 只改 `sims=50`、`gamma=0.99`，MAX_EP=300 看 greedy 能否过 −800。
3. **P2**：仍失败 → 评估 Gumbel-root 或加大 B；勿在失败区间叠 completedQ 等 CartPole ❌ 组件。
