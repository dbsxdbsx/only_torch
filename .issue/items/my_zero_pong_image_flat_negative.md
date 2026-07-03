# Pong 图像基准负结果：150 局 × 3 seed 学习曲线平直（2026-07-03）

> **⚠️ 时效声明（2026-07-03 晚）**：本跑基于 `b697e95`，**先于** `adfc02f` 框架修复批
> （per-seed env.reset 种子派生——本跑 3 seed 共享同一 reset 序列、统计独立性存疑；
> GroupNorm 图内可微修复；温度调度显式常数化）。数字降级为**历史参考**，
> 图像支柱的正式裁决以新数值流下的 S2 复跑为准；本文嫌疑人清单仍有效。

## 现象（官方 S2 基准，预注册口径）

- **命令**：`SEEDS=3 cargo run --example my_zero_pong --release --features blas-mkl`（image base 臂：consistency ON + recon OFF + two-hot + raw obs）
- **结果**：seeds 42/43/44 = best greedy **−20.4 / −21.0 / −21.0**，中位 −21.0，0/3 达标（门槛 −18，随机 ≈ −20.7）
- **曲线形态**：每 seed 14 个 eval 点 greedy 几乎恒 −21（仅偶发 −20.0/−20.4/−20.9），self-play avg_R 全程 −20~−21 无趋势；**loss 在下降（0.8 → −4.5 级）但完全不转化为策略改进**
- 每 seed 满 150 局 ≈ 134k env-steps、wall ~26 min；450 局全程零 panic（Ep53 陷阱一并无事）
- 按 [Phase 1 计划 §S2](../../.doc/design/rl_phase1_image_plan.md) 预注册判读：「平直 = 记负结果 issue」→ 本档案

## 数据位置

- 完整日志：`/tmp/pong_3seed_official.log`（含 PROFILE 分解）
- 账本行：[pong README](../../examples/my_zero/pong/README.md)

## PROFILE 快照（seed 42，供诊断参考）

wall 1548s = train_step 1030s + self_play 290s + eval 245s；
MCTS 内 root CNN 前向 867µs/call、recurrent 16µs/call——**性能不是嫌疑人**，纯学习信号问题。

## 嫌疑人清单（按先验排序，未裁决）

1. **训练量不足（先验最强）**：Atari-100k 口径下 MuZero 系通常需要远超 64 trains/ep × 150 局的优化步数才在 Pong 脱离随机；且 lr=0.003 为拍脑袋保守值，从未在图像域标定。
2. **表征坍缩 / consistency 在图像域失效**：CartPole 域 cons 表现暧昧（Phase 0 悬案），S3 的 cons-off 臂本来就要测。
3. **recon OFF 导致 CNN 无自监督信号**：CartPole 域 recon_coef=16 是最大杠杆（66.2k→9.8k），图像域 base 臂却关着 recon——S3 recon 臂或为最大嫌疑。
4. **稀疏 reward + γ=0.997 支撑不住**：Pong 单局 ~900 步只有 ±21 个非零 reward，value 学习信号稀薄。
5. sims=20 偏少 / 温度退火节奏（后 50% 才降温，满局数跑完退火才到 0.25）。

## 下一步（待裁决）

1. **先 S2 复跑**（新数值流：`adfc02f` 之后 + 后续框架级改动收敛后）——旧跑 3 seed 共享
   reset 序列，复跑本身可能改变结论；
2. 复跑仍平 → S3 三臂 A/B（半额 75 局 × 3 seed，~40min/臂）**兼当诊断**：recon 臂测嫌疑 3，
   cons-off 臂测嫌疑 2（recon coef pilot {1,4,16} 载体已备：`pong_image_ablation_bench.rs`）；
3. 若三臂全平 → 嫌疑 1/4 上位，评估加预算（更多局数/更高 replay ratio/lr 扫描）或 DIAG 诊断
   （value/reward 预测是否坍缩）；
4. 图像支柱「成立/不成立」的终局裁决在上述证据齐后再下，本 issue 保持 open。

## 关联

- [Phase 1 计划](../../.doc/design/rl_phase1_image_plan.md) §S2/S3 · [收口规划 §2](../../.doc/design/rl_closure_plan.md)
- [Ep53 panic 调查](./pong_ep53_panic_investigation.md)（本轮 450 局零复现，佐证「转被动监视」裁决）
