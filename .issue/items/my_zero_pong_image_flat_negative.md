# Pong 图像基准负结果：150 局 × 3 seed 学习曲线平直（2026-07-03）

> **✅ S2 复跑已完成（2026-07-04，新数值流 = 哨兵复裁收口后 HEAD）**：42/43/44 =
> best greedy **−21.0 / −20.3 / −20.6**，中位 −21.0，**0/3、曲线仍平直 → 负结果确认**。
> "旧数值流 / 3 seed 共享 reset 序列"两个嫌疑排除（per-seed 独立流已生效），下文
> 嫌疑人清单正式接管。下一步 = 「下一步」节第 2 条（S3 三臂兼诊断）；另 ROSMO 阶梯一
> 已进库（`src/rl/algo/my_zero/rosmo.rs`，`.rosmo(true)`），可作嫌疑 1（训练量不足）
> 的后续单变量臂（每份数据多榨优化步）。日志：`.bench/pong_s2_rerun_20260704.log`。
>
> **⚠️ 时效声明（2026-07-03 晚）**：首跑基于 `b697e95`，**先于** `adfc02f` 框架修复批
> （per-seed env.reset 种子派生——首跑 3 seed 共享同一 reset 序列、统计独立性存疑；
> GroupNorm 图内可微修复；温度调度显式常数化）。首跑数字降级为**历史参考**。

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

1. ~~**先 S2 复跑**~~ ✅ 2026-07-04 完成：仍 0/3 平直（见顶部声明），嫌疑"旧数值流 / 共享
   reset 序列"排除；
2. ~~复跑仍平 → S3 三臂 A/B **兼当诊断**~~ ✅ 2026-07-04 完成：**三臂全平**——
   recon pilot coef {1,4,16} = best greedy −21.0 / −20.4 / −21.0（嫌疑 3 排除）、
   cons-off 3-seed 中位 −21.0（嫌疑 2 排除）、HL-Gauss 3-seed 中位 −21.0（编码非瓶颈）。
   日志 `.bench/pong_s3_*.log`，账本已回填；
3. **当前状态：嫌疑 1（训练量不足）/ 4（稀疏 reward + γ 支撑）上位**。量化佐证（嫌疑 1）：
   本口径 150 局 × 64 trains × batch16 ≈ 9.6k updates，EfficientZero Atari-100k 参考
   口径 ~120k updates × batch256——优化量差 2 个数量级以上；且 buffer 内 replay ratio
   实测 ≈ 1.1 次/position（保守端）。对症臂（预注册待跑，后台机器时间即可）：
   ① trains_per_episode 加倍 × ROSMO 现算刷新顶 staleness（`.rosmo(true)` 已进库，
   CartPole 回归闸门 3/3 绿）；② lr {0.001, 0.003, 0.01} 扫描；③ DIAG 诊断
   value/reward 预测是否坍缩（嫌疑 4 判据）；
4. **战略调整（2026-07-04 规划修订）**：图像支柱裁决降级为**后台慢诊断**（吃机器时间不吃
   注意力），Phase 2 Gomoku 提前为主线（详见收口规划 §2/§3 修订注记）。本 issue 保持 open，
   终局裁决等预算标定臂证据齐后再下。

## 关联

- [Phase 1 计划](../../.doc/design/rl_phase1_image_plan.md) §S2/S3 · [收口规划 §2](../../.doc/design/rl_closure_plan.md)
- [Ep53 panic 调查](./pong_ep53_panic_investigation.md)（本轮 450 局零复现，佐证「转被动监视」裁决）
