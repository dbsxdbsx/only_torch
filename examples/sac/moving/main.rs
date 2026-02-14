//! # Moving-v0 Hybrid SAC 强化学习示例（方式 B — 独立连续分支）
//!
//! 展示 `only_torch` 在混合动作空间（离散 + 连续）强化学习场景的应用：
//! - SAC（Soft Actor-Critic）的 Hybrid 版本（Delalleau et al. 2019, Case #5）
//! - Actor 采用独立连续分支：Accelerate 头 / Turn 头 / Brake 无连续头
//! - 推理时按需调用分支（match），训练时全部分支前向
//! - 双温度（α_d, α_c）分别调节离散和连续探索
//! - 统一 Actor Loss 公式，log_prob_c 为 [batch, K]
//!
//! ## 运行
//! ```bash
//! cargo run --example moving_sac
//! ```
//!
//! 关于 SAC 算法的完整说明，请参阅 [`examples/sac/README.md`](../README.md)。

mod model;

use model::SacAgent;
use only_torch::nn::{
    Adam, Graph, GraphError, IntoVar, Module, Optimizer, Var, VarActivationOps, VarLossOps,
    VarReduceOps, VarShapeOps,
};
use only_torch::rl::GymEnv;
use only_torch::tensor::Tensor;
use pyo3::Python;
use rand::Rng;
use std::collections::VecDeque;

// ============================================================================
// 经验回放缓冲区
// ============================================================================

/// 单步经验（Hybrid：离散 + 连续动作）
#[derive(Clone)]
struct Experience {
    obs: Vec<f32>,
    /// 展平动作向量：[discrete_float, acceleration, rotation]
    action: Vec<f32>,
    reward: f32,
    next_obs: Vec<f32>,
    done: bool,
}

/// 简单的经验回放缓冲区
struct ReplayBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
}

impl ReplayBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, exp: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(exp);
    }

    fn sample(&self, batch_size: usize, rng: &mut impl Rng) -> Vec<Experience> {
        use rand::seq::SliceRandom;
        let indices: Vec<usize> = (0..self.buffer.len()).collect();
        let sampled: Vec<usize> = indices
            .choose_multiple(rng, batch_size.min(self.buffer.len()))
            .copied()
            .collect();
        sampled.iter().map(|&i| self.buffer[i].clone()).collect()
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }
}

// ============================================================================
// SAC 训练配置
// ============================================================================

struct SacConfig {
    buffer_size: usize,
    batch_size: usize,
    actor_lr: f32,
    critic_lr: f32,
    gamma: f32,
    hidden_dim: usize,
    start_training_after: usize,
    update_every: usize,
    max_episodes: usize,
    /// Moving-v0 最大 reward ≈ 1.0 + 距离改善，达到目标即 +1
    target_reward: f32,
}

impl Default for SacConfig {
    fn default() -> Self {
        Self {
            buffer_size: 100_000,
            batch_size: 128,
            actor_lr: 3e-4,
            critic_lr: 1e-3,
            gamma: 0.99,
            hidden_dim: 128, // Hybrid 任务复杂度高，128 兼顾表达力与 CPU 速度
            start_training_after: 500,
            update_every: 1, // 每步更新，与 CartPole/Pendulum 一致
            max_episodes: 2000,
            target_reward: -0.5, // 近 50 ep 平均 >= -0.5 即视为初步收敛
        }
    }
}

// ============================================================================
// 主函数
// ============================================================================

fn main() -> Result<(), GraphError> {
    println!("=== Moving-v0 Hybrid SAC（方式 B — 独立连续分支）===\n");

    let config = SacConfig::default();

    Python::attach(|py| {
        // 1. 创建环境
        println!("[1/6] 创建 Moving-v0 环境...");
        let env = GymEnv::new(py, "Moving-v0");
        env.print_env_basic_info();

        let obs_dim = env.get_flatten_observation_len(); // 10
        println!("  obs_dim={obs_dim}, discrete_actions=3, continuous_branches=2(Acc+Turn)");

        // 2. 创建 SAC Agent
        println!("\n[2/6] 创建 Hybrid SAC Agent...");
        let graph = Graph::new_with_seed(42);
        let mut agent = SacAgent::new(&graph, obs_dim, config.hidden_dim)?;

        // 目标网络初始化
        agent.target_critic1.hard_update_from(&agent.critic1);
        agent.target_critic2.hard_update_from(&agent.critic2);

        println!("  Actor: {obs_dim} → {} → {} → [离散(3), Acc(1), Turn(1)]",
            config.hidden_dim, config.hidden_dim);
        println!("  Critic: ({}+2) → {} → {} → Q(3)",
            obs_dim, config.hidden_dim, config.hidden_dim);
        println!("  Target Entropy: d={:.3}, c={:.3}",
            agent.target_entropy_d, agent.target_entropy_c);

        // 3. 优化器
        let mut actor_optimizer =
            Adam::new(&graph, &agent.actor.parameters(), config.actor_lr);
        let mut critic1_optimizer =
            Adam::new(&graph, &agent.critic1.parameters(), config.critic_lr);
        let mut critic2_optimizer =
            Adam::new(&graph, &agent.critic2.parameters(), config.critic_lr);

        // 4. 经验回放缓冲区
        let mut buffer = ReplayBuffer::new(config.buffer_size);

        // 5. 训练循环
        println!("\n[3/6] 开始训练...");
        println!("  目标: 单回合奖励 >= {}\n", config.target_reward);

        let mut rng = rand::thread_rng();
        let mut episode_rewards: VecDeque<f32> = VecDeque::with_capacity(100);
        let mut total_steps = 0usize;
        let mut solved = false;

        use std::time::Instant;

        for episode in 0..config.max_episodes {
            let episode_start = Instant::now();
            let mut obs = env.reset(None)[0].clone();
            let mut episode_reward = 0.0;
            let mut episode_length = 0;

            loop {
                // 选择动作（推理模式：按需调用分支）
                let obs_tensor = Tensor::new(&obs, &[1, obs_dim]);
                let (discrete_action, action_vec) =
                    agent.actor.select_action(&obs_tensor)?;
                let _ = discrete_action; // 仅用于调试

                // 执行动作
                let (next_obs_vec, reward, done) = env.step(&action_vec);
                let next_obs = next_obs_vec[0].clone();

                episode_reward += reward;
                episode_length += 1;
                total_steps += 1;

                // 存储经验
                buffer.push(Experience {
                    obs: obs.clone(),
                    action: action_vec,
                    reward,
                    next_obs: next_obs.clone(),
                    done,
                });

                // SAC 更新
                if buffer.len() >= config.start_training_after
                    && total_steps.is_multiple_of(config.update_every)
                {
                    let batch = buffer.sample(config.batch_size, &mut rng);
                    let bs = batch.len();

                    // 构建 batch tensors
                    let obs_data: Vec<f32> =
                        batch.iter().flat_map(|e| e.obs.iter().copied()).collect();
                    let obs_batch = Tensor::new(&obs_data, &[bs, obs_dim]);

                    let next_obs_data: Vec<f32> = batch
                        .iter()
                        .flat_map(|e| e.next_obs.iter().copied())
                        .collect();
                    let next_obs_batch = Tensor::new(&next_obs_data, &[bs, obs_dim]);

                    let rewards: Vec<f32> = batch.iter().map(|e| e.reward).collect();
                    let rewards_tensor = Tensor::new(&rewards, &[bs, 1]);

                    let done_masks: Vec<f32> = batch
                        .iter()
                        .map(|e| if e.done { 0.0 } else { 1.0 })
                        .collect();
                    let done_masks_tensor = Tensor::new(&done_masks, &[bs, 1]);

                    // 从 batch 提取存储的动作（用于 Critic Loss）
                    // action_vec = [discrete, acc, rot]
                    let stored_discrete: Vec<f32> =
                        batch.iter().map(|e| e.action[0]).collect();
                    let stored_discrete_tensor =
                        Tensor::new(&stored_discrete, &[bs, 1]);

                    let stored_cont: Vec<f32> = batch
                        .iter()
                        .flat_map(|e| [e.action[1], e.action[2]])
                        .collect();
                    let stored_cont_tensor = Tensor::new(&stored_cont, &[bs, 2]);

                    // ========== Target Q 计算（无梯度）==========
                    // 用当前策略在 next_obs 上全分支前向
                    let next_actor_out =
                        agent.actor.forward_all_branches(&next_obs_batch)?;
                    next_actor_out.probs.forward()?;
                    let next_probs = next_actor_out.probs.value()?.unwrap();
                    let next_log_probs = next_actor_out.log_probs.value()?.unwrap();

                    let next_acc_action = next_actor_out.acc_env_action.value()?.unwrap();
                    let next_turn_action =
                        next_actor_out.turn_env_action.value()?.unwrap();
                    let next_acc_lp = next_actor_out.acc_log_prob.value()?.unwrap();
                    let next_turn_lp = next_actor_out.turn_log_prob.value()?.unwrap();

                    // K 次 target Critic 前向
                    let zeros_bs1 = Tensor::zeros(&[bs, 1]);

                    // Accelerate: cont = [acc, 0]
                    let cont_for_acc = Tensor::concat(
                        &[&next_acc_action, &zeros_bs1],
                        1,
                    );
                    let tq1_acc =
                        agent.target_critic1.get_q_values(&next_obs_batch, &cont_for_acc)?;
                    let tq2_acc =
                        agent.target_critic2.get_q_values(&next_obs_batch, &cont_for_acc)?;
                    let tq_min_acc = tq1_acc.minimum(&tq2_acc);

                    // Turn: cont = [0, rot]
                    let cont_for_turn = Tensor::concat(
                        &[&zeros_bs1, &next_turn_action],
                        1,
                    );
                    let tq1_turn =
                        agent.target_critic1.get_q_values(&next_obs_batch, &cont_for_turn)?;
                    let tq2_turn =
                        agent.target_critic2.get_q_values(&next_obs_batch, &cont_for_turn)?;
                    let tq_min_turn = tq1_turn.minimum(&tq2_turn);

                    // Brake: cont = [0, 0]
                    let cont_for_brake = Tensor::zeros(&[bs, 2]);
                    let tq1_brake = agent
                        .target_critic1
                        .get_q_values(&next_obs_batch, &cont_for_brake)?;
                    let tq2_brake = agent
                        .target_critic2
                        .get_q_values(&next_obs_batch, &cont_for_brake)?;
                    let tq_min_brake = tq1_brake.minimum(&tq2_brake);

                    // 对角线取值：Q_acc[col=0], Q_turn[col=1], Q_brake[col=2]
                    let idx_0 = Tensor::new(&vec![0.0; bs], &[bs, 1]);
                    let idx_1 = Tensor::new(&vec![1.0; bs], &[bs, 1]);
                    let idx_2 = Tensor::new(&vec![2.0; bs], &[bs, 1]);
                    let tq_acc_col = tq_min_acc.gather(1, &idx_0);     // [bs, 1]
                    let tq_turn_col = tq_min_turn.gather(1, &idx_1);   // [bs, 1]
                    let tq_brake_col = tq_min_brake.gather(1, &idx_2); // [bs, 1]

                    // 构造 log_prob_c: [bs, 3]（Brake 列 = 0）
                    let zero_lp = Tensor::zeros(&[bs, 1]);
                    let next_log_prob_c = Tensor::concat(
                        &[&next_acc_lp, &next_turn_lp, &zero_lp],
                        1,
                    );
                    // 构造 min_Q: [bs, 3]
                    let next_min_q = Tensor::concat(
                        &[&tq_acc_col, &tq_turn_col, &tq_brake_col],
                        1,
                    );

                    // V(s') = Σ_d π(d) × (Q(d) - α_d·log π(d) - α_c·log π_c(d))
                    let alpha_d = agent.alpha_d();
                    let alpha_c = agent.alpha_c();
                    let v_next = (&next_probs
                        * &(&next_min_q
                            - &(&next_log_probs * alpha_d)
                            - &(&next_log_prob_c * alpha_c)))
                        .sum_axis_keepdims(1);

                    // target = r + γ * (1-done) * V(s')
                    let target_tensor =
                        &rewards_tensor + &(&done_masks_tensor * &(&v_next * config.gamma));

                    // ========== Critic 更新 ==========
                    // Critic 用存储的 (state, action) 对，不需要全分支
                    let obs_var1 = graph.input_named(&obs_batch, "obs")?;
                    let act_var1 = graph.input_named(&stored_cont_tensor, "action")?;
                    let q1_all = agent.critic1.forward_q(&obs_var1, &act_var1)?;
                    let q1_selected = q1_all.gather(1, &stored_discrete_tensor)?;
                    let critic1_loss = q1_selected.mse_loss(&target_tensor)?;

                    critic1_optimizer.zero_grad()?;
                    critic1_loss.backward()?;
                    critic1_optimizer.step()?;

                    let obs_var2 = graph.input_named(&obs_batch, "obs")?;
                    let act_var2 = graph.input_named(&stored_cont_tensor, "action")?;
                    let q2_all = agent.critic2.forward_q(&obs_var2, &act_var2)?;
                    let q2_selected = q2_all.gather(1, &stored_discrete_tensor)?;
                    let critic2_loss = q2_selected.mse_loss(&target_tensor)?;

                    critic2_optimizer.zero_grad()?;
                    critic2_loss.backward()?;
                    critic2_optimizer.step()?;

                    // ========== Actor 更新 ==========
                    let obs_var_actor = graph.input_named(&obs_batch, "obs")?;
                    let actor_out =
                        agent.actor.forward_all_branches(&obs_var_actor)?;

                    // K 次 Critic 前向（用当前 Actor 的全新采样）
                    let zeros_tensor = Tensor::new(&vec![0.0; bs], &[bs, 1]);
                    let zeros_var = zeros_tensor.into_var(&graph)?;

                    // Accelerate: [acc_action, 0]
                    let cont_acc_var =
                        Var::concat(&[&actor_out.acc_env_action, &zeros_var], 1)?;
                    let q1_acc = agent.critic1.forward_q(&obs_var_actor, &cont_acc_var)?;
                    let q2_acc = agent.critic2.forward_q(&obs_var_actor, &cont_acc_var)?;
                    let q_diff_acc = &q1_acc - &q2_acc;
                    let half = Tensor::new(&[0.5], &[1, 1]);
                    let q_min_acc =
                        (&q1_acc + &q2_acc - q_diff_acc.abs()) * &half;

                    // Turn: [0, turn_action]
                    let cont_turn_var =
                        Var::concat(&[&zeros_var, &actor_out.turn_env_action], 1)?;
                    let q1_turn =
                        agent.critic1.forward_q(&obs_var_actor, &cont_turn_var)?;
                    let q2_turn =
                        agent.critic2.forward_q(&obs_var_actor, &cont_turn_var)?;
                    let q_diff_turn = &q1_turn - &q2_turn;
                    let q_min_turn =
                        (&q1_turn + &q2_turn - q_diff_turn.abs()) * &half;

                    // Brake: [0, 0]
                    let cont_brake_var =
                        Var::concat(&[&zeros_var, &zeros_var], 1)?;
                    let q1_brake =
                        agent.critic1.forward_q(&obs_var_actor, &cont_brake_var)?;
                    let q2_brake =
                        agent.critic2.forward_q(&obs_var_actor, &cont_brake_var)?;
                    let q_diff_brake = &q1_brake - &q2_brake;
                    let q_min_brake =
                        (&q1_brake + &q2_brake - q_diff_brake.abs()) * &half;

                    // 对角线取值（Var 级）：Q_acc[col=0], Q_turn[col=1], Q_brake[col=2]
                    let q_acc_col0 = q_min_acc.gather(1, &idx_0)?;
                    let q_turn_col1 = q_min_turn.gather(1, &idx_1)?;
                    let q_brake_col2 = q_min_brake.gather(1, &idx_2)?;
                    // min_Q: [bs, 3]
                    let min_q_var =
                        Var::concat(&[&q_acc_col0, &q_turn_col1, &q_brake_col2], 1)?;

                    // log_prob_c: [bs, 3]（Brake 列 = 0 哑值）
                    let zero_lp_var = Tensor::new(&vec![0.0; bs], &[bs, 1])
                        .into_var(&graph)?;
                    let log_prob_c_var = Var::concat(
                        &[
                            &actor_out.acc_log_prob,
                            &actor_out.turn_log_prob,
                            &zero_lp_var,
                        ],
                        1,
                    )?;

                    // 统一 Actor Loss:
                    // L = -mean( Σ_d π(d) × (min_Q(d) - α_d·log π(d) - α_c·log π_c(d)) )
                    let alpha_d_t = Tensor::new(&[alpha_d], &[1, 1]);
                    let alpha_c_t = Tensor::new(&[alpha_c], &[1, 1]);

                    let actor_loss = (&actor_out.probs
                        * (&actor_out.log_probs * &alpha_d_t
                            + &log_prob_c_var * &alpha_c_t
                            - &min_q_var))
                        .sum_axis(1)
                        .mean();

                    // Forward 计算值，然后离线算 entropy 用于 alpha 更新
                    actor_loss.forward()?;

                    let probs_val = actor_out.probs.value()?.unwrap();
                    let log_probs_val = actor_out.log_probs.value()?.unwrap();
                    let log_prob_c_val = log_prob_c_var.value()?.unwrap();
                    let batch_f = bs as f32;

                    // 离散 entropy: -Σ π·log π
                    let avg_discrete_entropy = -(&probs_val * &log_probs_val)
                        .sum()
                        .get_data_number()
                        .unwrap()
                        / batch_f;

                    // 连续 entropy: -Σ_d π(d)·log_prob_c(d)（加权平均）
                    let avg_continuous_entropy = -(&probs_val * &log_prob_c_val)
                        .sum()
                        .get_data_number()
                        .unwrap()
                        / batch_f;

                    actor_optimizer.zero_grad()?;
                    actor_loss.backward()?;
                    actor_optimizer.step()?;

                    // ========== Alpha 更新 ==========
                    agent.update_alpha_d(avg_discrete_entropy);
                    agent.update_alpha_c(avg_continuous_entropy);

                    // ========== 软更新目标网络 ==========
                    agent.soft_update_targets();

                    // 可视化快照（首次训练步骤时拍摄）
                    graph.snapshot_once(&[
                        ("Actor Loss", &actor_loss),
                        ("Critic1 Loss", &critic1_loss),
                        ("Critic2 Loss", &critic2_loss),
                    ]);
                }

                if done {
                    break;
                }
                obs = next_obs;
            }

            // 记录回合奖励
            episode_rewards.push_back(episode_reward);
            if episode_rewards.len() > 100 {
                episode_rewards.pop_front();
            }

            let avg_reward: f32 =
                episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32;

            let episode_time = episode_start.elapsed().as_secs_f32();
            let param_count = graph.parameter_count();
            println!(
                "Ep {:4}: R={:7.3}, len={:3}, avg100={:7.3}, α_d={:.4}, α_c={:.4}, params={:6}, time={:.2}s",
                episode + 1,
                episode_reward,
                episode_length,
                avg_reward,
                agent.alpha_d(),
                agent.alpha_c(),
                param_count,
                episode_time,
            );

            // 检查是否达标（近 50 episode 平均）
            if episode_rewards.len() >= 50 {
                let recent: f32 = episode_rewards
                    .iter()
                    .rev()
                    .take(50)
                    .sum::<f32>()
                    / 50.0;
                if recent >= config.target_reward {
                    println!(
                        "\n  达到目标！近 50 ep 平均 R={:.3}（目标 >= {:.1}）",
                        recent,
                        config.target_reward
                    );
                    solved = true;
                    break;
                }
            }
        }

        // 6. 测试
        println!("\n[4/6] 测试训练好的策略...");
        let mut test_rewards = Vec::new();

        for i in 0..5 {
            let mut obs = env.reset(None)[0].clone();
            let mut episode_reward = 0.0;

            loop {
                let obs_tensor = Tensor::new(&obs, &[1, obs_dim]);
                let (_action, action_vec) =
                    agent.actor.select_action(&obs_tensor)?;
                let (next_obs_vec, reward, done) = env.step(&action_vec);
                episode_reward += reward;

                if done {
                    test_rewards.push(episode_reward);
                    println!("  测试 {}: R = {:.3}", i + 1, episode_reward);
                    break;
                }
                obs = next_obs_vec[0].clone();
            }
        }

        let avg_test_reward: f32 =
            test_rewards.iter().sum::<f32>() / test_rewards.len() as f32;
        println!(
            "\n测试平均奖励: {:.3}（目标 >= {:.1}）",
            avg_test_reward, config.target_reward
        );

        // 7. 保存计算图可视化
        println!("\n[5/6] 保存计算图可视化...");
        let vis_result =
            graph.visualize_snapshot("examples/sac/moving/moving_sac")?;
        println!("  计算图已保存: {}", vis_result.dot_path.display());
        if let Some(img_path) = &vis_result.image_path {
            println!("  可视化图像: {}", img_path.display());
        }
        if let Some(hint) = &vis_result.graphviz_hint {
            println!("  Graphviz 提示: {hint}");
        }

        // 8. 完成
        println!("\n[6/6] 完成！");
        if solved || avg_test_reward >= config.target_reward {
            println!("  Moving-v0 Hybrid SAC（方式 B）示例成功！（测试平均 R={avg_test_reward:.3}）");
        } else {
            println!(
                "  测试平均奖励未达标（{:.3} < {:.1}，可能需要更多训练）",
                avg_test_reward, config.target_reward
            );
        }

        Ok(())
    })
}
