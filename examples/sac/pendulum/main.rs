//! # Pendulum SAC-Continuous 强化学习示例
//!
//! 展示 `only_torch` 在连续动作强化学习场景的应用：
//! - SAC（Soft Actor-Critic）算法的连续动作版本
//! - TanhNormal 分布 + 重参数化采样
//! - Critic 使用 `Var::concat` 拼接 obs + action
//! - 动作缩放：TanhNormal 输出 [-1,1] → 环境范围 [-2,2]
//!
//! ## 运行
//! ```bash
//! cargo run --example pendulum_sac
//! ```
//!
//! 关于 SAC 算法的完整说明，请参阅 [`examples/sac/README.md`](../README.md)。

mod model;

use model::SacAgent;
use only_torch::nn::{
    Adam, Graph, GraphError, Module, Optimizer, VarActivationOps, VarLossOps, VarReduceOps,
};
use only_torch::rl::GymEnv;
use only_torch::tensor::Tensor;
use pyo3::Python;
use rand::Rng;
use std::collections::VecDeque;

// ============================================================================
// 经验回放缓冲区
// ============================================================================

/// 单步经验
#[derive(Clone)]
struct Experience {
    obs: Vec<f32>,
    action: Vec<f32>, // 连续动作（Pendulum: 1 维）
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
    /// Pendulum-v1 奖励范围约 [-16, 0]，平均 reward >= -200 视为收敛
    target_reward: f32,
}

impl Default for SacConfig {
    fn default() -> Self {
        Self {
            buffer_size: 100_000,
            batch_size: 256,
            actor_lr: 3e-4,   // Actor 学习率（CleanRL 默认）
            critic_lr: 1e-3,   // Critic 学习率（高于 Actor，让 Q 网络更快适应）
            gamma: 0.99,
            hidden_dim: 32,    // Pendulum 是简单任务，32 足够展示
            start_training_after: 500,
            update_every: 1,
            max_episodes: 300,
            target_reward: -300.0, // 展示用：首次达到即停止
        }
    }
}

// ============================================================================
// 主函数
// ============================================================================

fn main() -> Result<(), GraphError> {
    println!("=== Pendulum SAC-Continuous 强化学习示例 ===\n");

    let config = SacConfig::default();

    Python::attach(|py| {
        // 1. 创建环境
        println!("[1/6] 创建 Pendulum 环境...");
        let env = GymEnv::new(py, "Pendulum-v1");
        env.print_env_basic_info();

        let obs_dim = env.get_flatten_observation_len();
        let action_ranges = env.get_all_action_valid_range();
        let action_dim = action_ranges.len(); // Pendulum: 1
        let (action_low, action_high) = action_ranges[0].get_continuous_action_low_high();

        println!("  obs_dim={obs_dim}, action_dim={action_dim}");
        println!("  action range: [{action_low}, {action_high}]");

        // 2. 创建 SAC Agent
        println!("\n[2/6] 创建 SAC Agent...");
        let graph = Graph::new_with_seed(42);
        let mut agent = SacAgent::new(
            &graph,
            obs_dim,
            action_dim,
            config.hidden_dim,
            action_low,
            action_high,
        )?;

        // 目标网络初始化（硬拷贝）
        agent.target_critic1.hard_update_from(&agent.critic1);
        agent.target_critic2.hard_update_from(&agent.critic2);

        println!("  Actor:  {obs_dim} → {} → {} → (mean, log_std)", config.hidden_dim, config.hidden_dim);
        println!("  Critic: ({obs_dim}+{action_dim}) → {} → {} → 1", config.hidden_dim, config.hidden_dim);
        println!("  Target Entropy: {:.3}", agent.target_entropy);
        println!(
            "  Action Scale: {:.1}, Bias: {:.1}",
            agent.action_scale, agent.action_bias
        );

        // 3. 优化器（Critic 学习率高于 Actor，参考 CleanRL）
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
        println!(
            "  目标: 近 100 回合平均奖励 >= {}\n",
            config.target_reward
        );

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
                // 选择动作：TanhNormal 分布采样
                let obs_tensor = Tensor::new(&obs, &[1, obs_dim]);
                let (tanh_action, _log_prob) = agent.actor.sample_action(&obs_tensor)?;

                // 缩放到环境范围
                let env_action = agent.scale_action(&tanh_action);
                let env_action_vec: Vec<f32> = (0..action_dim)
                    .map(|i| env_action[[0, i]])
                    .collect();

                // 执行动作
                let (next_obs_vec, reward, done) = env.step(&env_action_vec);
                let next_obs = next_obs_vec[0].clone();

                episode_reward += reward;
                episode_length += 1;
                total_steps += 1;

                // 存储经验（存 env 范围的 action）
                buffer.push(Experience {
                    obs: obs.clone(),
                    action: env_action_vec,
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

                    let action_data: Vec<f32> =
                        batch.iter().flat_map(|e| e.action.iter().copied()).collect();
                    let action_batch = Tensor::new(&action_data, &[bs, action_dim]);
                    // 归一化 action 到 [-1,1]（Critic 输入标准化）
                    let action_batch_norm = agent.unscale_action(&action_batch);

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

                    // ========== Target Q 计算（无梯度）==========
                    // 从当前策略采样 next_action 和 log_prob
                    let (next_tanh_action, next_log_prob) =
                        agent.actor.sample_action(&next_obs_batch)?;
                    // log_prob: [batch, action_dim] → sum → [batch, 1]
                    let next_log_prob_sum = next_log_prob.sum_axis_keepdims(1);

                    // target Q = min(Q1', Q2') - α * log_prob
                    let target_q1 = agent
                        .target_critic1
                        .get_q_value(&next_obs_batch, &next_tanh_action)?;
                    let target_q2 = agent
                        .target_critic2
                        .get_q_value(&next_obs_batch, &next_tanh_action)?;
                    let target_q_min = target_q1.minimum(&target_q2);
                    let target_v = &target_q_min - &(&next_log_prob_sum * agent.alpha());

                    // y = r + γ * (1-done) * V(s')
                    let target_tensor =
                        &rewards_tensor + &(&done_masks_tensor * &(&target_v * config.gamma));

                    // ========== Critic1 更新 ==========
                    let obs_var1 = graph.input_named(&obs_batch, "obs")?;
                    let act_var1 = graph.input_named(&action_batch_norm, "action")?;
                    let q1_pred = agent.critic1.forward_q(&obs_var1, &act_var1)?;
                    let critic1_loss = q1_pred.mse_loss(&target_tensor)?;

                    critic1_optimizer.zero_grad()?;
                    critic1_loss.backward()?;
                    critic1_optimizer.step()?;

                    // ========== Critic2 更新 ==========
                    let obs_var2 = graph.input_named(&obs_batch, "obs")?;
                    let act_var2 = graph.input_named(&action_batch_norm, "action")?;
                    let q2_pred = agent.critic2.forward_q(&obs_var2, &act_var2)?;
                    let critic2_loss = q2_pred.mse_loss(&target_tensor)?;

                    critic2_optimizer.zero_grad()?;
                    critic2_loss.backward()?;
                    critic2_optimizer.step()?;

                    // ========== Actor 更新 ==========
                    // 从当前策略采样 action（保留计算图用于梯度流回 Actor）
                    let obs_var_actor = graph.input_named(&obs_batch, "obs")?;
                    let (action_var, log_prob_var) =
                        agent.actor.sample_for_update(&obs_var_actor)?;

                    // log_prob: [batch, action_dim] → sum → [batch, 1]
                    let log_prob_sum = log_prob_var.sum_axis(1); // [batch, 1]

                    // Q(s, a) — 梯度从 Q 经由 Concat → action → Actor params 流回
                    let q1_actor = agent.critic1.forward_q(&obs_var_actor, &action_var)?;
                    let q2_actor = agent.critic2.forward_q(&obs_var_actor, &action_var)?;
                    // min(Q1, Q2) = 0.5 * (Q1 + Q2 - |Q1 - Q2|)
                    // 使用数学恒等式实现 Var 级 element-wise min（保留梯度流）
                    let q_diff = &q1_actor - &q2_actor;
                    let half = Tensor::new(&[0.5], &[1, 1]);
                    let q_min_var = (&q1_actor + &q2_actor - q_diff.abs()) * half;

                    // Actor Loss: (α * log_prob - Q).mean()
                    let alpha_tensor = Tensor::new(&[agent.alpha()], &[1, 1]);
                    let actor_loss = (&log_prob_sum * alpha_tensor - &q_min_var).mean();

                    // 先 forward 计算值，再离线计算 entropy 用于 alpha 更新
                    actor_loss.forward()?;
                    let lp_val = log_prob_sum.value()?.unwrap();
                    let batch_f = bs as f32;
                    let avg_entropy =
                        -(lp_val.sum().get_data_number().unwrap()) / batch_f;

                    actor_optimizer.zero_grad()?;
                    actor_loss.backward()?;
                    actor_optimizer.step()?;

                    // ========== Alpha 更新 ==========
                    agent.update_alpha(avg_entropy);

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
                "Ep {:3}: R={:7.1}, len={:3}, avg100={:7.1}, α={:.4}, params={:6}, time={:.2}s",
                episode + 1,
                episode_reward,
                episode_length,
                avg_reward,
                agent.alpha(),
                param_count,
                episode_time,
            );

            // 检查是否达标（展示用：首次单回合达到目标即停止）
            if episode_reward >= config.target_reward {
                println!(
                    "\n✅ 达到目标！Ep {} R={:.1}（目标 >= {:.0}）",
                    episode + 1,
                    episode_reward,
                    config.target_reward
                );
                solved = true;
                break;
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
                let (tanh_action, _) = agent.actor.sample_action(&obs_tensor)?;
                let env_action = agent.scale_action(&tanh_action);
                let env_action_vec: Vec<f32> = (0..action_dim)
                    .map(|i| env_action[[0, i]])
                    .collect();

                let (next_obs_vec, reward, done) = env.step(&env_action_vec);
                episode_reward += reward;

                if done {
                    test_rewards.push(episode_reward);
                    println!("  测试 {}: R = {:.1}", i + 1, episode_reward);
                    break;
                }
                obs = next_obs_vec[0].clone();
            }
        }

        let avg_test_reward: f32 = test_rewards.iter().sum::<f32>() / test_rewards.len() as f32;
        println!(
            "\n测试平均奖励: {:.1}（目标 >= {:.0}）",
            avg_test_reward, config.target_reward
        );

        // 7. 保存计算图可视化
        println!("\n[5/6] 保存计算图可视化...");
        let vis_result = graph.visualize_snapshot("examples/sac/pendulum/pendulum_sac")?;
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
            println!("✅ Pendulum SAC-Continuous 示例成功！（测试平均 R={avg_test_reward:.1}）");
        } else {
            println!(
                "⚠️ 测试平均奖励未达标（{:.1} < {:.0}，可能需要更多训练）",
                avg_test_reward, config.target_reward
            );
        }

        Ok(())
    })
}
