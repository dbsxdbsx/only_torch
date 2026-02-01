//! # CartPole SAC-Discrete 强化学习示例
//!
//! 展示 `only_torch` 在强化学习场景的应用：
//! - SAC（Soft Actor-Critic）算法的离散动作版本
//! - 使用 `GymEnv` 与 Gymnasium 环境交互
//! - 完整的 off-policy 训练循环
//!
//! ## SAC-Discrete 特点
//! - Actor 输出离散动作的概率分布
//! - Critic 输出每个动作的 Q 值
//! - 自动调节的温度参数 alpha
//! - Twin Q 网络减少过估计
//!
//! ## 运行
//! ```bash
//! cargo run --example cartpole_sac
//! ```

mod model;

use model::SacAgent;
use only_torch::nn::{
    Adam, Graph, GraphError, Module, Optimizer, VarActivationOps, VarLossOps, VarReduceOps,
    VarShapeOps,
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
    action: usize,
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
            .cloned()
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
    learning_rate: f32,
    gamma: f32,
    start_training_after: usize, // 开始训练前需要收集的经验数
    update_every: usize,         // 每 N 步更新一次
    max_episodes: usize,
    target_reward: f32,
}

impl Default for SacConfig {
    fn default() -> Self {
        Self {
            buffer_size: 100_000,
            batch_size: 64,
            learning_rate: 0.001,
            gamma: 0.99,
            start_training_after: 1000,
            update_every: 1,
            max_episodes: 500,
            target_reward: 195.0,
        }
    }
}

// ============================================================================
// SAC 核心训练逻辑
// ============================================================================

/// 计算 V 值：V = Σ π(a|s) * (Q(s,a) - α * log π(a|s))
///
/// 这是 SAC 的核心公式，用于计算状态值函数（向量化实现）
fn compute_v_from_q(probs: &Tensor, q_values: &Tensor, log_probs: &Tensor, alpha: f32) -> Tensor {
    // V = Σ_a π(a|s) * (Q(s,a) - α * log π(a|s))
    (probs * &(q_values - &(log_probs * alpha))).sum_axis_keepdims(1)
}

/// 构建动作索引张量
fn build_action_indices(batch: &[Experience]) -> Tensor {
    let actions: Vec<f32> = batch.iter().map(|e| e.action as f32).collect();
    Tensor::new(&actions, &[batch.len(), 1])
}

// ============================================================================
// 主函数
// ============================================================================

fn main() -> Result<(), GraphError> {
    println!("=== CartPole SAC-Discrete 强化学习示例 ===\n");

    let config = SacConfig::default();

    Python::attach(|py| {
        // 1. 创建环境
        println!("[1/5] 创建 CartPole 环境...");
        let env = GymEnv::new(py, "CartPole-v1");
        env.print_env_basic_info();

        let obs_dim = env.get_flatten_observation_len();
        let action_dim = 2; // CartPole: 左/右

        // 2. 创建 SAC Agent
        println!("\n[2/5] 创建 SAC Agent...");
        let graph = Graph::new_with_seed(42);
        let mut agent = SacAgent::new(&graph, obs_dim, action_dim)?;

        println!("  Actor:  {} -> 64 -> {} (Softmax)", obs_dim, action_dim);
        println!("  Critic: {} -> 64 -> {} (Q values)", obs_dim, action_dim);
        println!("  Target Entropy: {:.3}", agent.target_entropy);

        // 3. 优化器
        let mut actor_optimizer = Adam::new(&graph, &agent.actor.parameters(), config.learning_rate);
        let mut critic1_optimizer = Adam::new(&graph, &agent.critic1.parameters(), config.learning_rate);
        let mut critic2_optimizer = Adam::new(&graph, &agent.critic2.parameters(), config.learning_rate);

        // 4. 经验回放缓冲区
        let mut buffer = ReplayBuffer::new(config.buffer_size);

        // 5. 训练循环
        println!("\n[3/5] 开始训练...");
        println!("  目标: 连续 100 回合平均奖励 >= {}\n", config.target_reward);

        let mut rng = rand::thread_rng();
        let mut episode_rewards: VecDeque<f32> = VecDeque::with_capacity(100);
        let mut total_steps = 0usize;

        // 调试计时
        use std::time::Instant;

        for episode in 0..config.max_episodes {
            let episode_start = Instant::now();
            let mut obs = env.reset(None)[0].clone();
            let mut episode_reward = 0.0;
            let mut episode_length = 0;

            loop {
                // 选择动作
                let obs_tensor = Tensor::new(&obs, &[1, obs_dim]);
                let (probs, _log_probs) = agent.actor.get_action_probs(&obs_tensor)?;
                let action = agent.actor.sample_action(&probs, &mut rng);

                // 执行动作
                let (next_obs_vec, reward, done) = env.step(&[action as f32]);
                let next_obs = next_obs_vec[0].clone();

                episode_reward += reward;
                episode_length += 1;
                total_steps += 1;

                // 存储经验
                buffer.push(Experience {
                    obs: obs.clone(),
                    action,
                    reward,
                    next_obs: next_obs.clone(),
                    done,
                });

                // SAC 更新
                if buffer.len() >= config.start_training_after 
                   && total_steps % config.update_every == 0 
                {
                    // 采样 batch
                    let batch = buffer.sample(config.batch_size, &mut rng);
                    
                    // 构建 batch tensors
                    let obs_data: Vec<f32> = batch.iter()
                        .flat_map(|e| e.obs.iter().cloned())
                        .collect();
                    let obs_batch = Tensor::new(&obs_data, &[batch.len(), obs_dim]);

                    // ========== Critic 更新 ==========
                    // 计算 target Q
                    let next_obs_data: Vec<f32> = batch.iter()
                        .flat_map(|e| e.next_obs.iter().cloned())
                        .collect();
                    let next_obs_batch = Tensor::new(&next_obs_data, &[batch.len(), obs_dim]);
                    
                    // 计算 target Q（这里使用 Tensor 操作，自然不参与梯度计算）
                    // 这等价于 PyTorch 的 with torch.no_grad():
                    let (next_probs, next_log_probs) = agent.actor.get_action_probs(&next_obs_batch)?;
                    let target_q1 = agent.target_critic1.get_q_values(&next_obs_batch)?;
                    let target_q2 = agent.target_critic2.get_q_values(&next_obs_batch)?;

                    // min(Q1', Q2') 并计算 V
                    let target_q_min = target_q1.minimum(&target_q2);
                    let v_next = compute_v_from_q(&next_probs, &target_q_min, &next_log_probs, agent.alpha());
                    
                    // target = r + γ * (1-done) * V(s')（向量化计算）
                    let rewards: Vec<f32> = batch.iter().map(|e| e.reward).collect();
                    let rewards_tensor = Tensor::new(&rewards, &[batch.len(), 1]);
                    let done_masks: Vec<f32> = batch.iter().map(|e| if e.done { 0.0 } else { 1.0 }).collect();
                    let done_masks_tensor = Tensor::new(&done_masks, &[batch.len(), 1]);
                    let target_tensor = &rewards_tensor + &(&done_masks_tensor * &(&v_next * config.gamma));
                    let action_indices = build_action_indices(&batch);

                    // ========== Critic1 更新 ==========
                    let q1_var = agent.critic1.forward(&obs_batch)?;
                    let q1_selected = q1_var.gather(1, &action_indices)?;  // 直接传 &Tensor
                    let critic1_loss = q1_selected.mse_loss(&target_tensor)?;

                    critic1_optimizer.zero_grad()?;
                    critic1_loss.backward()?;
                    critic1_optimizer.step()?;

                    // ========== Critic2 更新 ==========
                    let q2_var = agent.critic2.forward(&obs_batch)?;
                    let q2_selected = q2_var.gather(1, &action_indices)?;  // 直接传 &Tensor
                    let critic2_loss = q2_selected.mse_loss(&target_tensor)?;

                    critic2_optimizer.zero_grad()?;
                    critic2_loss.backward()?;
                    critic2_optimizer.step()?;

                    // ========== Actor 更新 ==========
                    // SAC-Discrete Actor Loss: 最小化 KL(π || exp(Q/α)/Z)
                    // 展开后等价于: L = Σ_a π(a|s) * (α * log π(a|s) - Q(s,a))
                    // 注意：KL 散度不对称，期望必须用当前策略 π 计算！
                    let q1 = agent.critic1.get_q_values(&obs_batch)?;
                    let q2 = agent.critic2.get_q_values(&obs_batch)?;
                    let q_min = q1.minimum(&q2);

                    // Actor 前向传播（整个计算保持在计算图中）
                    let actor_logits = agent.actor.forward(&obs_batch)?;

                    // log_softmax 比 softmax + ln 数值更稳定
                    let log_probs = actor_logits.log_softmax(); // Var [batch, action_dim]
                    let probs = actor_logits.softmax(); // Var [batch, action_dim]

                    // inside = α * log π(a|s) - Q(s,a)
                    // Var * Tensor（广播标量）= Var，Var - Tensor = Var
                    let alpha = agent.alpha();
                    let alpha_tensor = Tensor::ones(&[1, 1]) * alpha; // 广播标量
                    let inside = &log_probs * &alpha_tensor - &q_min; // Var [batch, action_dim]

                    // Actor Loss = Σ_a π(a|s) * inside，然后对 batch 求均值
                    let weighted = &probs * &inside; // Var * Var = Var
                    let action_sum = weighted.sum_axis(1); // Var [batch, 1]
                    let actor_loss = action_sum.mean(); // Var [1, 1]

                    // 先执行前向传播，获取 alpha 更新所需的值（backward 后会释放中间值）
                    actor_loss.forward()?;
                    let probs_val = probs.value()?.unwrap();
                    let log_probs_val = log_probs.value()?.unwrap();

                    // 然后执行反向传播和优化
                    actor_optimizer.zero_grad()?;
                    actor_loss.backward()?;
                    actor_optimizer.step()?;

                    // ========== Alpha 更新 ==========
                    agent.update_alpha(&log_probs_val, &probs_val);

                    // ========== 软更新目标网络 ==========
                    agent.soft_update_targets();

                    // 注意：这里不做 prune，因为 ModelState 的缓存需要保持稳定
                    // 临时节点的清理由 backward() 后的 release_intermediate_results() 处理
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

            let avg_reward: f32 = episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32;

            // 打印进度（每个 episode 都打印，便于调试）
            let episode_time = episode_start.elapsed().as_secs_f32();
            let node_count = graph.node_count();
            println!(
                "Ep {:3}: r={:5.1}, len={:3}, avg={:5.1}, α={:.3}, nodes={:5}, time={:.2}s",
                episode + 1,
                episode_reward,
                episode_length,
                avg_reward,
                agent.alpha(),
                node_count,
                episode_time,
            );

            // 检查是否达标
            if episode_rewards.len() >= 100 && avg_reward >= config.target_reward {
                println!("\n✅ 达到目标！连续 100 回合平均奖励 = {:.1}", avg_reward);
                break;
            }
        }

        // 6. 测试
        println!("\n[4/5] 测试训练好的策略...");
        let mut test_rewards = Vec::new();

        for i in 0..10 {
            let mut obs = env.reset(None)[0].clone();
            let mut episode_reward = 0.0;

            loop {
                let obs_tensor = Tensor::new(&obs, &[1, obs_dim]);
                let (probs, _) = agent.actor.get_action_probs(&obs_tensor)?;
                
                // 测试时使用贪婪策略（argmax）
                let action = probs.argmax(1)[[0]] as usize;

                let (next_obs_vec, reward, done) = env.step(&[action as f32]);
                episode_reward += reward;

                if done {
                    test_rewards.push(episode_reward);
                    println!("  测试 {:2}: 奖励 = {:.1}", i + 1, episode_reward);
                    break;
                }

                obs = next_obs_vec[0].clone();
            }
        }

        let avg_test_reward: f32 = test_rewards.iter().sum::<f32>() / test_rewards.len() as f32;
        println!("\n测试平均奖励: {:.1}", avg_test_reward);

        // 7. 完成
        println!("\n[5/5] 完成！");
        if avg_test_reward >= config.target_reward {
            println!("✅ CartPole SAC-Discrete 示例成功！");
        } else {
            println!("⚠️ 测试奖励未达标（可能需要更多训练）");
        }

        Ok(())
    })
}
