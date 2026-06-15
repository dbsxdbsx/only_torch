//! # Platform-v0 Hybrid SAC 强化学习示例
//!
//! 混合动作空间（离散平台选择 + 连续跳跃参数）的 SAC 实现。
//! 使用 `flatten_obs` 处理 Tuple 观察空间 `(Box(9,), Discrete(200))` → 10 维。
//!
//! ## 运行
//! ```bash
//! pip install hybrid-platform
//! cargo run --example platform_sac
//! ```

mod model;

use model::SacAgent;
use only_torch::nn::{
    Adam, Graph, GraphError, Module, Optimizer, Var, VarLossOps, VarReduceOps, VarShapeOps,
};
use only_torch::rl::GymEnv;
use only_torch::tensor::Tensor;
use pyo3::Python;
use rand::Rng;
use std::collections::VecDeque;

// ============================================================================
// 经验回放
// ============================================================================

#[derive(Clone)]
struct Experience {
    obs: Vec<f32>,
    /// [discrete_idx, c0, c1, c2]
    action: Vec<f32>,
    reward: f32,
    next_obs: Vec<f32>,
    terminated: bool,
}

struct ReplayBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
}

impl ReplayBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity.min(10_000)),
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
        let len = self.buffer.len();
        (0..batch_size)
            .map(|_| self.buffer[rng.gen_range(0..len)].clone())
            .collect()
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }
}

// ============================================================================
// 配置
// ============================================================================

struct SacConfig {
    buffer_size: usize,
    batch_size: usize,
    actor_lr: f32,
    critic_lr: f32,
    gamma: f32,
    tau: f32,
    hidden_dim: usize,
    start_training_after: usize,
    max_episodes: usize,
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
            tau: 0.005,
            hidden_dim: 128,
            start_training_after: 500,
            max_episodes: 2000,
            target_reward: 0.0,
        }
    }
}

// ============================================================================
// 主函数
// ============================================================================

fn main() -> Result<(), GraphError> {
    println!("=== Platform-v0 Hybrid SAC ===\n");

    let config = SacConfig::default();

    Python::attach(|py| {
        let env = GymEnv::new(py, "Platform-v0");
        env.print_env_basic_info();

        let obs_dim = env.get_flatten_observation_len(); // 9 + 1 = 10
        println!("  obs_dim={obs_dim}, discrete=3, continuous=3\n");

        let graph = Graph::new_with_seed(42);
        let mut agent = SacAgent::new(&graph, obs_dim, config.hidden_dim)?;

        // 初始化目标网络
        agent.target_critic1.soft_update_from(&agent.critic1, 1.0);
        agent.target_critic2.soft_update_from(&agent.critic2, 1.0);

        let mut actor_optimizer = Adam::new(&graph, &agent.actor.parameters(), config.actor_lr);
        let mut critic1_optimizer =
            Adam::new(&graph, &agent.critic1.parameters(), config.critic_lr);
        let mut critic2_optimizer =
            Adam::new(&graph, &agent.critic2.parameters(), config.critic_lr);

        let mut buffer = ReplayBuffer::new(config.buffer_size);
        let mut rng = rand::thread_rng();
        let mut episode_rewards: VecDeque<f32> = VecDeque::with_capacity(50);
        let mut total_steps = 0usize;

        println!(
            "[训练] target_entropy_d={:.3}, target_entropy_c={:.3}\n",
            agent.target_entropy_d, agent.target_entropy_c
        );

        for episode in 0..config.max_episodes {
            let obs_vec = env.reset(None);
            let mut obs = env.flatten_obs(&obs_vec);
            let mut episode_reward = 0.0;

            loop {
                let obs_tensor = Tensor::new(&obs, &[1, obs_dim]);
                let (_discrete, action_vec) = agent.actor.select_action(&obs_tensor)?;

                let (next_obs_vec, reward, terminated, truncated) = env.step(&action_vec);
                let next_obs = env.flatten_obs(&next_obs_vec);

                episode_reward += reward;
                total_steps += 1;

                buffer.push(Experience {
                    obs: obs.clone(),
                    action: action_vec,
                    reward,
                    next_obs: next_obs.clone(),
                    terminated,
                });

                // SAC 更新
                if buffer.len() >= config.start_training_after {
                    let batch = buffer.sample(config.batch_size, &mut rng);
                    let bs = batch.len();

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

                    let not_terminated: Vec<f32> = batch
                        .iter()
                        .map(|e| if e.terminated { 0.0 } else { 1.0 })
                        .collect();
                    let not_terminated_tensor = Tensor::new(&not_terminated, &[bs, 1]);

                    let stored_discrete: Vec<f32> = batch.iter().map(|e| e.action[0]).collect();
                    let stored_discrete_tensor = Tensor::new(&stored_discrete, &[bs, 1]);

                    let stored_cont: Vec<f32> = batch
                        .iter()
                        .flat_map(|e| e.action[1..4].iter().copied())
                        .collect();
                    let stored_cont_tensor = Tensor::new(&stored_cont, &[bs, 3]);

                    // ===== Target Q（无梯度）=====
                    let (next_probs, next_log_probs) = agent.actor.get_discrete_probs(&next_obs_batch)?;
                    let (next_cont, next_cont_lp) = agent.actor.sample_cont(&next_obs_batch)?;

                    let tq1 = agent.target_critic1.get_q_values(&next_obs_batch, &next_cont)?;
                    let tq2 = agent.target_critic2.get_q_values(&next_obs_batch, &next_cont)?;
                    let tq_min = tq1.minimum(&tq2);

                    let alpha_d = agent.alpha_d();
                    let alpha_c = agent.alpha_c();

                    // cont_log_prob: [bs, CONT_DIM] → sum to [bs, 1]
                    let cont_lp_sum = next_cont_lp.sum_axis_keepdims(1);

                    // V(s') = Σ_d π(d) × [Q(d) - α_d·log π(d)] - α_c·log_prob_c
                    let q_minus_entropy_d = &tq_min - &(&next_log_probs * alpha_d);
                    let v_next_per_d = &next_probs * &q_minus_entropy_d; // [bs, 3]
                    let v_next_sum = v_next_per_d.sum_axis_keepdims(1); // [bs, 1]
                    let v_next = &v_next_sum - &(&cont_lp_sum * alpha_c);

                    let target =
                        &rewards_tensor + &(&not_terminated_tensor * &(&v_next * config.gamma));

                    // ===== Critic 更新 =====
                    let obs_var1 = graph.input_named(&obs_batch, "obs")?;
                    let cont_var1 = graph.input_named(&stored_cont_tensor, "cont")?;
                    let q1_all = agent.critic1.forward_q(&obs_var1, &cont_var1)?;
                    let q1_sel = q1_all.gather(1, &stored_discrete_tensor)?;
                    let critic1_loss = q1_sel.mse_loss(&target)?;

                    critic1_optimizer.zero_grad()?;
                    critic1_loss.backward()?;
                    critic1_optimizer.step()?;

                    let obs_var2 = graph.input_named(&obs_batch, "obs")?;
                    let cont_var2 = graph.input_named(&stored_cont_tensor, "cont")?;
                    let q2_all = agent.critic2.forward_q(&obs_var2, &cont_var2)?;
                    let q2_sel = q2_all.gather(1, &stored_discrete_tensor)?;
                    let critic2_loss = q2_sel.mse_loss(&target)?;

                    critic2_optimizer.zero_grad()?;
                    critic2_loss.backward()?;
                    critic2_optimizer.step()?;

                    // ===== Actor 更新 =====
                    let obs_var_a = graph.input_named(&obs_batch, "obs")?;
                    let (probs, log_probs_var, cont_action, cont_lp_var) =
                        agent.actor.forward_train(&obs_var_a)?;

                    let q1_a = agent.critic1.forward_q(&obs_var_a, &cont_action)?;
                    let q2_a = agent.critic2.forward_q(&obs_var_a, &cont_action)?;
                    let q_min = Var::minimum(&q1_a, &q2_a)?;

                    // cont_log_prob: [bs, CONT_DIM] → sum → [bs, 1]
                    let cont_lp_sum_var = cont_lp_var.sum_axis(1);

                    // Actor loss = E[α_d·H(π_d) + α_c·log_prob_c - Σ_d π(d)·Q(d)]
                    let weighted_q = (&probs * &q_min).sum_axis(1); // [bs,1]
                    let discrete_entropy_cost = (&probs * &log_probs_var).sum_axis(1); // [bs,1]
                    let actor_loss = (&(&discrete_entropy_cost * alpha_d)
                        + &(&cont_lp_sum_var * alpha_c)
                        - &weighted_q)
                        .mean();

                    actor_optimizer.zero_grad()?;
                    actor_loss.backward()?;
                    actor_optimizer.step()?;

                    // ===== 温度更新（手动梯度下降）=====
                    let entropy_d_tensor = -(&next_probs * &next_log_probs).sum_axis_keepdims(1);
                    let avg_entropy_d = entropy_d_tensor.mean()[[0, 0]];
                    agent.log_alpha_d -= agent.alpha_lr
                        * agent.alpha_d()
                        * (agent.target_entropy_d - avg_entropy_d);
                    agent.log_alpha_d = agent.log_alpha_d.clamp(-20.0, 2.0);

                    let avg_cont_lp = next_cont_lp.mean()[[0, 0]];
                    agent.log_alpha_c -= agent.alpha_lr
                        * agent.alpha_c()
                        * (agent.target_entropy_c - (-avg_cont_lp));
                    agent.log_alpha_c = agent.log_alpha_c.clamp(-20.0, 2.0);

                    // 目标网络软更新
                    agent.soft_update_targets(config.tau);
                }

                if terminated || truncated {
                    break;
                }
                obs = next_obs;
            }

            episode_rewards.push_back(episode_reward);
            if episode_rewards.len() > 50 {
                episode_rewards.pop_front();
            }

            if (episode + 1) % 50 == 0 {
                let avg: f32 =
                    episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32;
                println!(
                    "Ep {:>4} | steps {:>6} | R={:.2} | avg50={:.2} | αd={:.4} αc={:.4}",
                    episode + 1,
                    total_steps,
                    episode_reward,
                    avg,
                    agent.alpha_d(),
                    agent.alpha_c()
                );

                if avg >= config.target_reward {
                    println!("\n[达标] 近 50 回合均值 >= {}", config.target_reward);
                    break;
                }
            }
        }

        env.close();
        Ok(())
    })
}
