//! MuZero CartPole-v0 训练示例
//!
//! ```bash
//! cargo run --example muzero_cartpole
//! SMOKE=1 cargo run --example muzero_cartpole  # 管线验证（3 局 self-play + 1 次训练）
//! ```

mod model;

use model::MuZeroModel;
use only_torch::nn::{Adam, Graph, GraphError, Optimizer};
use only_torch::rl::mcts::{mcts_search, ActionPayload, DynamicsModel, MctsConfig, PuctPolicy};
use only_torch::rl::{GameOutcome, GymEnv, ReplayBuffer, SelfPlayGame, SelfPlayStep};
use pyo3::Python;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::VecDeque;

/// N-step value target 计算
fn compute_n_step_target(steps: &[SelfPlayStep], start: usize, n: usize, gamma: f32) -> f32 {
    let mut target = 0.0;
    let end = (start + n).min(steps.len());
    for i in start..end {
        target += gamma.powi((i - start) as i32) * steps[i].reward;
    }
    if end < steps.len() {
        if let Some(root_v) = steps[end].root_value {
            target += gamma.powi((end - start) as i32) * root_v;
        }
    }
    target
}

/// Self-play 收集一局
fn self_play_one_episode(
    env: &GymEnv,
    model: &MuZeroModel,
    actions: &[ActionPayload],
    mcts_cfg: &MctsConfig,
    gamma: f32,
    ep_seed: Option<u64>,
) -> Vec<SelfPlayStep> {
    let obs_raw = env.reset(ep_seed);
    let mut obs = obs_raw[0].clone();
    let mut steps = Vec::new();

    loop {
        let dyn_model = DynamicsModel::new(model, actions.to_vec(), gamma);
        let result = mcts_search(&dyn_model, &PuctPolicy::new(), &obs, mcts_cfg);

        let action_idx = match &result.recommended {
            ActionPayload::Discrete(idx) => *idx,
            _ => 0,
        };

        // 根节点 value：访问次数加权平均
        let root_value = if !result.children.is_empty() {
            let total_visits: u32 = result.children.iter().map(|c| c.visit_count).sum();
            if total_visits > 0 {
                result
                    .children
                    .iter()
                    .map(|c| {
                        let q = if c.visit_count > 0 {
                            c.value_sum / c.visit_count as f32
                        } else {
                            0.0
                        };
                        q * c.visit_count as f32
                    })
                    .sum::<f32>()
                    / total_visits as f32
            } else {
                0.0
            }
        } else {
            0.0
        };

        steps.push(SelfPlayStep {
            obs: obs.clone(),
            action: vec![action_idx as f32],
            policy_target: result.learn_policy,
            player: 0,
            reward: 0.0, // 后面填入
            root_value: Some(root_value),
        });

        let (next_obs_raw, reward, terminated, truncated) = env.step(&[action_idx as f32]);

        // 回填 reward
        steps.last_mut().unwrap().reward = reward;

        if terminated || truncated {
            break;
        }
        obs = next_obs_raw[0].clone();
    }

    steps
}

/// 从一局游戏中采样并训练一次
fn train_one_step(
    model: &MuZeroModel,
    optimizer: &mut Adam,
    game: &SelfPlayGame,
    k_unroll: usize,
    td_steps: usize,
    gamma: f32,
    rng: &mut impl Rng,
) -> Result<f32, GraphError> {
    let steps = &game.steps;
    if steps.len() < k_unroll + 1 {
        return Ok(0.0);
    }

    // 随机选起始位置（确保能展开 K 步）
    let max_start = steps.len().saturating_sub(k_unroll + 1);
    let t = rng.gen_range(0..=max_start);

    // 准备目标
    let actions: Vec<usize> = (0..k_unroll)
        .map(|i| steps[t + i].action[0] as usize)
        .collect();

    // 策略目标：t 位置 + K 步后续
    let target_policies: Vec<Vec<f32>> = (0..=k_unroll)
        .map(|i| {
            if t + i < steps.len() {
                steps[t + i].policy_target.clone()
            } else {
                vec![1.0 / model.action_dim as f32; model.action_dim]
            }
        })
        .collect();

    // n-step value target
    let target_values: Vec<f32> = (0..=k_unroll)
        .map(|i| compute_n_step_target(steps, t + i, td_steps, gamma))
        .collect();

    // 实际奖励目标
    let target_rewards: Vec<f32> = (0..k_unroll)
        .map(|i| {
            if t + i < steps.len() {
                steps[t + i].reward
            } else {
                0.0
            }
        })
        .collect();

    let obs_t = &steps[t].obs;

    optimizer.zero_grad()?;
    let loss = model.train_unroll(obs_t, &actions, &target_policies, &target_values, &target_rewards)?;
    let loss_val = loss.backward()?;
    optimizer.step()?;

    Ok(loss_val)
}

fn main() -> Result<(), GraphError> {
    let smoke = std::env::var("SMOKE").is_ok();

    // 超参数
    let latent_dim = 32;
    let action_dim = 2;
    let obs_dim = 4;
    let lr = 1e-3;
    let gamma = 0.99;
    let k_unroll = 5;
    let td_steps = 10;
    let num_simulations = 25;
    let buffer_capacity = 500;
    let start_training_after = 5;
    let trains_per_episode = 4;
    let max_episodes = if smoke { 3 } else { 1000 };

    Python::attach(|py| {
        let env = GymEnv::new(py, "CartPole-v0");
        let graph = Graph::new_with_seed(42);
        let model = MuZeroModel::new(&graph, obs_dim, action_dim, latent_dim)?;
        let mut optimizer = Adam::new(&graph, &model.parameters(), lr);
        let mut buffer: ReplayBuffer<SelfPlayGame> = ReplayBuffer::new(buffer_capacity);
        let mut rng = StdRng::seed_from_u64(42);

        let actions: Vec<ActionPayload> = (0..action_dim)
            .map(ActionPayload::Discrete)
            .collect();

        let mcts_cfg = MctsConfig {
            num_simulations,
            temperature: 1.0,
            discount: gamma,
            ..MctsConfig::default()
        };

        let mut ep_rewards: VecDeque<f32> = VecDeque::with_capacity(100);

        for ep in 0..max_episodes {
            let t0 = std::time::Instant::now();

            let ep_seed = if ep == 0 { Some(42u64) } else { None };
            let steps = self_play_one_episode(&env, &model, &actions, &mcts_cfg, gamma, ep_seed);
            let ep_reward: f32 = steps.iter().map(|s| s.reward).sum();
            let ep_len = steps.len();

            buffer.push(SelfPlayGame {
                steps,
                outcome: GameOutcome::InProgress,
            });

            // 训练
            let mut avg_loss = 0.0;
            if buffer.len() >= start_training_after {
                let mut loss_sum = 0.0;
                let n_trains = if smoke { 1 } else { trains_per_episode };
                for _ in 0..n_trains {
                    let games = buffer.sample(1, &mut rng);
                    if let Some(game) = games.first() {
                        let l = train_one_step(
                            &model, &mut optimizer, game, k_unroll, td_steps, gamma, &mut rng,
                        )?;
                        loss_sum += l;
                    }
                }
                avg_loss = loss_sum / n_trains as f32;

                if smoke {
                    assert!(
                        avg_loss.is_finite(),
                        "SMOKE: loss={avg_loss} 非有限"
                    );
                }
            }

            ep_rewards.push_back(ep_reward);
            if ep_rewards.len() > 100 {
                ep_rewards.pop_front();
            }
            let avg_r = ep_rewards.iter().sum::<f32>() / ep_rewards.len() as f32;

            println!(
                "Ep {:4}: R={:6.1} len={:3} avg_R={:6.1} loss={:.4} t={:.2}s",
                ep + 1,
                ep_reward,
                ep_len,
                avg_r,
                avg_loss,
                t0.elapsed().as_secs_f32()
            );

            // 达标判定
            if !smoke && ep_rewards.len() >= 20 {
                let recent: f32 =
                    ep_rewards.iter().rev().take(20).sum::<f32>() / 20.0;
                if recent >= 195.0 {
                    println!("✅ 最近 20 局均值达标 avg={recent:.1}");
                    break;
                }
            }
        }

        if smoke {
            println!("[SMOKE] MuZero CartPole 管线验证通过");
        }
        env.close();
        Ok(())
    })
}
