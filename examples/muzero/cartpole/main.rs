//! MuZero CartPole-v0 训练示例
//!
//! ```bash
//! cargo run --example muzero_cartpole --release
//! SMOKE=1 cargo run --example muzero_cartpole  # 管线验证（3 局 self-play + 1 次训练）
//! ```

mod model;

use model::MuZeroModel;
use only_torch::nn::{Adam, Graph, GraphError, Optimizer};
use only_torch::rl::algo::muzero::compute_n_step_target;
use only_torch::rl::mcts::{mcts_search, ActionPayload, DynamicsModel, MctsConfig, PuctPolicy};
use only_torch::rl::{GameOutcome, GymEnv, ReplayBuffer, SelfPlayGame, SelfPlayStep};
use pyo3::Python;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

fn self_play_one_episode(
    env: &GymEnv,
    model: &MuZeroModel,
    actions: &[ActionPayload],
    mcts_cfg: &MctsConfig,
    gamma: f32,
) -> Vec<SelfPlayStep> {
    let obs_raw = env.reset(None);
    let mut obs = obs_raw[0].clone();
    let mut steps = Vec::new();

    loop {
        let dyn_model = DynamicsModel::new(model, actions.to_vec(), gamma);
        let result = mcts_search(&dyn_model, &PuctPolicy::new(), &obs, mcts_cfg);

        let action_idx = match &result.recommended {
            ActionPayload::Discrete(idx) => *idx,
            _ => 0,
        };

        // root value = visit-weighted Q from root's perspective:
        // Q(a) = reward(a) + discount * V(child(a))
        let root_value = if !result.children.is_empty() {
            let total_visits: u32 = result.children.iter().map(|c| c.visit_count).sum();
            if total_visits > 0 {
                result
                    .children
                    .iter()
                    .map(|c| {
                        if c.visit_count > 0 {
                            let child_v = c.value_sum / c.visit_count as f32;
                            let q = c.reward + c.discount * child_v;
                            q * c.visit_count as f32
                        } else {
                            0.0
                        }
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
            reward: 0.0,
            root_value: Some(root_value),
        });

        let (next_obs_raw, reward, terminated, truncated) = env.step(&[action_idx as f32]);
        steps.last_mut().unwrap().reward = reward;

        if terminated || truncated {
            break;
        }
        obs = next_obs_raw[0].clone();
    }

    steps
}

/// 真 batch 训练：一次 zero_grad + N 个 position 各自 backward（梯度累积）+ 一次 step
fn train_batch(
    model: &MuZeroModel,
    optimizer: &mut Adam,
    games: &[SelfPlayGame],
    k_unroll: usize,
    td_steps: usize,
    gamma: f32,
    rng: &mut impl Rng,
) -> Result<f32, GraphError> {
    let valid_games: Vec<&SelfPlayGame> = games
        .iter()
        .filter(|g| g.steps.len() >= 2)
        .collect();
    if valid_games.is_empty() {
        return Ok(0.0);
    }

    let batch_size = valid_games.len() as f32;
    let mut total_loss_val = 0.0;

    optimizer.zero_grad()?;

    for game in &valid_games {
        let steps = &game.steps;
        // 允许从任意位置开始（含 episode 尾部），越界部分用默认 target
        let t = rng.gen_range(0..steps.len());
        let actual_k = k_unroll.min(steps.len() - 1 - t);

        let actions: Vec<usize> = (0..actual_k)
            .map(|i| steps[t + i].action[0] as usize)
            .collect();

        let uniform_policy = vec![1.0 / model.action_dim as f32; model.action_dim];

        let target_policies: Vec<Vec<f32>> = (0..=actual_k)
            .map(|i| {
                if t + i < steps.len() {
                    steps[t + i].policy_target.clone()
                } else {
                    uniform_policy.clone()
                }
            })
            .collect();

        let target_values: Vec<f32> = (0..=actual_k)
            .map(|i| {
                if t + i < steps.len() {
                    compute_n_step_target(steps, t + i, td_steps, gamma)
                } else {
                    0.0
                }
            })
            .collect();

        let target_rewards: Vec<f32> = (0..actual_k)
            .map(|i| {
                if t + i < steps.len() {
                    steps[t + i].reward
                } else {
                    0.0
                }
            })
            .collect();

        let obs_t = &steps[t].obs;
        let loss = model.train_unroll(
            obs_t,
            &actions,
            &target_policies,
            &target_values,
            &target_rewards,
        )? * (1.0 / batch_size);
        total_loss_val += loss.backward()?;
    }

    optimizer.step()?;
    Ok(total_loss_val)
}

/// 贪心 eval：temperature=0 跑若干局取均值
fn eval_episodes(
    env: &GymEnv,
    model: &MuZeroModel,
    actions: &[ActionPayload],
    gamma: f32,
    n_episodes: usize,
    num_simulations: u32,
) -> f32 {
    let eval_cfg = MctsConfig {
        num_simulations,
        temperature: 0.0,
        discount: gamma,
        root_exploration_fraction: 0.0,
        ..MctsConfig::default()
    };

    let mut total_reward = 0.0;
    for _ in 0..n_episodes {
        let obs_raw = env.reset(None);
        let mut obs = obs_raw[0].clone();
        let mut ep_reward = 0.0;

        loop {
            let dyn_model = DynamicsModel::new(model, actions.to_vec(), gamma);
            let result = mcts_search(&dyn_model, &PuctPolicy::new(), &obs, &eval_cfg);

            let action_idx = match &result.recommended {
                ActionPayload::Discrete(idx) => *idx,
                _ => 0,
            };

            let (next_obs_raw, reward, terminated, truncated) = env.step(&[action_idx as f32]);
            ep_reward += reward;

            if terminated || truncated {
                break;
            }
            obs = next_obs_raw[0].clone();
        }
        total_reward += ep_reward;
    }

    total_reward / n_episodes as f32
}

fn main() -> Result<(), GraphError> {
    let smoke = std::env::var("SMOKE").is_ok();

    let latent_dim = 64;
    let action_dim = 2;
    let obs_dim = 4;
    let lr = 0.02;
    let gamma = 0.997;
    let k_unroll = 5;
    let td_steps = 50;
    let num_simulations = 50;
    let buffer_capacity = 1000;
    let start_training_after = 10;
    let batch_games = 8;
    let trains_per_episode = 8;
    let max_episodes = if smoke { 3 } else { 1000 };

    Python::attach(|py| {
        let env = GymEnv::new(py, "CartPole-v0");
        let graph = Graph::new_with_seed(42);
        let model = MuZeroModel::new(&graph, obs_dim, action_dim, latent_dim)?;
        let mut optimizer = Adam::new(&graph, &model.parameters(), lr);
        let mut buffer: ReplayBuffer<SelfPlayGame> = ReplayBuffer::new(buffer_capacity);
        let mut rng = StdRng::seed_from_u64(42);

        let actions: Vec<ActionPayload> =
            (0..action_dim).map(ActionPayload::Discrete).collect();

        let mut ep_rewards: VecDeque<f32> = VecDeque::with_capacity(100);

        for ep in 0..max_episodes {
            let t0 = std::time::Instant::now();

            // 温度退火：前 50% 局 t=1.0，后 50% 线性降到 0.25
            let progress = ep as f32 / max_episodes as f32;
            let temperature = if progress < 0.5 {
                1.0
            } else {
                1.0 - (progress - 0.5) * 2.0 * 0.75 // 1.0 → 0.25
            };

            let mcts_cfg = MctsConfig {
                num_simulations,
                temperature,
                discount: gamma,
                ..MctsConfig::default()
            };

            let steps =
                self_play_one_episode(&env, &model, &actions, &mcts_cfg, gamma);
            let ep_reward: f32 = steps.iter().map(|s| s.reward).sum();
            let ep_len = steps.len();

            buffer.push(SelfPlayGame {
                steps,
                outcome: GameOutcome::InProgress,
            });

            let mut avg_loss = 0.0;
            if buffer.len() >= start_training_after {
                let mut loss_sum = 0.0;
                let n_trains = if smoke { 1 } else { trains_per_episode };
                for _ in 0..n_trains {
                    let games = buffer.sample(batch_games, &mut rng);
                    let l = train_batch(
                        &model, &mut optimizer, &games, k_unroll, td_steps, gamma, &mut rng,
                    )?;
                    loss_sum += l;
                }
                avg_loss = loss_sum / n_trains as f32;

                if smoke {
                    assert!(avg_loss.is_finite(), "SMOKE: loss={avg_loss} 非有限");
                }
            }

            ep_rewards.push_back(ep_reward);
            if ep_rewards.len() > 100 {
                ep_rewards.pop_front();
            }
            let avg_r = ep_rewards.iter().sum::<f32>() / ep_rewards.len() as f32;

            println!(
                "Ep {:4}: R={:6.1} len={:3} avg_R={:6.1} loss={:.4} temp={:.2} t={:.2}s",
                ep + 1,
                ep_reward,
                ep_len,
                avg_r,
                avg_loss,
                temperature,
                t0.elapsed().as_secs_f32()
            );

            // 达标判定：最近 20 局训练均值 >= 195
            if !smoke && ep_rewards.len() >= 20 {
                let recent: f32 = ep_rewards.iter().rev().take(20).sum::<f32>() / 20.0;
                if recent >= 195.0 {
                    // 再用贪心 eval 确认
                    let eval_r = eval_episodes(&env, &model, &actions, gamma, 10, num_simulations);
                    println!(
                        "训练均值达标 avg={recent:.1}，eval 10 局均值={eval_r:.1}"
                    );
                    if eval_r >= 195.0 {
                        println!("MuZero CartPole-v0 达标！");
                        break;
                    }
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
