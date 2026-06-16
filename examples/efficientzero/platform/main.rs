//! EfficientZero Platform-v0 示例（Phase 2b，混合 Tuple 动作）
//!
//! **smoke 级实现**：把混合动作（3 个平台选择 × 连续跳跃参数）**离散化成固定候选集**，
//! 内部以 `Discrete(idx)` 走既有离散 MCTS + EZ 管线，仅在 `env.step` 边界把 idx 映射回
//! `[discrete_idx, c0, c1, c2]`。忠实的 Gumbel 混合搜索（per-state 连续参数采样）留作后续，
//! 本文件只验「管线闭环、loss 有限、无 panic」。
//!
//! ```bash
//! pip install hybrid-platform
//! cargo run --example efficientzero_platform --release
//! SMOKE=1 cargo run --example efficientzero_platform  # 管线验证
//! ```

#[path = "../cartpole/model.rs"]
mod model;

use model::EzModel;
use only_torch::nn::{Adam, Graph, GraphError, Optimizer};
use only_torch::rl::algo::efficientzero::EfficientZeroConfig;
use only_torch::rl::algo::muzero::compute_n_step_target;
use only_torch::rl::mcts::{ActionPayload, DynamicsModel, MctsConfig, PuctPolicy, mcts_search};
use only_torch::rl::{GameOutcome, GymEnv, ReplayBuffer, SelfPlayGame, SelfPlayStep};
use pyo3::Python;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

/// 离散候选 idx → Platform 环境动作 `[discrete_idx, c0, c1, c2]`。
///
/// 3 个平台选择 × 2 个连续参数预设（low/high，∈[-1,1] 对齐 SAC TanhNormal 域）= 6 个候选。
fn candidate_to_env(idx: usize) -> Vec<f32> {
    let presets: [[f32; 3]; 2] = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]];
    let discrete = idx / presets.len();
    let preset = &presets[idx % presets.len()];
    let mut v = vec![discrete as f32];
    v.extend_from_slice(preset);
    v
}

const NUM_ACTIONS: usize = 6; // 3 discrete × 2 连续预设

fn self_play_one_episode(
    env: &GymEnv,
    model: &EzModel,
    actions: &[ActionPayload],
    mcts_cfg: &MctsConfig,
    gamma: f32,
    rng: &mut StdRng,
) -> Vec<SelfPlayStep> {
    let mut obs = env.flatten_obs(&env.reset(None));
    let mut steps = Vec::new();

    loop {
        let dyn_model = DynamicsModel::new(model, actions.to_vec(), gamma);
        let result = mcts_search(&dyn_model, &PuctPolicy::new(), &obs, mcts_cfg, rng);
        let action_idx = match &result.recommended {
            ActionPayload::Discrete(idx) => *idx,
            _ => 0,
        };
        let root_value = result.root_value();

        steps.push(SelfPlayStep {
            obs: obs.clone(),
            action: vec![action_idx as f32],
            policy_target: result.learn_policy,
            player: 0,
            reward: 0.0,
            root_value: Some(root_value),
            terminated: false,
            extras: Default::default(),
        });

        let (next_obs_raw, reward, terminated, truncated) = env.step(&candidate_to_env(action_idx));
        let last = steps.last_mut().unwrap();
        last.reward = reward;
        last.terminated = terminated;

        if terminated || truncated {
            break;
        }
        obs = env.flatten_obs(&next_obs_raw);
    }

    steps
}

#[allow(clippy::too_many_arguments)]
fn train_batch(
    model: &EzModel,
    optimizer: &mut Adam,
    games: &[SelfPlayGame],
    k_unroll: usize,
    td_steps: usize,
    gamma: f32,
    consistency_coef: f32,
    rng: &mut impl Rng,
) -> Result<f32, GraphError> {
    let valid_games: Vec<&SelfPlayGame> = games.iter().filter(|g| g.steps.len() >= 2).collect();
    if valid_games.is_empty() {
        return Ok(0.0);
    }
    let batch_size = valid_games.len() as f32;
    let mut total_loss_val = 0.0;
    optimizer.zero_grad()?;

    for game in &valid_games {
        let steps = &game.steps;
        let len = steps.len();
        let ep_terminated = steps[len - 1].terminated;
        let t = rng.gen_range(0..len);
        let actual_k = if ep_terminated {
            k_unroll
        } else {
            k_unroll.min(len - 1 - t)
        };

        let uniform_policy = vec![1.0 / model.action_dim as f32; model.action_dim];
        let target_policies: Vec<Vec<f32>> = (0..=actual_k)
            .map(|i| {
                if t + i < len {
                    steps[t + i].policy_target.clone()
                } else {
                    uniform_policy.clone()
                }
            })
            .collect();
        let target_values: Vec<f32> = (0..=actual_k)
            .map(|i| {
                if t + i < len {
                    compute_n_step_target(steps, t + i, td_steps, gamma)
                } else {
                    0.0
                }
            })
            .collect();
        let target_rewards: Vec<f32> = (0..actual_k)
            .map(|i| {
                if t + i < len {
                    steps[t + i].reward
                } else {
                    0.0
                }
            })
            .collect();
        let actions: Vec<usize> = (0..actual_k)
            .map(|i| {
                if t + i < len {
                    steps[t + i].action[0] as usize
                } else {
                    0
                }
            })
            .collect();
        let next_obs: Vec<Option<Vec<f32>>> = (0..actual_k)
            .map(|i| {
                let idx = t + i + 1;
                if idx < len {
                    Some(steps[idx].obs.clone())
                } else {
                    None
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
            &next_obs,
            consistency_coef,
        )? * (1.0 / batch_size);
        total_loss_val += loss.backward()?;
    }

    optimizer.step()?;
    Ok(total_loss_val)
}

fn main() -> Result<(), GraphError> {
    let smoke = std::env::var("SMOKE").is_ok();
    let latent_dim = 64;
    let action_dim = NUM_ACTIONS;

    let cfg = EfficientZeroConfig::default();
    let consistency_coef = if std::env::var("EZ_CONSISTENCY").is_ok() {
        cfg.loss.consistency_coef
    } else {
        0.0
    };
    let base = cfg.base;

    let max_episodes = if smoke {
        3
    } else {
        std::env::var("MAX_EP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000)
    };
    let start_training_after = if smoke { 2 } else { base.start_training_after };

    Python::attach(|py| {
        let env = GymEnv::new(py, "Platform-v0");
        let obs_dim = env.get_flatten_observation_len();
        println!(
            "[EZ Platform smoke] obs_dim={obs_dim} 混合动作→离散 {NUM_ACTIONS} 候选（3 选择×2 预设）consistency={}",
            consistency_coef > 0.0
        );

        let graph = Graph::new_with_seed(42);
        let model = EzModel::new(&graph, obs_dim, action_dim, latent_dim, false)?;
        let mut optimizer = Adam::new(&graph, &model.parameters(), base.lr);
        let mut buffer: ReplayBuffer<SelfPlayGame> = ReplayBuffer::new(base.buffer_capacity);
        let mut rng = StdRng::seed_from_u64(42);

        let actions: Vec<ActionPayload> = (0..action_dim).map(ActionPayload::Discrete).collect();
        let mut ep_rewards: VecDeque<f32> = VecDeque::with_capacity(100);

        for ep in 0..max_episodes {
            let t0 = std::time::Instant::now();
            let progress = ep as f32 / max_episodes as f32;
            let temperature = if progress < 0.5 {
                1.0
            } else {
                1.0 - (progress - 0.5) * 2.0 * 0.75
            };
            let mcts_cfg = MctsConfig {
                num_simulations: base.num_simulations,
                temperature,
                discount: base.gamma,
                ..MctsConfig::default()
            };

            let steps =
                self_play_one_episode(&env, &model, &actions, &mcts_cfg, base.gamma, &mut rng);
            let ep_reward: f32 = steps.iter().map(|s| s.reward).sum();
            let ep_len = steps.len();
            buffer.push(SelfPlayGame {
                steps,
                outcome: GameOutcome::InProgress,
            });

            let mut avg_loss = 0.0;
            if buffer.len() >= start_training_after {
                let n_trains = if smoke { 1 } else { base.trains_per_episode };
                let mut loss_sum = 0.0;
                for _ in 0..n_trains {
                    let games = buffer.sample(base.batch_games, &mut rng);
                    loss_sum += train_batch(
                        &model,
                        &mut optimizer,
                        &games,
                        base.k_unroll,
                        base.td_steps,
                        base.gamma,
                        consistency_coef,
                        &mut rng,
                    )?;
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
                "Ep {:4}: R={:7.3} len={:3} avg_R={:7.3} loss={:.4} temp={:.2} t={:.2}s",
                ep + 1,
                ep_reward,
                ep_len,
                avg_r,
                avg_loss,
                temperature,
                t0.elapsed().as_secs_f32()
            );
        }

        if smoke {
            println!("[SMOKE] EZ Platform 管线验证通过");
        }
        env.close();
        Ok(())
    })
}
