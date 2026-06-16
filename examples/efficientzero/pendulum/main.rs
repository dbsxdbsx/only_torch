//! EfficientZero Pendulum-v1 示例（Phase 2a，纯连续动作）
//!
//! **smoke 级实现**：把连续力矩 `[lo, hi]` **离散化成 K 个候选动作**，复用既有离散 MCTS +
//! EZ 训练管线（内部 `action_dim = K`，仅在 `env.step` 边界映射 idx → 连续力矩）。忠实的
//! Gumbel 连续搜索（per-state 采样 + sequential halving）留作 Phase 2a **达标**目标，本文件
//! 只验「管线跑通、loss 有限、无 panic」。
//!
//! ```bash
//! cargo run --example efficientzero_pendulum --release
//! SMOKE=1 cargo run --example efficientzero_pendulum  # 管线验证（3 局 + 1 次训练）
//! ```

#[path = "../cartpole/model.rs"]
mod model;

use model::EzModel;
use only_torch::nn::{Adam, Graph, GraphError, Optimizer};
use only_torch::rl::algo::efficientzero::{EfficientZeroConfig, sve_blend};
use only_torch::rl::algo::muzero::compute_n_step_target;
use only_torch::rl::mcts::{ActionPayload, DynamicsModel, MctsConfig, PuctPolicy, mcts_search};
use only_torch::rl::{GameOutcome, GymEnv, ReplayBuffer, SelfPlayGame, SelfPlayStep};
use pyo3::Python;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

/// 候选动作离散粒度（连续力矩离散成 K 档）。
const NUM_ACTIONS: usize = 9;
/// Pendulum reward ∈ [-16.27, 0]；缩放使累计 value 落入 categorical support 域。
const REWARD_SCALE: f32 = 0.1;

/// 离散候选 idx → 连续力矩值（线性映射到 [lo, hi]）。
fn idx_to_torque(idx: usize, lo: f32, hi: f32) -> f32 {
    if NUM_ACTIONS <= 1 {
        return 0.5 * (lo + hi);
    }
    lo + (hi - lo) * (idx as f32) / ((NUM_ACTIONS - 1) as f32)
}

fn self_play_one_episode(
    env: &GymEnv,
    model: &EzModel,
    actions: &[ActionPayload],
    mcts_cfg: &MctsConfig,
    gamma: f32,
    lo: f32,
    hi: f32,
    rng: &mut StdRng,
) -> Vec<SelfPlayStep> {
    let obs_raw = env.reset(None);
    let mut obs = obs_raw[0].clone();
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

        let torque = idx_to_torque(action_idx, lo, hi);
        let (next_obs_raw, reward, terminated, truncated) = env.step(&[torque]);
        let last = steps.last_mut().unwrap();
        last.reward = reward * REWARD_SCALE;
        last.terminated = terminated;

        if terminated || truncated {
            break;
        }
        obs = next_obs_raw[0].clone();
    }

    steps
}

/// 真 batch 训练（与 cartpole 同口径：一次 zero_grad + N position 梯度累积 + 一次 step）。
#[allow(clippy::too_many_arguments)]
fn train_batch(
    model: &EzModel,
    optimizer: &mut Adam,
    games: &[SelfPlayGame],
    k_unroll: usize,
    td_steps: usize,
    gamma: f32,
    consistency_coef: f32,
    sve_weight: f32,
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
                    let n_step = compute_n_step_target(steps, t + i, td_steps, gamma);
                    if sve_weight > 0.0 {
                        let search_v = steps[t + i].root_value.unwrap_or(n_step);
                        sve_blend(n_step, search_v, sve_weight)
                    } else {
                        n_step
                    }
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

/// 贪心 eval：temperature=0 跑若干局取均值（返回**原始**未缩放 return）。
#[allow(clippy::too_many_arguments)]
fn eval_episodes(
    env: &GymEnv,
    model: &EzModel,
    actions: &[ActionPayload],
    gamma: f32,
    lo: f32,
    hi: f32,
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
    let mut eval_rng = StdRng::seed_from_u64(0xE7A1);
    let mut total_reward = 0.0;
    for i in 0..n_episodes {
        let obs_raw = env.reset(Some(0xE7A1 + i as u64));
        let mut obs = obs_raw[0].clone();
        loop {
            let dyn_model = DynamicsModel::new(model, actions.to_vec(), gamma);
            let result = mcts_search(
                &dyn_model,
                &PuctPolicy::new(),
                &obs,
                &eval_cfg,
                &mut eval_rng,
            );
            let action_idx = match &result.recommended {
                ActionPayload::Discrete(idx) => *idx,
                _ => 0,
            };
            let torque = idx_to_torque(action_idx, lo, hi);
            let (next_obs_raw, reward, terminated, truncated) = env.step(&[torque]);
            total_reward += reward;
            if terminated || truncated {
                break;
            }
            obs = next_obs_raw[0].clone();
        }
    }
    total_reward / n_episodes as f32
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
    let sve_weight = if std::env::var("EZ_SVE").is_ok() {
        0.5
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
            .unwrap_or(800)
    };
    let start_training_after = if smoke { 2 } else { base.start_training_after };

    Python::attach(|py| {
        let env = GymEnv::new(py, "Pendulum-v1");
        let obs_dim = env.get_flatten_observation_len();
        let ranges = env.get_all_action_valid_range();
        let (lo, hi) = ranges[0].get_continuous_action_low_high();
        println!(
            "[EZ Pendulum smoke] obs_dim={obs_dim} action(连续)→离散 {NUM_ACTIONS} 档 ∈[{lo:.2},{hi:.2}] consistency={} sve={:.2}",
            consistency_coef > 0.0,
            sve_weight
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

            let steps = self_play_one_episode(
                &env, &model, &actions, &mcts_cfg, base.gamma, lo, hi, &mut rng,
            );
            // 原始 return（反缩放回报告）
            let ep_reward: f32 = steps.iter().map(|s| s.reward).sum::<f32>() / REWARD_SCALE;
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
                        sve_weight,
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
                "Ep {:4}: R={:8.1} len={:3} avg_R={:8.1} loss={:.4} temp={:.2} t={:.2}s",
                ep + 1,
                ep_reward,
                ep_len,
                avg_r,
                avg_loss,
                temperature,
                t0.elapsed().as_secs_f32()
            );

            if !smoke && ep_rewards.len() >= 20 && (ep + 1) % 25 == 0 {
                let eval_r = eval_episodes(
                    &env,
                    &model,
                    &actions,
                    base.gamma,
                    lo,
                    hi,
                    10,
                    base.num_simulations,
                );
                println!("  贪心 eval 10 局均值={eval_r:.1}（best-effort，建议达标 ≥ -200）");
            }
        }

        if smoke {
            println!("[SMOKE] EZ Pendulum 管线验证通过");
        }
        env.close();
        Ok(())
    })
}
