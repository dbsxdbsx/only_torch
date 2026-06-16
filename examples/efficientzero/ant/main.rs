//! EfficientZero Ant-v5 示例（Phase 4，高维连续 pipeline smoke）
//!
//! **best-effort smoke**：Ant 动作为 8 维连续 ∈[-1,1]，无法网格离散化，故用**固定 K 个候选动作
//! 向量**（启动时一次性采样）走既有离散 MCTS + EZ 管线，仅在 `env.step` 边界映射 idx → 8 维向量。
//! 只验「高维连续管线跑通、loss 有限、无 panic」，**不追分数**（CPU + 离散化候选远非达标配置）。
//!
//! ```bash
//! pip install "gymnasium[mujoco]"
//! cargo run --example efficientzero_ant --release
//! SMOKE=1 cargo run --example efficientzero_ant  # 管线验证
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

const NUM_ACTIONS: usize = 12; // 固定候选动作向量数
const ACTION_DIM: usize = 8; // Ant 连续动作维度
const REWARD_SCALE: f32 = 0.1;
const SMOKE_STEP_CAP: usize = 40; // smoke 下每局步数上限（Ant 局可达 1000 步）

/// 启动时一次性生成 K 个固定候选动作（8 维 ∈[-1,1]）。
fn build_candidates() -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(7);
    (0..NUM_ACTIONS)
        .map(|_| (0..ACTION_DIM).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn self_play_one_episode(
    env: &GymEnv,
    model: &EzModel,
    actions: &[ActionPayload],
    candidates: &[Vec<f32>],
    mcts_cfg: &MctsConfig,
    gamma: f32,
    step_cap: usize,
    rng: &mut StdRng,
) -> Vec<SelfPlayStep> {
    let mut obs = env.reset(None)[0].clone();
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

        let (next_obs_raw, reward, terminated, truncated) = env.step(&candidates[action_idx]);
        let last = steps.last_mut().unwrap();
        last.reward = reward * REWARD_SCALE;
        last.terminated = terminated;

        if terminated || truncated || steps.len() >= step_cap {
            break;
        }
        obs = next_obs_raw[0].clone();
    }
    steps
}

fn train_batch(
    model: &EzModel,
    optimizer: &mut Adam,
    games: &[SelfPlayGame],
    k_unroll: usize,
    td_steps: usize,
    gamma: f32,
    rng: &mut impl Rng,
) -> Result<f32, GraphError> {
    let valid: Vec<&SelfPlayGame> = games.iter().filter(|g| g.steps.len() >= 2).collect();
    if valid.is_empty() {
        return Ok(0.0);
    }
    let batch_size = valid.len() as f32;
    let mut total = 0.0;
    optimizer.zero_grad()?;

    for game in &valid {
        let steps = &game.steps;
        let len = steps.len();
        let ep_terminated = steps[len - 1].terminated;
        let t = rng.gen_range(0..len);
        let actual_k = if ep_terminated {
            k_unroll
        } else {
            k_unroll.min(len - 1 - t)
        };
        let uniform = vec![1.0 / model.action_dim as f32; model.action_dim];
        let target_policies: Vec<Vec<f32>> = (0..=actual_k)
            .map(|i| {
                if t + i < len {
                    steps[t + i].policy_target.clone()
                } else {
                    uniform.clone()
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
        let next_obs: Vec<Option<Vec<f32>>> = (0..actual_k).map(|_| None).collect();

        let loss = model.train_unroll(
            &steps[t].obs,
            &actions,
            &target_policies,
            &target_values,
            &target_rewards,
            &next_obs,
            0.0,
        )? * (1.0 / batch_size);
        total += loss.backward()?;
    }
    optimizer.step()?;
    Ok(total)
}

fn main() -> Result<(), GraphError> {
    let smoke = std::env::var("SMOKE").is_ok();
    let latent_dim = 64;
    let base = EfficientZeroConfig::default().base;
    let candidates = build_candidates();
    let max_episodes = if smoke {
        3
    } else {
        std::env::var("MAX_EP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300)
    };
    let step_cap = if smoke { SMOKE_STEP_CAP } else { usize::MAX };
    let num_sim = if smoke { 12 } else { base.num_simulations };

    Python::attach(|py| {
        let env = GymEnv::new(py, "Ant-v5");
        let obs_dim = env.get_flatten_observation_len();
        println!(
            "[EZ Ant smoke] obs_dim={obs_dim} 连续 {ACTION_DIM} 维→固定 {NUM_ACTIONS} 候选 num_sim={num_sim}"
        );

        let graph = Graph::new_with_seed(42);
        let model = EzModel::new(&graph, obs_dim, NUM_ACTIONS, latent_dim, false)?;
        let mut optimizer = Adam::new(&graph, &model.parameters(), base.lr);
        let mut buffer: ReplayBuffer<SelfPlayGame> = ReplayBuffer::new(base.buffer_capacity);
        let mut rng = StdRng::seed_from_u64(42);
        let actions: Vec<ActionPayload> = (0..NUM_ACTIONS).map(ActionPayload::Discrete).collect();
        let mut ep_rewards: VecDeque<f32> = VecDeque::with_capacity(100);

        for ep in 0..max_episodes {
            let t0 = std::time::Instant::now();
            let mcts_cfg = MctsConfig {
                num_simulations: num_sim,
                temperature: 1.0,
                discount: base.gamma,
                ..MctsConfig::default()
            };
            let steps = self_play_one_episode(
                &env,
                &model,
                &actions,
                &candidates,
                &mcts_cfg,
                base.gamma,
                step_cap,
                &mut rng,
            );
            let ep_reward: f32 = steps.iter().map(|s| s.reward).sum::<f32>() / REWARD_SCALE;
            let ep_len = steps.len();
            buffer.push(SelfPlayGame {
                steps,
                outcome: GameOutcome::InProgress,
            });

            let mut avg_loss = 0.0;
            if buffer.len() >= 2 {
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
                "Ep {:4}: R={:8.1} len={:3} avg_R={:8.1} loss={:.4} t={:.2}s",
                ep + 1,
                ep_reward,
                ep_len,
                avg_r,
                avg_loss,
                t0.elapsed().as_secs_f32()
            );
        }

        if smoke {
            println!("[SMOKE] EZ Ant 高维连续管线验证通过");
        }
        env.close();
        Ok(())
    })
}
