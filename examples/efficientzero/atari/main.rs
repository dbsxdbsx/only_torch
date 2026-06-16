//! EfficientZero Atari 示例（Phase 4，图像离散 pipeline smoke）
//!
//! **best-effort smoke**：像素 obs `(210,160,3)` 在 Rust 侧**降采样为灰度小图**（21×16=336）后喂
//! 既有 MLP EZ 管线，只验「图像 env 交互 + 训练管线跑通、loss 有限、无 panic」。忠实的 `Conv2d`
//! CNN repr（plan §Phase 4）留作后续，本文件用降采样换取 CPU 友好的快速 smoke。
//!
//! ```bash
//! pip install "gymnasium[atari]" ale-py autorom
//! cargo run --example efficientzero_atari --release
//! SMOKE=1 cargo run --example efficientzero_atari  # 管线验证
//! ```

#[path = "../cartpole/model.rs"]
mod model;

use model::EzModel;
use only_torch::nn::{Adam, Graph, GraphError, Optimizer};
use only_torch::rl::algo::muzero::compute_n_step_target;
use only_torch::rl::mcts::{ActionPayload, DynamicsModel, MctsConfig, PuctPolicy, mcts_search};
use only_torch::rl::{GameOutcome, GymEnv, ReplayBuffer, SelfPlayGame, SelfPlayStep};
use pyo3::Python;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

const RAW_H: usize = 210;
const RAW_W: usize = 160;
const RAW_C: usize = 3;
const DS: usize = 10; // 降采样步长
const DH: usize = RAW_H / DS; // 21
const DW: usize = RAW_W / DS; // 16
const DOBS: usize = DH * DW; // 336（灰度）
const ACTION_DIM: usize = 4; // Breakout 离散动作数
const REWARD_SCALE: f32 = 1.0;
const SMOKE_STEP_CAP: usize = 40;

/// 像素 (210,160,3) HWC flat → 灰度降采样 (21,16) flat，∈[0,1]。
fn downsample(flat: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(DOBS);
    for h in 0..DH {
        for w in 0..DW {
            let base = ((h * DS) * RAW_W + (w * DS)) * RAW_C;
            let g = if base + 2 < flat.len() {
                (flat[base] + flat[base + 1] + flat[base + 2]) / (3.0 * 255.0)
            } else {
                0.0
            };
            out.push(g);
        }
    }
    out
}

fn self_play_one_episode(
    env: &GymEnv,
    model: &EzModel,
    actions: &[ActionPayload],
    mcts_cfg: &MctsConfig,
    gamma: f32,
    step_cap: usize,
    rng: &mut StdRng,
) -> Vec<SelfPlayStep> {
    let mut obs = downsample(&env.reset(None)[0]);
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

        let (next_obs_raw, reward, terminated, truncated) = env.step(&[action_idx as f32]);
        let last = steps.last_mut().unwrap();
        last.reward = reward * REWARD_SCALE;
        last.terminated = terminated;

        if terminated || truncated || steps.len() >= step_cap {
            break;
        }
        obs = downsample(&next_obs_raw[0]);
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
    let lr = 0.02;
    let num_sim = if smoke { 12 } else { 50 };
    let max_episodes = if smoke {
        3
    } else {
        std::env::var("MAX_EP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(200)
    };
    let step_cap = if smoke { SMOKE_STEP_CAP } else { usize::MAX };

    Python::attach(|py| {
        // ALE 命名空间需先 import ale_py 注册（GymEnv 只调 gymnasium.make）。
        py.import("ale_py")
            .expect("import ale_py 失败：请 `pip install ale-py autorom` 并下载 ROM");
        let env = GymEnv::new(py, "ALE/Breakout-v5");
        println!(
            "[EZ Atari smoke] ALE/Breakout-v5 像素降采样 {DH}x{DW}={DOBS} actions={ACTION_DIM} num_sim={num_sim}（CNN repr 留 TODO）"
        );

        let graph = Graph::new_with_seed(42);
        let model = EzModel::new(&graph, DOBS, ACTION_DIM, latent_dim, false)?;
        let mut optimizer = Adam::new(&graph, &model.parameters(), lr);
        let mut buffer: ReplayBuffer<SelfPlayGame> = ReplayBuffer::new(1000);
        let mut rng = StdRng::seed_from_u64(42);
        let actions: Vec<ActionPayload> = (0..ACTION_DIM).map(ActionPayload::Discrete).collect();
        let mut ep_rewards: VecDeque<f32> = VecDeque::with_capacity(100);

        for ep in 0..max_episodes {
            let t0 = std::time::Instant::now();
            let mcts_cfg = MctsConfig {
                num_simulations: num_sim,
                temperature: 1.0,
                discount: 0.99,
                ..MctsConfig::default()
            };
            let steps =
                self_play_one_episode(&env, &model, &actions, &mcts_cfg, 0.99, step_cap, &mut rng);
            let ep_reward: f32 = steps.iter().map(|s| s.reward).sum();
            let ep_len = steps.len();
            buffer.push(SelfPlayGame {
                steps,
                outcome: GameOutcome::InProgress,
            });

            let mut avg_loss = 0.0;
            if buffer.len() >= 2 {
                let n_trains = if smoke { 1 } else { 8 };
                let mut loss_sum = 0.0;
                for _ in 0..n_trains {
                    let games = buffer.sample(4, &mut rng);
                    loss_sum += train_batch(&model, &mut optimizer, &games, 3, 10, 0.99, &mut rng)?;
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
                "Ep {:4}: R={:6.1} len={:3} avg_R={:6.1} loss={:.4} t={:.2}s",
                ep + 1,
                ep_reward,
                ep_len,
                avg_r,
                avg_loss,
                t0.elapsed().as_secs_f32()
            );
        }

        if smoke {
            println!("[SMOKE] EZ Atari 图像管线验证通过");
        }
        env.close();
        Ok(())
    })
}
