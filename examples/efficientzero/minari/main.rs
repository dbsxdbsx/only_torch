//! EfficientZero Minari 离线示例（Phase 4，离线 pipeline smoke）
//!
//! **best-effort smoke**：从 Minari 离线数据集 load 轨迹 → 离散化连续动作 → 跑一步 EZ K-step
//! unroll 训练，只验「离线数据加载 + 训练管线跑通、loss 有限、无 panic」，**不进性能门禁**。
//! 无本地数据集时优雅跳过（不触发网络下载，避免 smoke 受网络影响）。
//!
//! ```bash
//! pip install minari
//! minari download D4RL/pointmaze/umaze-v2   # 预先下载任一数据集
//! cargo run --example efficientzero_minari --release
//! SMOKE=1 cargo run --example efficientzero_minari
//! ```

#[path = "../cartpole/model.rs"]
mod model;

use model::EzModel;
use only_torch::nn::{Adam, Graph, GraphError, Optimizer};
use only_torch::rl::algo::muzero::compute_n_step_target;
use only_torch::rl::{Episode, GameOutcome, MinariDataset, SelfPlayGame, SelfPlayStep};
use pyo3::Python;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const BINS: usize = 3; // 每个连续维度离散档数
const NUM_ACTIONS: usize = BINS * BINS; // 2 维动作 → 9 候选（高维退化为 idx 0）

/// 连续动作（≈[-1,1]）→ 离散候选 idx（2 维网格；其它维度退化为 0）。
fn action_to_idx(a: &[f32]) -> usize {
    let bucket = |x: f32| -> usize {
        let t = ((x + 1.0) * 0.5 * (BINS as f32 - 1.0)).round();
        (t.max(0.0) as usize).min(BINS - 1)
    };
    if a.len() >= 2 {
        bucket(a[0]) * BINS + bucket(a[1])
    } else if a.len() == 1 {
        bucket(a[0])
    } else {
        0
    }
}

/// 把一个离线 Episode 转成 SelfPlayGame（policy 用 uniform，value 由 n-step 兜底）。
fn episode_to_game(ep: &Episode) -> SelfPlayGame {
    let t = ep
        .actions
        .len()
        .min(ep.rewards.len())
        .min(ep.observations.len());
    let uniform = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
    let mut steps = Vec::with_capacity(t);
    for i in 0..t {
        let terminated = ep.terminations.get(i).copied().unwrap_or(false);
        steps.push(SelfPlayStep {
            obs: ep.observations[i].clone(),
            action: vec![action_to_idx(&ep.actions[i]) as f32],
            policy_target: uniform.clone(),
            player: 0,
            reward: ep.rewards[i],
            root_value: None,
            terminated,
            extras: Default::default(),
        });
    }
    SelfPlayGame {
        steps,
        outcome: GameOutcome::InProgress,
    }
}

fn train_step(
    model: &EzModel,
    optimizer: &mut Adam,
    game: &SelfPlayGame,
    k_unroll: usize,
    td_steps: usize,
    gamma: f32,
    rng: &mut impl Rng,
) -> Result<f32, GraphError> {
    let steps = &game.steps;
    let len = steps.len();
    if len < 2 {
        return Ok(0.0);
    }
    let t = rng.gen_range(0..len - 1);
    let actual_k = k_unroll.min(len - 1 - t);
    let uniform = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];

    let target_policies: Vec<Vec<f32>> = (0..=actual_k)
        .map(|i| {
            steps
                .get(t + i)
                .map(|s| s.policy_target.clone())
                .unwrap_or_else(|| uniform.clone())
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
        .map(|i| steps.get(t + i).map(|s| s.action[0] as usize).unwrap_or(0))
        .collect();
    let next_obs: Vec<Option<Vec<f32>>> = (0..actual_k).map(|_| None).collect();

    optimizer.zero_grad()?;
    let loss = model.train_unroll(
        &steps[t].obs,
        &actions,
        &target_policies,
        &target_values,
        &target_rewards,
        &next_obs,
        0.0,
    )?;
    let v = loss.backward()?;
    optimizer.step()?;
    Ok(v)
}

fn main() -> Result<(), GraphError> {
    let smoke = std::env::var("SMOKE").is_ok();
    let latent_dim = 64;

    Python::attach(|py| {
        let local = MinariDataset::list_local(py);
        if local.is_empty() {
            println!(
                "[EZ Minari smoke] 无本地 Minari 数据集——best-effort 跳过。\n  先 `minari download D4RL/pointmaze/umaze-v2` 再跑（不在 smoke 内联网下载）。"
            );
            if smoke {
                println!("[SMOKE] EZ Minari 优雅跳过（无本地数据集，pipeline 编译/加载逻辑就绪）");
            }
            return Ok(());
        }
        let name = std::env::var("MINARI_DATASET").unwrap_or_else(|_| local[0].clone());
        println!("[EZ Minari smoke] 使用本地数据集: {name}");
        let dataset = MinariDataset::load(py, &name);
        dataset.print_info();

        let n = if smoke { 2 } else { 8 };
        let episodes = dataset.sample_episodes(n);
        let games: Vec<SelfPlayGame> = episodes.iter().map(episode_to_game).collect();
        let obs_dim = games
            .iter()
            .find_map(|g| g.steps.first().map(|s| s.obs.len()))
            .unwrap_or(0);
        if obs_dim == 0 {
            println!("[EZ Minari smoke] 采样到空 episode，跳过训练");
            if smoke {
                println!("[SMOKE] EZ Minari 优雅跳过（空数据）");
            }
            return Ok(());
        }
        println!(
            "[EZ Minari smoke] obs_dim={obs_dim} 连续动作→离散 {NUM_ACTIONS} 候选 episodes={}",
            games.len()
        );

        let graph = Graph::new_with_seed(42);
        let model = EzModel::new(&graph, obs_dim, NUM_ACTIONS, latent_dim, false)?;
        let mut optimizer = Adam::new(&graph, &model.parameters(), 0.01);
        let mut rng = StdRng::seed_from_u64(42);

        let n_steps = if smoke { 2 } else { 50 };
        let mut last_loss = 0.0;
        for it in 0..n_steps {
            for game in &games {
                last_loss = train_step(&model, &mut optimizer, game, 3, 5, 0.99, &mut rng)?;
            }
            if it == 0 || (it + 1) % 10 == 0 {
                println!("  iter {:3}: loss={:.4}", it + 1, last_loss);
            }
        }
        if smoke {
            assert!(last_loss.is_finite(), "SMOKE: loss={last_loss} 非有限");
            println!("[SMOKE] EZ Minari 离线训练管线验证通过");
        }
        Ok(())
    })
}
