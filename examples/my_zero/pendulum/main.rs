//! MyZero Pendulum-v1 训练示例（纯连续动作 → 离散化候选）
//!
//! 奥卡姆剃刀路径：把 1 维连续力矩 `[lo, hi]` 离散成 K 档候选，复用既有离散 MCTS
//! （`PuctPolicy`）+ MyZero 三网络架构，仅在 `env.step` 边界映射 idx → 连续力矩。
//! 忠实 Gumbel 连续搜索留作后续——**仅当离散化触顶（达不到 SAC 水平）时才需要**。
//!
//! ```bash
//! cargo run --example my_zero_pendulum --release
//! EZ_CONS=1 cargo run --example my_zero_pendulum --release    # S1 consistency
//! SEEDS=3 cargo run --example my_zero_pendulum --release      # 多 seed 稳定基线（取中位数）
//! SMOKE=1 cargo run --example my_zero_pendulum                # 管线验证（3 局 + 1 次训练）
//! ```
//!
//! 达标（plan G2）：greedy(temp=0) eval return ≥ -200，且样本效率优于 SAC。

#[path = "../cartpole/model.rs"]
mod model;

use model::MyZeroModel;
use only_torch::nn::{Adam, Graph, GraphError, Optimizer};
use only_torch::rl::algo::efficientzero::reward_prefix_targets;
use only_torch::rl::algo::muzero::{MuZeroConfig, compute_n_step_target, reanalyze_game};
use only_torch::rl::algo::my_zero::{FeatureSet, MyZeroConfig, completed_q_policy_target};
use only_torch::rl::mcts::{ActionPayload, DynamicsModel, MctsConfig, PuctPolicy, mcts_search};
use only_torch::rl::{GameOutcome, GymEnv, ReplayBuffer, SelfPlayGame, SelfPlayStep};
use pyo3::Python;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

/// 连续力矩离散粒度（[lo, hi] 等分 K 档）。触顶可调大（代价是需要更多模拟）。
const NUM_ACTIONS: usize = 9;
/// Pendulum reward ∈ [-16.27, 0]；缩放使累计 value 落入 categorical support 域。
const REWARD_SCALE: f32 = 0.1;

/// 离散候选 idx → 连续力矩（线性映射到 [lo, hi]）。
fn idx_to_torque(idx: usize, lo: f32, hi: f32) -> f32 {
    if NUM_ACTIONS <= 1 {
        return 0.5 * (lo + hi);
    }
    lo + (hi - lo) * (idx as f32) / ((NUM_ACTIONS - 1) as f32)
}

/// 从环境变量读取 FeatureSet 消融开关
fn features_from_env() -> FeatureSet {
    let mut fs = FeatureSet::base();
    if std::env::var("EZ_CONS").is_ok() {
        fs.consistency = true;
    }
    if std::env::var("EZ_VP").is_ok() {
        fs.value_prefix = true;
    }
    if std::env::var("EZ_TARGET").is_ok() {
        fs.target_net = true;
    }
    if let Ok(v) = std::env::var("EZ_SVE") {
        if let Ok(w) = v.parse::<f32>() {
            fs.sve_weight = w;
        }
    }
    if std::env::var("CQ").is_ok() {
        fs.completed_q_target = true;
    }
    fs
}

/// 打印当前启用的消融特征
fn print_features(fs: &FeatureSet) {
    let mut tags = Vec::new();
    if fs.consistency {
        tags.push("consistency(S1)");
    }
    if fs.value_prefix {
        tags.push("value_prefix(S2)");
    }
    if fs.target_net {
        tags.push("target_net(S3)");
    }
    if fs.sve_enabled() {
        tags.push("SVE(S4)");
    }
    if fs.gumbel {
        tags.push("Gumbel(S5)");
    }
    if fs.completed_q_target {
        tags.push("completedQ-target");
    }
    if tags.is_empty() {
        println!("[MyZero] features: base (all features off)");
    } else {
        println!("[MyZero] features: {}", tags.join(" + "));
    }
}

#[allow(clippy::too_many_arguments)]
fn self_play_one_episode(
    env: &GymEnv,
    model: &MyZeroModel,
    actions: &[ActionPayload],
    mcts_cfg: &MctsConfig,
    gamma: f32,
    lo: f32,
    hi: f32,
    cq: Option<f32>,
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
        // 策略目标：completedQ 改进策略 或 visit-count（A/B 开关）
        let policy_target = match cq {
            Some(c_scale) => completed_q_policy_target(&result.children, root_value, 50.0, c_scale),
            None => result.learn_policy,
        };

        steps.push(SelfPlayStep {
            obs: obs.clone(),
            action: vec![action_idx as f32],
            policy_target,
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

/// 真 batch 训练：一次 zero_grad + N 个 position 各自 backward（梯度累积）+ 一次 step
fn train_batch(
    model: &MyZeroModel,
    optimizer: &mut Adam,
    games: &[SelfPlayGame],
    k_unroll: usize,
    td_steps: usize,
    gamma: f32,
    features: &FeatureSet,
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

        let obs_t = &steps[t].obs;

        let final_rewards = if features.value_prefix {
            reward_prefix_targets(&target_rewards)
        } else {
            target_rewards
        };

        let consistency_coef = if features.consistency { 2.0 } else { 0.0 };
        let next_obs_list: Option<Vec<Vec<f32>>> = if features.consistency {
            Some(
                (0..actual_k)
                    .take_while(|&i| t + i + 1 < len)
                    .map(|i| steps[t + i + 1].obs.clone())
                    .collect(),
            )
        } else {
            None
        };

        let loss = model.train_unroll(
            obs_t,
            &actions,
            &target_policies,
            &target_values,
            &final_rewards,
            next_obs_list.as_deref(),
            consistency_coef,
            features.value_prefix,
        )? * (1.0 / batch_size);
        total_loss_val += loss.backward()?;
    }

    optimizer.step()?;
    Ok(total_loss_val)
}

/// 贪心 eval：temperature=0 跑若干局取均值（返回**原始未缩放** return）。
#[allow(clippy::too_many_arguments)]
fn eval_episodes(
    env: &GymEnv,
    model: &MyZeroModel,
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

/// 单次训练运行的 benchmark 结果（多 seed 汇总用）
struct RunResult {
    seed: u64,
    wall_secs: f32,
    greedy_eval: f32,
    /// 首次 greedy eval ≥ solved 时的累计 env-step（None = 未达标）
    solved_at_steps: Option<u64>,
}

/// 取中位数（多 seed 汇总口径）
fn median_f32(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// 打印多 seed 汇总（中位数 = 稳定基线口径）
fn print_multiseed_summary(results: &[RunResult], solved: f32) {
    let n = results.len();
    let greedy: Vec<f32> = results.iter().map(|r| r.greedy_eval).collect();
    let walls: Vec<f32> = results.iter().map(|r| r.wall_secs).collect();
    let solved_steps: Vec<f32> = results
        .iter()
        .filter_map(|r| r.solved_at_steps.map(|s| s as f32))
        .collect();
    let n_solved = solved_steps.len();

    println!("\n========== 多 seed 汇总（{n} seeds, solved≥{solved}）==========");
    for r in results {
        let eff = r
            .solved_at_steps
            .map(|s| s.to_string())
            .unwrap_or_else(|| "未达标".to_string());
        println!(
            "  seed={:<3} greedy={:8.1} env_steps_to_solved={:>8} wall={:.1}s",
            r.seed, r.greedy_eval, eff, r.wall_secs
        );
    }
    let steps_med = if n_solved > 0 {
        format!("{:.0}", median_f32(solved_steps))
    } else {
        "n/a".to_string()
    };
    println!(
        "  中位数: greedy={:.1} | env_steps_to_solved={} ({}/{} 达标) | wall={:.1}s",
        median_f32(greedy),
        steps_med,
        n_solved,
        n,
        median_f32(walls),
    );
}

/// 跑一次完整训练（固定 seed），返回 benchmark 结果
#[allow(clippy::too_many_arguments)]
fn run_one_training(
    py: Python<'_>,
    seed: u64,
    cfg: &MyZeroConfig,
    latent_dim: usize,
    max_episodes: usize,
    smoke: bool,
    solved: f32,
) -> Result<RunResult, GraphError> {
    let wall_t0 = std::time::Instant::now();

    let MuZeroConfig {
        gamma,
        k_unroll,
        td_steps,
        num_simulations,
        lr,
        batch_games,
        trains_per_episode,
        buffer_capacity,
        start_training_after,
        reanalyze_fraction,
    } = cfg.base;
    let features = cfg.features.clone();
    // completedQ 改进策略目标的 c_scale（CQ_SCALE 可调；默认 0.02 同 CartPole，按环境再扫）。
    let cq: Option<f32> = if features.completed_q_target {
        Some(
            std::env::var("CQ_SCALE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.02_f32),
        )
    } else {
        None
    };

    let env = GymEnv::new(py, "Pendulum-v1");
    let obs_dim = env.get_flatten_observation_len();
    let ranges = env.get_all_action_valid_range();
    let (lo, hi) = ranges[0].get_continuous_action_low_high();
    let action_dim = NUM_ACTIONS;

    if seed == 42 {
        println!(
            "[MyZero Pendulum] obs_dim={obs_dim} action(连续)→离散 {NUM_ACTIONS} 档 ∈[{lo:.2},{hi:.2}] gamma={gamma} sims={num_simulations}"
        );
    }

    let graph = Graph::new_with_seed(seed);
    let model = MyZeroModel::new(&graph, obs_dim, action_dim, latent_dim)?;
    let mut optimizer = Adam::new(&graph, &model.parameters(), lr);
    let mut buffer: ReplayBuffer<SelfPlayGame> = ReplayBuffer::new(buffer_capacity);
    let mut rng = StdRng::seed_from_u64(seed);

    let actions: Vec<ActionPayload> = (0..action_dim).map(ActionPayload::Discrete).collect();

    let mut ep_rewards: VecDeque<f32> = VecDeque::with_capacity(100);
    let mut total_steps: u64 = 0;
    let mut hit_solved: Option<(usize, u64)> = None;

    for ep in 0..max_episodes {
        let t0 = std::time::Instant::now();

        let progress = ep as f32 / max_episodes as f32;
        let temperature = if progress < 0.5 {
            1.0
        } else {
            1.0 - (progress - 0.5) * 2.0 * 0.75
        };

        let mcts_cfg = MctsConfig {
            num_simulations,
            temperature,
            discount: gamma,
            ..MctsConfig::default()
        };

        let steps =
            self_play_one_episode(&env, &model, &actions, &mcts_cfg, gamma, lo, hi, cq, &mut rng);
        // 原始 return（反缩放回报告）
        let ep_reward: f32 = steps.iter().map(|s| s.reward).sum::<f32>() / REWARD_SCALE;
        let ep_len = steps.len();
        total_steps += ep_len as u64;

        buffer.push(SelfPlayGame {
            steps,
            outcome: GameOutcome::InProgress,
        });

        let mut avg_loss = 0.0;
        if buffer.len() >= start_training_after {
            let mut loss_sum = 0.0;
            let n_trains = if smoke { 1 } else { trains_per_episode };
            for _ in 0..n_trains {
                let mut games = buffer.sample(batch_games, &mut rng);
                if reanalyze_fraction > 0.0 {
                    let re_cfg = MctsConfig {
                        num_simulations,
                        temperature: 1.0,
                        discount: gamma,
                        root_exploration_fraction: 0.0,
                        ..MctsConfig::default()
                    };
                    let re_policy = PuctPolicy::new();
                    for g in games.iter_mut() {
                        if rng.gen_range(0.0..1.0) < reanalyze_fraction {
                            let dyn_model = DynamicsModel::new(&model, actions.clone(), gamma);
                            reanalyze_game(&dyn_model, &re_policy, g, &re_cfg, &mut rng);
                        }
                    }
                }
                let l = train_batch(
                    &model,
                    &mut optimizer,
                    &games,
                    k_unroll,
                    td_steps,
                    gamma,
                    &features,
                    &mut rng,
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
            let eval_r = eval_episodes(&env, &model, &actions, gamma, lo, hi, 10, num_simulations);
            let recent: f32 = ep_rewards.iter().rev().take(20).sum::<f32>() / 20.0;
            println!(
                "  贪心 eval 10 局均值={eval_r:.1}（self-play 近20均值={recent:.1} env_steps={total_steps}）"
            );
            if eval_r >= solved {
                hit_solved = Some((ep + 1, total_steps));
                println!(
                    "✅ MyZero Pendulum-v1 达标！greedy eval={eval_r:.1} ≥ {solved}（ep={} env_steps={}）",
                    ep + 1,
                    total_steps
                );
                break;
            }
        }
    }

    let wall_secs = wall_t0.elapsed().as_secs_f32();
    let greedy_eval = if smoke {
        0.0
    } else {
        eval_episodes(&env, &model, &actions, gamma, lo, hi, 10, num_simulations)
    };

    println!(
        "[benchmark] seed={seed} env_steps={total_steps} wall_clock={wall_secs:.1}s greedy_eval={greedy_eval:.1}"
    );

    if !smoke {
        let eff = hit_solved
            .map(|(e, s)| format!("ep{e} / {s} env-steps"))
            .unwrap_or_else(|| "未达标".to_string());
        println!("📈 [样本效率] Pendulum-v1 MyZero 到 {solved}: {eff}");
    }

    env.close();

    Ok(RunResult {
        seed,
        wall_secs,
        greedy_eval,
        solved_at_steps: hit_solved.map(|(_, s)| s),
    })
}

fn main() -> Result<(), GraphError> {
    let smoke = std::env::var("SMOKE").is_ok();

    let latent_dim = 64;
    // Pendulum 接近最优 return ≈ -150；plan G2 门禁 ≥ -200。可用 SOLVED 覆盖。
    let solved: f32 = std::env::var("SOLVED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(-200.0);

    let features = features_from_env();
    print_features(&features);

    let mut cfg = MyZeroConfig {
        features,
        ..MyZeroConfig::default()
    };

    // Pendulum 是 200 步密集奖励连续控制：用更短折扣（0.99）通常比 0.997 学得更稳更快。
    cfg.base.gamma = std::env::var("GAMMA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.99);

    if let Ok(v) = std::env::var("SIMS") {
        if let Ok(n) = v.parse::<u32>() {
            cfg.base.num_simulations = n;
        }
    }
    if let Ok(v) = std::env::var("REANALYZE") {
        if let Ok(f) = v.parse::<f32>() {
            cfg.base.reanalyze_fraction = f.clamp(0.0, 1.0);
        }
    }

    let max_episodes = if smoke {
        3
    } else {
        std::env::var("MAX_EP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(600)
    };

    let n_seeds: u64 = std::env::var("SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
    let base_seed: u64 = 42;

    Python::attach(|py| {
        let mut results = Vec::new();
        for i in 0..n_seeds {
            let seed = base_seed + i;
            if n_seeds > 1 {
                println!("\n===== seed {seed}（{}/{n_seeds}）=====", i + 1);
            }
            let r = run_one_training(py, seed, &cfg, latent_dim, max_episodes, smoke, solved)?;
            results.push(r);
        }

        if n_seeds > 1 && !smoke {
            print_multiseed_summary(&results, solved);
        }

        if smoke {
            println!("[SMOKE] MyZero Pendulum 管线验证通过");
        }
        Ok(())
    })
}
