//! EfficientZero CartPole-v1 示例（Phase 1）
//!
//! **base** = canonical MuZero（无 EZ 增量），作逐增量消融的基线。后续增量
//! （consistency / value prefix / target net / reanalyze 调强 / SVE）将以可开关方式叠加。
//!
//! ```bash
//! cargo run --example efficientzero_cartpole --release
//! SMOKE=1 cargo run --example efficientzero_cartpole  # 管线验证（3 局 + 1 次训练）
//! ```
//!
//! 验收口径（区别于架构层 v0 的 195）：CartPole-v1（500 步上限），达标门禁建议
//! greedy(temp=0) eval 20 局均值 ≥ 450（base 不强制达标，仅记录基线）。

mod model;

use model::EzModel;
use only_torch::nn::{Adam, Graph, GraphError, Optimizer};
use only_torch::rl::algo::efficientzero::{
    EfficientZeroConfig, hard_update, sve_blend, sync_target,
};
use only_torch::rl::algo::muzero::{compute_n_step_target, reanalyze_game};
use only_torch::rl::mcts::{ActionPayload, DynamicsModel, MctsConfig, PuctPolicy, mcts_search};
use only_torch::rl::{GameOutcome, GymEnv, ReplayBuffer, SelfPlayGame, SelfPlayStep};
use pyo3::Python;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

/// CartPole-v1 达标门禁（greedy eval 20 局均值）。base 仅记录，不强制。
const EVAL_TARGET: f32 = 450.0;

fn self_play_one_episode(
    env: &GymEnv,
    model: &EzModel,
    actions: &[ActionPayload],
    mcts_cfg: &MctsConfig,
    gamma: f32,
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

        let (next_obs_raw, reward, terminated, truncated) = env.step(&[action_idx as f32]);
        let last = steps.last_mut().unwrap();
        last.reward = reward;
        last.terminated = terminated;

        if terminated || truncated {
            break;
        }
        obs = next_obs_raw[0].clone();
    }

    steps
}

/// 真 batch 训练：一次 zero_grad + N 个 position 各自 backward（梯度累积）+ 一次 step。
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
        // canonical absorbing：终止局 full-K + 越界填 absorbing；截断局短 unroll、不补 absorbing。
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
                    // n-step bootstrap 用 buffer 的 root_value（已由 reanalyze 用 target net 刷新，官方口径）
                    let n_step = compute_n_step_target(steps, t + i, td_steps, gamma);
                    // +SVE：把（reanalyze 后刷新的）搜索 root value blend 进 value 目标
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

        // +consistency 的 target：第 i 个 action 之后的真实 next obs（absorbing/越界为 None）
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

/// 贪心 eval：temperature=0 跑若干局取均值（独立种子 rng，不扰训练复现）。
fn eval_episodes(
    env: &GymEnv,
    model: &EzModel,
    actions: &[ActionPayload],
    gamma: f32,
    n_episodes: usize,
    num_simulations: u32,
    seed_base: u64,
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
        // 固定每局 env 种子 → greedy eval 可复现（兑现 README 的 reset(Some(seed)) 承诺）
        let obs_raw = env.reset(Some(seed_base + i as u64));
        let mut obs = obs_raw[0].clone();
        let mut ep_reward = 0.0;

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

    // EZ 组合配置 + 增量消融开关（base 全关；逐步打开对比）。
    let mut cfg = EfficientZeroConfig::default();
    let consistency_coef = if std::env::var("EZ_CONSISTENCY").is_ok() {
        cfg.loss.consistency_coef
    } else {
        0.0
    };
    let use_target = std::env::var("EZ_TARGET").is_ok();
    cfg.target.enabled = use_target;
    // 官方口径：target 用 hard update（每 sync_interval 训练步整体 copy），而非每步 EMA
    cfg.target.sync_interval = 200;
    if std::env::var("EZ_REANALYZE").is_ok() {
        cfg.reanalyze.fraction = 0.5;
    }
    if let Ok(v) = std::env::var("REANALYZE") {
        if let Ok(f) = v.parse::<f32>() {
            cfg.reanalyze.fraction = f.clamp(0.0, 1.0);
        }
    }
    let sve_weight = if std::env::var("EZ_SVE").is_ok() {
        0.5
    } else {
        0.0
    };
    // +SVE 依赖 reanalyze 提供新鲜 root value：若开 SVE 但未开 reanalyze，自动启用
    if sve_weight > 0.0 && cfg.reanalyze.fraction == 0.0 {
        cfg.reanalyze.fraction = 0.5;
        println!("[EZ] SVE 已开启 → 自动启用 reanalyze(0.5) 提供新鲜 root value");
    }
    let use_value_prefix = std::env::var("EZ_VALUE_PREFIX").is_ok();

    let mut base = cfg.base;
    // 调参旋钮：环境变量覆盖（不重编译即可试 num_sim / 训练强度 / lr）
    if let Some(n) = std::env::var("NUM_SIM").ok().and_then(|s| s.parse().ok()) {
        base.num_simulations = n;
    }
    if let Some(n) = std::env::var("TRAINS").ok().and_then(|s| s.parse().ok()) {
        base.trains_per_episode = n;
    }
    if let Some(lr) = std::env::var("LR").ok().and_then(|s| s.parse().ok()) {
        base.lr = lr;
    }
    let reanalyze_fraction = cfg.reanalyze.fraction;
    println!(
        "[EZ ablation] consistency={} value_prefix={} target={} reanalyze={:.2} sve={:.2}",
        consistency_coef > 0.0,
        use_value_prefix,
        use_target,
        reanalyze_fraction,
        sve_weight
    );

    let max_episodes = if smoke {
        3
    } else {
        std::env::var("MAX_EP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2000)
    };
    // smoke 下降低开训阈值，确保 3 局内触发训练（覆盖 train_unroll + backward 管线）
    let start_training_after = if smoke { 2 } else { base.start_training_after };
    // 诊断模式：设 EVAL_EVERY=N → 每 N 局**无条件**跑 greedy eval（不等 self-play 阈值），
    // 用于画 greedy vs self-play 曲线、定位 greedy<self-play 根因。
    let eval_every: usize = std::env::var("EVAL_EVERY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);
    let diag_eval = std::env::var("EVAL_EVERY").is_ok();

    Python::attach(|py| {
        let env = GymEnv::new(py, "CartPole-v1");
        let graph = Graph::new_with_seed(42);
        let model = EzModel::new(&graph, obs_dim, action_dim, latent_dim, use_value_prefix)?;
        let mut optimizer = Adam::new(&graph, &model.parameters(), base.lr);

        // target net（+target）：独立 graph 避免参数命名冲突；初始 hard copy 自 online。
        let target: Option<EzModel> = if use_target {
            let tgraph = Graph::new_with_seed(43);
            let tmodel = EzModel::new(&tgraph, obs_dim, action_dim, latent_dim, use_value_prefix)?;
            hard_update(&model.parameters(), &tmodel.parameters());
            Some(tmodel)
        } else {
            None
        };
        let mut buffer: ReplayBuffer<SelfPlayGame> = ReplayBuffer::new(base.buffer_capacity);
        let mut rng = StdRng::seed_from_u64(42);

        let actions: Vec<ActionPayload> = (0..action_dim).map(ActionPayload::Discrete).collect();
        let mut ep_rewards: VecDeque<f32> = VecDeque::with_capacity(100);
        let mut train_step: u32 = 0;

        for ep in 0..max_episodes {
            let t0 = std::time::Instant::now();

            // 温度退火：前 50% 局 t=1.0，后 50% 线性降到 0.25
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
                let mut loss_sum = 0.0;
                let n_trains = if smoke { 1 } else { base.trains_per_episode };
                for _ in 0..n_trains {
                    let mut games = buffer.sample(base.batch_games, &mut rng);
                    if reanalyze_fraction > 0.0 {
                        let re_cfg = MctsConfig {
                            num_simulations: base.num_simulations,
                            temperature: 1.0,
                            discount: base.gamma,
                            root_exploration_fraction: 0.0,
                            ..MctsConfig::default()
                        };
                        let re_policy = PuctPolicy::new();
                        // +reanalyze：优先用 target net（+target 开启时）刷新旧轨迹目标
                        let re_model = target.as_ref().unwrap_or(&model);
                        for g in games.iter_mut() {
                            if rng.gen_range(0.0..1.0) < reanalyze_fraction {
                                let dyn_model =
                                    DynamicsModel::new(re_model, actions.clone(), base.gamma);
                                reanalyze_game(&dyn_model, &re_policy, g, &re_cfg, &mut rng);
                            }
                        }
                    }
                    let l = train_batch(
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
                    loss_sum += l;

                    // +target（官方口径）：每 sync_interval 训练步对 target 做 hard copy（非 EMA）
                    train_step += 1;
                    if let Some(t) = &target {
                        sync_target(
                            &model.parameters(),
                            &t.parameters(),
                            &cfg.target,
                            train_step,
                        );
                    }
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

            // 定期贪心 eval。诊断模式（EVAL_EVERY 设置）下每 eval_every 局无条件 eval；
            // 否则沿用达标门禁（self-play 近 20 均值 ≥ 200 才触发）。
            if !smoke && ep_rewards.len() >= 20 && (ep + 1) % eval_every == 0 {
                let recent: f32 = ep_rewards.iter().rev().take(20).sum::<f32>() / 20.0;
                if diag_eval || recent >= 200.0 {
                    let eval_r = eval_episodes(
                        &env,
                        &model,
                        &actions,
                        base.gamma,
                        20,
                        base.num_simulations,
                        0xE7A1,
                    );
                    println!(
                        "  贪心 eval 20 局均值={eval_r:.1}（self-play 近20均值={recent:.1}，目标≥{EVAL_TARGET}）"
                    );
                    if eval_r >= EVAL_TARGET {
                        println!("EZ CartPole-v1 达标！greedy eval={eval_r:.1} ≥ {EVAL_TARGET}");
                        break;
                    }
                }
            }
        }

        if smoke {
            println!("[SMOKE] EZ CartPole-v1 base 管线验证通过");
        }
        env.close();
        Ok(())
    })
}
