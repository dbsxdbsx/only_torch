//! MyZero 统一运行入口：self-play + 训练 + greedy eval + 多 seed + SMOKE + DIAG。

use super::action::ActionAdapter;
use super::checkpoint::BestTracker;
use super::component::Components;
use super::config::{MyZeroConfig, greedy_episode_seed, self_play_episode_seed};
use super::my_zero::MyZero;
use super::n_step::compute_n_step_target;
use super::network::MyZeroModel;
use super::reanalyze::reanalyze_game;
use super::report::TrainReport;
use super::target::completed_q_policy_target;
use super::value_prefix::reward_prefix_targets;
use crate::nn::{Adam, Graph, GraphError, Optimizer};
use crate::rl::mcts::{
    ActionPayload, Dynamics, DynamicsModel, MctsConfig, PuctPolicy, mcts_search,
};
use crate::rl::{GameOutcome, GymEnv, ReplayBuffer, SelfPlayGame, SelfPlayStep};
use pyo3::Python;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

/// 打印已启用的消融组件（全关时不输出）。
fn print_components(c: &Components) {
    let mut tags = Vec::new();
    if c.consistency {
        tags.push("consistency");
    }
    if c.value_prefix {
        tags.push("value_prefix");
    }
    if c.target_net {
        tags.push("target_net");
    }
    if c.sve_enabled() {
        tags.push("SVE");
    }
    if c.gumbel {
        tags.push("Gumbel");
    }
    if c.completed_q_target {
        tags.push("completedQ");
    }
    if !tags.is_empty() {
        println!("[MyZero] {}", tags.join(" + "));
    }
}

#[allow(clippy::too_many_arguments)]
fn self_play_one_episode(
    env: &GymEnv,
    model: &MyZeroModel,
    adapter: &ActionAdapter,
    mcts_cfg: &MctsConfig,
    gamma: f32,
    cq: Option<(f32, f32)>, // Some((c_visit, c_scale)) → completedQ 策略目标；None → visit-count
    reward_scale: f32,
    reset_seed: u64,
    rng: &mut StdRng,
) -> Vec<SelfPlayStep> {
    let obs_raw = env.reset(Some(reset_seed));
    let mut obs = obs_raw[0].clone();
    let mut steps = Vec::new();

    loop {
        let dyn_model = DynamicsModel::new(model, adapter.candidates().to_vec(), gamma);
        let result = mcts_search(&dyn_model, &PuctPolicy::new(), &obs, mcts_cfg, rng);

        let action_idx = match &result.recommended {
            ActionPayload::Discrete(idx) => *idx,
            _ => 0,
        };

        let root_value = result.root_value();

        // 策略目标：completedQ 改进策略 或 visit-count（A/B 开关）
        let policy_target = match cq {
            Some((c_visit, c_scale)) => {
                completed_q_policy_target(&result.children, root_value, c_visit, c_scale)
            }
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

        let env_action = adapter.to_env(action_idx);
        let (next_obs_raw, reward, terminated, truncated) = env.step(&env_action);
        let last = steps.last_mut().unwrap();
        last.reward = reward * reward_scale;
        last.terminated = terminated;

        if terminated || truncated {
            break;
        }
        obs = next_obs_raw[0].clone();
    }

    steps
}

/// 真 batch 训练：一次 zero_grad + N 个 position 各自 backward（梯度累积）+ 一次 step
#[allow(clippy::too_many_arguments)]
fn train_batch(
    model: &MyZeroModel,
    optimizer: &mut Adam,
    games: &[SelfPlayGame],
    k_unroll: usize,
    td_steps: usize,
    gamma: f32,
    components: &Components,
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

        // value prefix：把单步 reward 转为累计前缀目标
        let final_rewards = if components.value_prefix {
            reward_prefix_targets(&target_rewards)
        } else {
            target_rewards
        };

        // consistency：收集 unroll 每步对应的真实 next_obs
        let consistency_coef = if components.consistency { 2.0 } else { 0.0 };
        let next_obs_list: Option<Vec<Vec<f32>>> = if components.consistency {
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
            components.value_prefix,
        )? * (1.0 / batch_size);
        total_loss_val += loss.backward()?;
    }

    optimizer.step()?;
    Ok(total_loss_val)
}

/// 贪心 rollout 单局（原始未缩放 return + 步数）。
pub(crate) fn greedy_one_episode(
    env: &GymEnv,
    model: &MyZeroModel,
    adapter: &ActionAdapter,
    gamma: f32,
    num_simulations: u32,
    reset_seed: u64,
) -> (f32, usize) {
    let eval_cfg = MctsConfig {
        num_simulations,
        temperature: 0.0,
        discount: gamma,
        root_exploration_fraction: 0.0,
        ..MctsConfig::default()
    };
    let mut rng = StdRng::seed_from_u64(reset_seed);
    let mut obs = env.reset(Some(reset_seed))[0].clone();
    let mut total_reward = 0.0f32;
    let mut length = 0usize;
    loop {
        let dyn_model = DynamicsModel::new(model, adapter.candidates().to_vec(), gamma);
        let result = mcts_search(&dyn_model, &PuctPolicy::new(), &obs, &eval_cfg, &mut rng);
        let action_idx = match &result.recommended {
            ActionPayload::Discrete(idx) => *idx,
            _ => 0,
        };
        let env_action = adapter.to_env(action_idx);
        let (next_obs_raw, reward, terminated, truncated) = env.step(&env_action);
        total_reward += reward;
        length += 1;
        if terminated || truncated {
            break;
        }
        obs = next_obs_raw[0].clone();
    }
    (total_reward, length)
}

/// 贪心 eval：跑 `n_episodes` 局，返回均值与各局 return（**原始未缩放**）。
pub(crate) fn greedy_eval_episodes(
    env: &GymEnv,
    model: &MyZeroModel,
    adapter: &ActionAdapter,
    gamma: f32,
    n_episodes: usize,
    num_simulations: u32,
    eval_seed: u64,
) -> (f32, Vec<f32>) {
    let mut returns = Vec::with_capacity(n_episodes);
    for i in 0..n_episodes {
        let (r, _) = greedy_one_episode(
            env,
            model,
            adapter,
            gamma,
            num_simulations,
            greedy_episode_seed(eval_seed, i as u64),
        );
        returns.push(r);
    }
    let mean = if n_episodes == 0 {
        0.0
    } else {
        returns.iter().sum::<f32>() / n_episodes as f32
    };
    (mean, returns)
}

/// 贪心 eval：temperature=0 跑若干局取均值（返回**原始未缩放** return）。
#[allow(dead_code)]
fn eval_episodes(
    env: &GymEnv,
    model: &MyZeroModel,
    adapter: &ActionAdapter,
    gamma: f32,
    n_episodes: usize,
    num_simulations: u32,
    eval_seed: u64,
) -> f32 {
    greedy_eval_episodes(
        env,
        model,
        adapter,
        gamma,
        n_episodes,
        num_simulations,
        eval_seed,
    )
    .0
}

/// Dynamics 诊断：对比 learned model 的「想象」reward/value 与真实环境。
///
/// 用 greedy 策略跑一个 episode，逐步比较模型预测（反 scale 回原始空间）与真实值。
/// 若 reward/value 预测**坍缩成常数**或与真实严重不符，说明 learned model 没学准，
/// 瓶颈在训练/表示/目标，而非搜索分辨率。仅诊断、不改训练。
fn dynamics_diagnostic(
    env: &GymEnv,
    model: &MyZeroModel,
    adapter: &ActionAdapter,
    gamma: f32,
    reward_scale: f32,
    num_simulations: u32,
) {
    let eval_cfg = MctsConfig {
        num_simulations,
        temperature: 0.0,
        discount: gamma,
        root_exploration_fraction: 0.0,
        ..MctsConfig::default()
    };
    let mut rng = StdRng::seed_from_u64(0xD1A6);
    let mut obs = env.reset(Some(0xD1A6))[0].clone();

    let mut obses: Vec<Vec<f32>> = Vec::new();
    let mut act_idxs: Vec<usize> = Vec::new();
    let mut true_rewards: Vec<f32> = Vec::new();
    loop {
        let dyn_model = DynamicsModel::new(model, adapter.candidates().to_vec(), gamma);
        let result = mcts_search(&dyn_model, &PuctPolicy::new(), &obs, &eval_cfg, &mut rng);
        let action_idx = match &result.recommended {
            ActionPayload::Discrete(idx) => *idx,
            _ => 0,
        };
        let env_action = adapter.to_env(action_idx);
        let (next_obs_raw, reward, terminated, truncated) = env.step(&env_action);
        obses.push(obs.clone());
        act_idxs.push(action_idx);
        true_rewards.push(reward);
        if terminated || truncated {
            break;
        }
        obs = next_obs_raw[0].clone();
    }

    let n = obses.len();
    if n == 0 {
        println!("[诊断] 空 episode，跳过");
        return;
    }

    // 真实折扣 MC return（原始空间）
    let mut mc_return = vec![0.0f32; n];
    let mut acc = 0.0;
    for t in (0..n).rev() {
        acc = true_rewards[t] + gamma * acc;
        mc_return[t] = acc;
    }

    let mut r_preds = Vec::with_capacity(n);
    let mut v_roots = Vec::with_capacity(n);
    let mut reward_mae = 0.0f32;
    let mut value_mae = 0.0f32;
    for t in 0..n {
        let (latent, _prior, rv_scaled) = Dynamics::initial_state(&model, &obses[t]);
        let out = Dynamics::recurrent(&model, &latent, &ActionPayload::Discrete(act_idxs[t]));
        let r_pred = out.reward / reward_scale;
        let v_root = rv_scaled / reward_scale;
        reward_mae += (r_pred - true_rewards[t]).abs();
        value_mae += (v_root - mc_return[t]).abs();
        r_preds.push(r_pred);
        v_roots.push(v_root);
    }
    reward_mae /= n as f32;
    value_mae /= n as f32;

    let stats = |v: &[f32]| -> (f32, f32, f32, f32) {
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        (mean, std, min, max)
    };
    let (tr_m, tr_s, tr_lo, tr_hi) = stats(&true_rewards);
    let (rp_m, rp_s, rp_lo, rp_hi) = stats(&r_preds);
    let (vt_m, vt_s, vt_lo, vt_hi) = stats(&mc_return);
    let (vp_m, vp_s, vp_lo, vp_hi) = stats(&v_roots);

    println!(
        "\n========== Dynamics 诊断（greedy episode，n={n} 步；已反 scale 回原始空间）=========="
    );
    println!("逐步样本（t | r_true vs r_pred | v_mc vs v_root_pred）：");
    for t in 0..n {
        if t < 5 || t % 40 == 0 || t == n - 1 {
            println!(
                "  t={t:3} | r {:8.3} vs {:8.3} | v {:9.1} vs {:9.1}",
                true_rewards[t], r_preds[t], mc_return[t], v_roots[t]
            );
        }
    }
    println!("----------------------------------------------------------------------------------");
    println!("单步 reward：真实 mean={tr_m:.3} std={tr_s:.3} range=[{tr_lo:.3},{tr_hi:.3}]");
    println!(
        "            预测 mean={rp_m:.3} std={rp_s:.3} range=[{rp_lo:.3},{rp_hi:.3}] | MAE={reward_mae:.3}"
    );
    println!("root value：真实(MC) mean={vt_m:.1} std={vt_s:.1} range=[{vt_lo:.1},{vt_hi:.1}]");
    println!(
        "            预测     mean={vp_m:.1} std={vp_s:.1} range=[{vp_lo:.1},{vp_hi:.1}] | MAE={value_mae:.1}"
    );
    println!("==================================================================================");
}

/// 单次训练运行的 benchmark 结果（多 seed 汇总用）
struct SeedSummary {
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

/// 打印多 seed 汇总（中位数 = 稳定基线口径，排除单 seed 偶发 spike）
fn print_multiseed_summary(results: &[SeedSummary], solved: f32) {
    let n = results.len();
    let greedy: Vec<f32> = results.iter().map(|r| r.greedy_eval).collect();
    let walls: Vec<f32> = results.iter().map(|r| r.wall_secs).collect();
    let solved_steps: Vec<f32> = results
        .iter()
        .filter_map(|r| r.solved_at_steps.map(|s| s as f32))
        .collect();
    let n_solved = solved_steps.len();

    println!("\n--- 多 seed 汇总 ({n} seeds, 门槛 {solved}) ---");
    for r in results {
        let eff = r
            .solved_at_steps
            .map(|s| s.to_string())
            .unwrap_or_else(|| "未达标".to_string());
        println!(
            "  seed={:<3} greedy={:8.1} total_env_steps={:>8} wall={:.1}s",
            r.seed, r.greedy_eval, eff, r.wall_secs
        );
    }
    let steps_med = if n_solved > 0 {
        format!("{:.0}", median_f32(solved_steps))
    } else {
        "n/a".to_string()
    };
    println!(
        "  中位数 greedy={:.1} | total_env_steps={} ({}/{} 达标) | wall={:.1}s",
        median_f32(greedy),
        steps_med,
        n_solved,
        n,
        median_f32(walls),
    );
}

/// 物化空权重实例（冷启动推理前内部使用）。
pub(crate) fn materialize(
    py: Python<'_>,
    cfg: &MyZeroConfig,
    seed: u64,
) -> Result<MyZero, GraphError> {
    let env = GymEnv::new(py, cfg.env.env_id);
    let adapter = ActionAdapter::resolve(&env, cfg.env.action);
    let obs_dim = env.get_flatten_observation_len();
    let action_dim = adapter.action_dim();
    let graph = Graph::new_with_seed(seed);
    let model = MyZeroModel::new(&graph, obs_dim, action_dim, cfg.model.latent_dim)?;
    env.close();
    Ok(MyZero::from_parts(cfg.clone(), model, adapter))
}

/// 跑一次完整训练（固定 seed），返回训练后的 [`MyZero`] 与报告。
fn train_one_seed(
    py: Python<'_>,
    seed: u64,
    cfg: &MyZeroConfig,
    base_seed: u64,
) -> Result<(MyZero, TrainReport), GraphError> {
    let wall_t0 = std::time::Instant::now();

    let t = &cfg.train;
    let gamma = t.gamma;
    let smoke = cfg.eval.smoke;
    let solved = cfg.eval.solved;
    let reward_scale = cfg.env.reward_scale;

    // completedQ 改进策略目标的超参（开启 completed_q_target 时生效）
    let cq: Option<(f32, f32)> = if cfg.components.completed_q_target {
        Some((cfg.components.cq_c_visit, cfg.components.cq_c_scale))
    } else {
        None
    };

    let env = GymEnv::new(py, cfg.env.env_id);
    let adapter = ActionAdapter::resolve(&env, cfg.env.action);
    let obs_dim = env.get_flatten_observation_len();
    let action_dim = adapter.action_dim();
    let latent_dim = cfg.model.latent_dim;

    if seed == base_seed {
        println!(
            "[MyZero {}] obs={obs_dim} {} γ={gamma} sims={} r_scale={reward_scale}",
            cfg.env.env_id,
            adapter.describe(),
            t.num_simulations,
        );
    }

    let graph = Graph::new_with_seed(seed);
    let model = MyZeroModel::new(&graph, obs_dim, action_dim, latent_dim)?;
    let mut optimizer = Adam::new(&graph, &model.parameters(), t.lr);
    let mut buffer: ReplayBuffer<SelfPlayGame> = ReplayBuffer::new(t.buffer_capacity);
    let mut rng = StdRng::seed_from_u64(seed);

    let mut ep_rewards: VecDeque<f32> = VecDeque::with_capacity(100);
    let mut total_steps: u64 = 0;
    let mut hit_solved: Option<(usize, u64)> = None;
    let mut ckpt = BestTracker::new(
        &cfg.eval.checkpoint,
        seed,
        cfg.eval.seed_runs.max(1),
        obs_dim,
        smoke,
    );

    let max_episodes = if smoke { 3 } else { cfg.eval.max_episodes };
    let mut last_ep = 0usize;

    for ep in 0..max_episodes {
        last_ep = ep + 1;
        let t0 = std::time::Instant::now();

        // 温度退火：前 50% 局 t=1.0，后 50% 线性降到 0.25
        let progress = ep as f32 / max_episodes as f32;
        let temperature = if progress < 0.5 {
            1.0
        } else {
            1.0 - (progress - 0.5) * 2.0 * 0.75
        };

        let mcts_cfg = MctsConfig {
            num_simulations: t.num_simulations,
            temperature,
            discount: gamma,
            ..MctsConfig::default()
        };

        let steps = self_play_one_episode(
            &env,
            &model,
            &adapter,
            &mcts_cfg,
            gamma,
            cq,
            reward_scale,
            self_play_episode_seed(cfg.eval.seed, ep),
            &mut rng,
        );
        // 原始 return（反缩放回报告）
        let ep_reward: f32 = steps.iter().map(|s| s.reward).sum::<f32>() / reward_scale;
        let ep_len = steps.len();
        total_steps += ep_len as u64;

        buffer.push(SelfPlayGame {
            steps,
            outcome: GameOutcome::InProgress,
        });

        let mut avg_loss = 0.0;
        if buffer.len() >= t.start_training_after {
            let mut loss_sum = 0.0;
            let n_trains = if smoke { 1 } else { t.trains_per_episode };
            for _ in 0..n_trains {
                let mut games = buffer.sample(t.batch_games, &mut rng);
                if t.reanalyze_fraction > 0.0 {
                    let re_cfg = MctsConfig {
                        num_simulations: t.num_simulations,
                        temperature: 1.0,
                        discount: gamma,
                        root_exploration_fraction: 0.0,
                        ..MctsConfig::default()
                    };
                    let re_policy = PuctPolicy::new();
                    for g in games.iter_mut() {
                        if rng.gen_range(0.0..1.0) < t.reanalyze_fraction {
                            let dyn_model =
                                DynamicsModel::new(&model, adapter.candidates().to_vec(), gamma);
                            reanalyze_game(&dyn_model, &re_policy, g, &re_cfg, &mut rng);
                        }
                    }
                }
                let l = train_batch(
                    &model,
                    &mut optimizer,
                    &games,
                    t.k_unroll,
                    t.td_steps,
                    gamma,
                    &cfg.components,
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
            "Ep {:4}: R={:8.1} len={:3} avg_R={:8.1} loss={:.4} temp={:.2} total_env_steps={} t={:.2}s",
            ep + 1,
            ep_reward,
            ep_len,
            avg_r,
            avg_loss,
            temperature,
            total_steps,
            t0.elapsed().as_secs_f32()
        );

        if !smoke && ep_rewards.len() >= 20 && (ep + 1) % cfg.eval.eval_every == 0 {
            let eval_r = eval_episodes(
                &env,
                &model,
                &adapter,
                gamma,
                cfg.eval.eval_episodes,
                t.num_simulations,
                cfg.eval.seed,
            );
            let recent: f32 = ep_rewards.iter().rev().take(20).sum::<f32>() / 20.0;
            println!(
                "  greedy eval {eval_r:.1} / {solved}（近20局 self-play {recent:.1}，total_env_steps={total_steps}）",
            );
            ckpt.maybe_update(&model, cfg, eval_r, ep + 1, total_steps)?;
            if eval_r >= solved {
                hit_solved = Some((ep + 1, total_steps));
                println!("  ✅ 达标 ep={} total_env_steps={}", ep + 1, total_steps);
                break;
            }
        }
    }

    let wall_secs = wall_t0.elapsed().as_secs_f32();
    let final_greedy = if smoke {
        0.0
    } else {
        eval_episodes(
            &env,
            &model,
            &adapter,
            gamma,
            cfg.eval.eval_episodes,
            t.num_simulations,
            cfg.eval.seed,
        )
    };

    if !smoke {
        ckpt.save_last(&model, cfg, last_ep, total_steps, final_greedy)?;
    }

    if !smoke {
        let eff = hit_solved
            .map(|(e, s)| format!("ep{e} total_env_steps={s}"))
            .unwrap_or_else(|| "未达标".to_string());
        println!(
            "📈 {} greedy={final_greedy:.1} | 门槛 {solved}: {eff} | {wall_secs:.1}s",
            cfg.env.env_id,
        );
    }

    // Dynamics 诊断（DIAG=1）：对比 learned model 想象的 reward/value 与真实环境。
    if !smoke && cfg.eval.diagnose {
        dynamics_diagnostic(
            &env,
            &model,
            &adapter,
            gamma,
            reward_scale,
            t.num_simulations,
        );
    }

    env.close();

    let report = TrainReport {
        seed,
        wall_secs,
        final_greedy,
        solved_at_steps: hit_solved.map(|(_, s)| s),
        solved_threshold: solved,
        best_greedy: if ckpt.best_score().is_finite() {
            ckpt.best_score()
        } else {
            final_greedy
        },
        best_at_episode: ckpt.best_episode(),
        model_path: ckpt.model_path(),
    };
    let mz = MyZero::from_parts(cfg.clone(), model, adapter).with_train_report(report.clone());
    Ok((mz, report))
}

/// 多 seed 训练；返回**最后一 seed** 的权重实例。
pub(crate) fn train_all_seeds(mut cfg: MyZeroConfig) -> Result<MyZero, GraphError> {
    cfg.merge_from_env();
    print_components(&cfg.components);
    let smoke = cfg.eval.smoke;
    let n_runs = cfg.eval.seed_runs.max(1);
    let base_seed = cfg.eval.seed;

    Python::attach(|py| {
        let mut summaries = Vec::new();
        let mut last: Option<MyZero> = None;
        for i in 0..n_runs {
            let seed = base_seed.wrapping_add(i);
            if n_runs > 1 {
                println!("\n--- seed {seed} ({}/{n_runs}) ---", i + 1);
            }
            let (mz, report) = train_one_seed(py, seed, &cfg, base_seed)?;
            summaries.push(SeedSummary {
                seed: report.seed,
                wall_secs: report.wall_secs,
                greedy_eval: report.final_greedy,
                solved_at_steps: report.solved_at_steps,
            });
            last = Some(mz);
        }

        if n_runs > 1 && !smoke {
            print_multiseed_summary(&summaries, cfg.eval.solved);
        }

        if smoke {
            println!("[SMOKE] {} 通过", cfg.env.env_id);
        }
        last.ok_or_else(|| GraphError::InvalidOperation("MyZero: 无训练 seed".into()))
    })
}
