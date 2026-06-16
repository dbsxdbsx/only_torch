//! MuZero CartPole-v0 训练示例
//!
//! ```bash
//! cargo run --example muzero_cartpole --release
//! SMOKE=1 cargo run --example muzero_cartpole  # 管线验证（3 局 self-play + 1 次训练）
//! ```

mod model;

use model::MuZeroModel;
use only_torch::nn::{Adam, Graph, GraphError, Optimizer};
use only_torch::rl::algo::muzero::{MuZeroConfig, compute_n_step_target, reanalyze_game};
use only_torch::rl::mcts::{ActionPayload, DynamicsModel, MctsConfig, PuctPolicy, mcts_search};
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

        // root value：visit 加权的子节点 Q（与 reanalyze 共用 SearchResult::root_value 口径）
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
        // 仅记录 MDP 真终止（杆倒）；truncation（撞 200 步）保持 false 以便 n-step 末端 bootstrap
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
    model: &MuZeroModel,
    optimizer: &mut Adam,
    games: &[SelfPlayGame],
    k_unroll: usize,
    td_steps: usize,
    gamma: f32,
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
        // canonical absorbing state：
        // - 终止局（杆倒）：full-K unroll，越过终局的位置用 absorbing 目标（reward0/value0/uniform）
        //   补齐，让模型学到「终局后回报恒 0」。
        // - 截断局（撞步数上限、杆未倒）：短 unroll、不补 absorbing（否则会低估满分局末端），
        //   value 目标由 compute_n_step_target 的 truncation bootstrap 兜底。
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
                    uniform_policy.clone() // absorbing → uniform
                }
            })
            .collect();

        let target_values: Vec<f32> = (0..=actual_k)
            .map(|i| {
                if t + i < len {
                    compute_n_step_target(steps, t + i, td_steps, gamma)
                } else {
                    0.0 // absorbing → value 0
                }
            })
            .collect();

        let target_rewards: Vec<f32> = (0..actual_k)
            .map(|i| {
                if t + i < len {
                    steps[t + i].reward
                } else {
                    0.0 // absorbing → reward 0
                }
            })
            .collect();

        let actions: Vec<usize> = (0..actual_k)
            .map(|i| {
                if t + i < len {
                    steps[t + i].action[0] as usize
                } else {
                    0 // absorbing → 占位动作（target reward/value 恒 0，与动作无关）
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

    // eval 用独立种子 rng，使训练可复现不受 eval 调用次数影响（greedy 下 rng 不影响输出）
    let mut eval_rng = StdRng::seed_from_u64(0xE7A1);
    let mut total_reward = 0.0;
    for _ in 0..n_episodes {
        let obs_raw = env.reset(None);
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

    // 网络拓扑（环境相关，留示例）
    let latent_dim = 64;
    let action_dim = 2;
    let obs_dim = 4;

    // 算法超参：库级 MuZeroConfig 容器，按环境配置（CartPole 用 CPU 友好默认）。
    // 换环境时在此覆盖字段，尤其 num_simulations（棋类自对弈应显著调高）。
    let mut cfg = MuZeroConfig::default();
    // reanalyze 默认关闭（CPU 上较贵）；用 `REANALYZE=<比例>` 开启 demo（如 0.5）。
    if let Ok(v) = std::env::var("REANALYZE") {
        if let Ok(f) = v.parse::<f32>() {
            cfg.reanalyze_fraction = f.clamp(0.0, 1.0);
        }
    }
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
    } = cfg;

    let max_episodes = if smoke {
        3
    } else {
        std::env::var("MAX_EP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000)
    };

    Python::attach(|py| {
        let env = GymEnv::new(py, "CartPole-v0");
        let graph = Graph::new_with_seed(42);
        let model = MuZeroModel::new(&graph, obs_dim, action_dim, latent_dim)?;
        let mut optimizer = Adam::new(&graph, &model.parameters(), lr);
        let mut buffer: ReplayBuffer<SelfPlayGame> = ReplayBuffer::new(buffer_capacity);
        let mut rng = StdRng::seed_from_u64(42);

        let actions: Vec<ActionPayload> = (0..action_dim).map(ActionPayload::Discrete).collect();

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

            let steps = self_play_one_episode(&env, &model, &actions, &mcts_cfg, gamma, &mut rng);
            let ep_reward: f32 = steps.iter().map(|s| s.reward).sum();
            let ep_len = steps.len();

            // 诊断：观察 root_value 分布与 MCTS policy target 是否差异化
            if std::env::var("DBG").is_ok() && ep % 20 == 0 {
                let rvs: Vec<f32> = steps.iter().filter_map(|s| s.root_value).collect();
                let rv_min = rvs.iter().cloned().fold(f32::INFINITY, f32::min);
                let rv_max = rvs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let p0 = steps
                    .first()
                    .map(|s| s.policy_target.clone())
                    .unwrap_or_default();
                let pmid = steps
                    .get(ep_len / 2)
                    .map(|s| s.policy_target.clone())
                    .unwrap_or_default();
                println!(
                    "  [dbg] ep{ep} root_value∈[{rv_min:.2},{rv_max:.2}] π0={p0:?} πmid={pmid:?}"
                );
            }

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
                    // reanalyze（可选，CPU 贵）：用最新模型对部分采样局重跑 MCTS，
                    // 刷新 policy/value 目标（生成干净目标，关 Dirichlet 噪声）。
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
                "Ep {:4}: R={:6.1} len={:3} avg_R={:6.1} loss={:.4} temp={:.2} t={:.2}s",
                ep + 1,
                ep_reward,
                ep_len,
                avg_r,
                avg_loss,
                temperature,
                t0.elapsed().as_secs_f32()
            );

            // 达标判定：CartPole「solved」的真实度量是**贪心(temp=0) eval 均值 ≥195**。
            // self-play 带温度采样会系统性压低其均值，故不以 self-play 均值当门槛，
            // 而是在 self-play 近 20 局均值达到合理预阈值后，**定期跑贪心 eval 确认**。
            // 真实门槛仍是 greedy eval ≥195（比 Gym 的训练均值口径更严）。
            if !smoke && ep_rewards.len() >= 20 && (ep + 1) % 25 == 0 {
                let recent: f32 = ep_rewards.iter().rev().take(20).sum::<f32>() / 20.0;
                if recent >= 170.0 {
                    let eval_r = eval_episodes(&env, &model, &actions, gamma, 20, num_simulations);
                    println!("  贪心 eval 20 局均值={eval_r:.1}（self-play 近20均值={recent:.1}）");
                    if eval_r >= 195.0 {
                        println!("MuZero CartPole-v0 达标！贪心 eval 20 局均值={eval_r:.1} ≥ 195");
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
