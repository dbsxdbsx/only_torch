//! CartPole PPO 示例（CartPole-v1，500 制）
//!
//! ```bash
//! cargo run --example ppo_cartpole --release
//! SMOKE=1 cargo run --example ppo_cartpole        # 管线验证
//! SEED=43 cargo run --example ppo_cartpole --release  # 多 seed 基线重测（默认 42）
//! ```
//!
//! 达标：greedy(temp=0) eval 20 局（固定 seed）均值 ≥ 475（Gymnasium CartPole-v1 官方 solved）。
//! 全项目 CartPole 统一用 v1，不再使用 v0。

mod model;

use model::{PpoActor, PpoCritic};
use only_torch::nn::{Adam, Graph, GraphError, Module, Optimizer};
use only_torch::rl::algo::ppo::{
    clipped_policy_loss, compute_gae, entropy_bonus, normalize_advantages, value_loss,
};
use only_torch::rl::{GymEnv, RolloutBuffer, RolloutStep};
use only_torch::tensor::Tensor;
use pyo3::Python;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::collections::VecDeque;

fn eval_cartpole(
    env: &GymEnv,
    actor: &PpoActor,
    obs_dim: usize,
    n_episodes: usize,
    seed_base: Option<u64>,
) -> Result<(f32, f32), GraphError> {
    let mut total = 0.0f32;
    let mut max_r = f32::NEG_INFINITY;
    for i in 0..n_episodes {
        let mut obs = env.reset(seed_base.map(|b| b + i as u64))[0].clone();
        let mut ep_r = 0.0f32;
        loop {
            let logits = actor.forward(&Tensor::new(&obs, &[1, obs_dim]))?;
            let logits_val = logits.value()?.unwrap();
            let probs = logits_val.softmax(1);
            let action = probs
                .data_as_slice()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let (nobs, reward, terminated, truncated) = env.step(&[action as f32]);
            ep_r += reward;
            if terminated || truncated {
                break;
            }
            obs = nobs[0].clone();
        }
        total += ep_r;
        if ep_r > max_r {
            max_r = ep_r;
        }
    }
    Ok((total / n_episodes as f32, max_r))
}

fn main() -> Result<(), GraphError> {
    let smoke = std::env::var("SMOKE").is_ok();
    // CartPole-v1 达标门槛（Gymnasium 官方 solved = greedy eval 均值 ≥ 475）。
    let solved = 475.0_f32;
    // SEED：多 seed 基线重测协议（默认 42；权重初始化 / rollout shuffle / 首局 reset 均派生）。
    let seed: u64 = std::env::var("SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    let n_steps: usize = if smoke { 64 } else { 2048 };
    let max_updates: usize = if smoke {
        2
    } else {
        std::env::var("MAX_UPD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(600)
    };
    let (lr, gamma, lambda) = (3e-4, 0.99, 0.95);
    let (clip_eps, ent_coef, vf_coef) = (0.2, 0.01, 0.5);
    let ppo_epochs = 4;
    let minibatch_size: usize = 64;

    Python::attach(|py| {
        let env = GymEnv::new(py, "CartPole-v1");
        let obs_dim = env.get_flatten_observation_len();
        let action_dim = 2;

        let graph = Graph::new_with_seed(seed);
        let actor = PpoActor::new(&graph, obs_dim, action_dim)?;
        let critic = PpoCritic::new(&graph, obs_dim)?;

        let mut actor_opt = Adam::new(&graph, &actor.parameters(), lr);
        let mut critic_opt = Adam::new(&graph, &critic.parameters(), lr);

        let mut ep_rewards: VecDeque<f32> = VecDeque::with_capacity(100);
        let mut obs = env.reset(Some(seed))[0].clone();
        let mut ep_r = 0.0f32;
        // 样本效率：PPO rollout 连续跨 update，用独立 eval env 以免打断 obs 连续性。
        let mut total_steps: u64 = 0;
        let mut total_episodes: u64 = 0;
        let mut hit_solved: Option<(u64, u64)> = None;
        let eval_env = GymEnv::new(py, "CartPole-v1");

        for update in 0..max_updates {
            let t0 = std::time::Instant::now();
            let mut buffer = RolloutBuffer::new(n_steps);

            // 采集 rollout（每步存真实 next_value，解决 truncated 边界 bootstrap 问题）
            let mut next_values_buf: Vec<f32> = Vec::with_capacity(n_steps);
            while !buffer.is_full() {
                let obs_t = Tensor::new(&obs, &[1, obs_dim]);
                let (action, log_prob) = actor.sample_action(&obs_t)?;
                let value = critic.get_value(&obs_t)?;

                let (nobs, reward, terminated, truncated) = env.step(&[action as f32]);
                ep_r += reward;

                // 关键：truncated 时用被截断状态的后继 V，terminated 时为 0
                let next_v = if terminated {
                    0.0
                } else {
                    critic.get_value(&Tensor::new(&nobs[0], &[1, obs_dim]))?
                };
                next_values_buf.push(next_v);

                buffer.push(RolloutStep {
                    obs: obs.clone(),
                    action: vec![action as f32],
                    log_prob,
                    value,
                    reward,
                    terminated,
                    truncated,
                });

                if terminated || truncated {
                    ep_rewards.push_back(ep_r);
                    if ep_rewards.len() > 100 {
                        ep_rewards.pop_front();
                    }
                    total_episodes += 1;
                    ep_r = 0.0;
                    obs = env.reset(None)[0].clone();
                } else {
                    obs = nobs[0].clone();
                }
            }

            let last_value = critic.get_value(&Tensor::new(&obs, &[1, obs_dim]))?;

            let steps = buffer.steps();
            let rewards: Vec<f32> = steps.iter().map(|s| s.reward).collect();
            let values: Vec<f32> = steps.iter().map(|s| s.value).collect();
            let term: Vec<bool> = steps.iter().map(|s| s.terminated).collect();
            let trunc: Vec<bool> = steps.iter().map(|s| s.truncated).collect();

            let (mut advantages, returns) = compute_gae(
                &rewards,
                &values,
                &term,
                &trunc,
                &next_values_buf,
                last_value,
                gamma,
                lambda,
            );
            normalize_advantages(&mut advantages);

            // 准备原始数据用于 shuffle
            let raw_obs: Vec<&[f32]> = steps.iter().map(|s| s.obs.as_slice()).collect();
            let raw_act: Vec<&[f32]> = steps.iter().map(|s| s.action.as_slice()).collect();
            let raw_olp: Vec<f32> = steps.iter().map(|s| s.log_prob).collect();
            let action_dim = steps[0].action.len();
            let n = steps.len();

            // PPO 更新（多 epoch，每 epoch shuffle 索引）
            let mut rng = StdRng::seed_from_u64(seed + update as u64);
            for _epoch in 0..ppo_epochs {
                let mut indices: Vec<usize> = (0..n).collect();
                indices.shuffle(&mut rng);

                for mb_start in (0..n).step_by(minibatch_size) {
                    let mb_end = (mb_start + minibatch_size).min(n);
                    let mb_idx = &indices[mb_start..mb_end];
                    let bs = mb_idx.len();

                    let mb_obs_data: Vec<f32> = mb_idx
                        .iter()
                        .flat_map(|&i| raw_obs[i].iter().copied())
                        .collect();
                    let mb_act_data: Vec<f32> = mb_idx
                        .iter()
                        .flat_map(|&i| raw_act[i].iter().copied())
                        .collect();
                    let mb_olp_data: Vec<f32> = mb_idx.iter().map(|&i| raw_olp[i]).collect();
                    let mb_adv_data: Vec<f32> = mb_idx.iter().map(|&i| advantages[i]).collect();
                    let mb_ret_data: Vec<f32> = mb_idx.iter().map(|&i| returns[i]).collect();

                    let mb_obs = Tensor::new(&mb_obs_data, &[bs, obs_dim]);
                    let mb_actions = Tensor::new(&mb_act_data, &[bs, action_dim]);

                    let obs_var = graph.input_named(&mb_obs, "o")?;
                    let (new_log_probs, dist_entropy) =
                        actor.evaluate_actions(&obs_var, &mb_actions)?;
                    let old_lp_var =
                        graph.input_named(&Tensor::new(&mb_olp_data, &[bs, 1]), "old_lp")?;
                    let adv_var = graph.input_named(&Tensor::new(&mb_adv_data, &[bs, 1]), "adv")?;
                    let ret_var = graph.input_named(&Tensor::new(&mb_ret_data, &[bs, 1]), "ret")?;

                    let pi_loss =
                        clipped_policy_loss(&new_log_probs, &old_lp_var, &adv_var, clip_eps);
                    let ent_loss = entropy_bonus(&dist_entropy);
                    let actor_loss = &pi_loss + &(&ent_loss * ent_coef);
                    actor_loss.forward()?;

                    if smoke {
                        let v = actor_loss.value()?.unwrap()[[0, 0]];
                        assert!(v.is_finite(), "SMOKE: actor_loss={v} 非有限");
                    }

                    actor_opt.zero_grad()?;
                    actor_loss.backward()?;
                    actor_opt.step()?;

                    let new_values = critic.forward(&graph.input_named(&mb_obs, "o")?)?;
                    let v_loss = value_loss(&new_values, &ret_var);
                    let critic_loss = &v_loss * vf_coef;
                    critic_loss.forward()?;

                    critic_opt.zero_grad()?;
                    critic_loss.backward()?;
                    critic_opt.step()?;
                }
            }

            let avg = if ep_rewards.is_empty() {
                0.0
            } else {
                ep_rewards.iter().sum::<f32>() / ep_rewards.len() as f32
            };
            total_steps += n_steps as u64;
            println!(
                "Update {:3}: avg={:5.1} episodes={} env_steps={} t={:.2}s",
                update + 1,
                avg,
                ep_rewards.len(),
                total_steps,
                t0.elapsed().as_secs_f32()
            );

            // 每 10 个 update 跑 greedy eval（固定 seed、独立 env）；达标即记录样本量并停。
            if !smoke && (update + 1) % 10 == 0 {
                let (eval_mean, _) = eval_cartpole(&eval_env, &actor, obs_dim, 20, Some(12345))?;
                println!(
                    "  📊 greedy eval@upd{}: mean={:.1}（env_steps={} episodes={}）",
                    update + 1,
                    eval_mean,
                    total_steps,
                    total_episodes
                );
                if eval_mean >= solved {
                    hit_solved = Some((total_episodes, total_steps));
                    println!(
                        "✅ CartPole-v1 达标！greedy eval={eval_mean:.1} ≥ {solved}（episodes={total_episodes} env_steps={total_steps}）"
                    );
                    break;
                }
            }
        }

        if !smoke {
            let (eval_mean, eval_max) = eval_cartpole(&eval_env, &actor, obs_dim, 20, Some(12345))?;
            println!("📊 Eval 20 局: mean={eval_mean:.1} max={eval_max:.0}");
            if eval_mean >= solved {
                println!("✅ Eval 达标 mean={eval_mean:.1} ≥ {solved}");
            } else {
                println!("⚠️ Eval 未达标 mean={eval_mean:.1} < {solved}");
            }
            let eff = hit_solved
                .map(|(e, s)| format!("episodes{e} / {s} env-steps"))
                .unwrap_or_else(|| "未达到".to_string());
            println!("📈 [样本效率] CartPole-v1 PPO 到 {solved}: {eff}");
        }
        eval_env.close();

        if smoke {
            println!("[SMOKE] 通过");
        }
        env.close();
        Ok(())
    })
}
