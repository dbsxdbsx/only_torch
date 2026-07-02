//! CartPole SAC-Discrete 示例（CartPole-v1，500 制）
//!
//! ```bash
//! cargo run --example cartpole_sac --release
//! SMOKE=1 cargo run --example cartpole_sac  # 管线验证（3 ep，不验收敛）
//! SEED=43 cargo run --example cartpole_sac --release  # 多 seed 基线重测（默认 42）
//! ```
//!
//! 达标：greedy(temp=0) eval 20 局（固定 seed）均值 ≥ 475（Gymnasium CartPole-v1 官方 solved）。
//! 全项目 CartPole 统一用 v1，不再使用 v0。

mod model;

use model::SacAgent;
use only_torch::nn::distributions::Categorical;
use only_torch::nn::{
    Adam, Graph, GraphError, Module, Optimizer, VarLossOps, VarReduceOps, VarShapeOps,
};
use only_torch::rl::algo::sac::{
    compute_td_target, compute_v_discrete, transitions_to_batch, update_alpha,
};
use only_torch::rl::{GymEnv, ReplayBuffer, Transition};
use only_torch::tensor::Tensor;
use pyo3::Python;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::VecDeque;

fn eval_cartpole(
    env: &GymEnv,
    agent: &model::SacAgent,
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
            let logits = agent.actor.forward(&Tensor::new(&obs, &[1, obs_dim]))?;
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
    // SEED：多 seed 基线重测协议（默认 42；权重初始化 / buffer 采样 / 首局 reset 均派生）。
    let seed: u64 = std::env::var("SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let max_ep = if smoke {
        3
    } else {
        std::env::var("MAX_EP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2000)
    };
    let (start_after, batch_size) = if smoke { (32, 32) } else { (1000, 64) };
    let (lr, gamma) = (0.001, 0.99);

    Python::attach(|py| {
        let env = GymEnv::new(py, "CartPole-v1");
        let obs_dim = env.get_flatten_observation_len();
        let action_dim = 2;

        let graph = Graph::new_with_seed(seed);
        let mut agent = SacAgent::new(&graph, obs_dim, action_dim)?;
        agent.target_critic1.hard_update_from(&agent.critic1);
        agent.target_critic2.hard_update_from(&agent.critic2);

        let mut actor_opt = Adam::new(&graph, &agent.actor.parameters(), lr);
        let mut c1_opt = Adam::new(&graph, &agent.critic1.parameters(), lr);
        let mut c2_opt = Adam::new(&graph, &agent.critic2.parameters(), lr);
        let mut buffer = ReplayBuffer::new(100_000);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut ep_rewards: VecDeque<f32> = VecDeque::with_capacity(100);
        // 样本效率：累计真实 env-step + 首次达标的 (episode, env_step)
        let mut total_steps: u64 = 0;
        let mut hit_solved: Option<(usize, u64)> = None;
        for ep in 0..max_ep {
            let t0 = std::time::Instant::now();
            let reset_seed = if ep == 0 { Some(seed) } else { None };
            let mut obs = env.reset(reset_seed)[0].clone();
            let (mut ep_r, mut ep_len) = (0.0f32, 0);

            loop {
                let (action, _) = agent
                    .actor
                    .sample_action(&Tensor::new(&obs, &[1, obs_dim]))?;
                let (nobs, reward, terminated, truncated) = env.step(&[action as f32]);
                let next_obs = nobs[0].clone();
                ep_r += reward;
                ep_len += 1;

                buffer.push(Transition {
                    obs: obs.clone(),
                    action: vec![action as f32],
                    reward,
                    next_obs: next_obs.clone(),
                    terminated,
                    truncated,
                });

                if buffer.len() >= start_after {
                    let batch = buffer.sample(batch_size, &mut rng);
                    let b = transitions_to_batch(&batch, obs_dim);

                    // Target V
                    let (np, nlp) = agent.actor.get_action_probs(&b.next_obs)?;
                    let tq1 = agent.target_critic1.get_q_values(&b.next_obs)?;
                    let tq2 = agent.target_critic2.get_q_values(&b.next_obs)?;
                    let v_next = compute_v_discrete(&np, &tq1.minimum(&tq2), &nlp, agent.alpha());
                    let target = compute_td_target(&b.rewards, &v_next, &b.not_terminated, gamma);

                    // Critic 更新
                    let q1 = agent.critic1.forward(&graph.input_named(&b.obs, "o")?)?;
                    let c1_loss = q1.gather(1, &b.actions)?.mse_loss(&target)?;
                    c1_opt.zero_grad()?;
                    c1_loss.backward()?;
                    c1_opt.step()?;
                    let q2 = agent.critic2.forward(&graph.input_named(&b.obs, "o")?)?;
                    let c2_loss = q2.gather(1, &b.actions)?.mse_loss(&target)?;
                    c2_opt.zero_grad()?;
                    c2_loss.backward()?;
                    c2_opt.step()?;

                    // Actor 更新
                    let q_min = agent
                        .critic1
                        .get_q_values(&b.obs)?
                        .minimum(&agent.critic2.get_q_values(&b.obs)?);
                    let logits = agent.actor.forward(&graph.input_named(&b.obs, "o")?)?;
                    let dist = Categorical::new(logits);
                    let (probs, log_probs) = (dist.probs(), dist.log_probs());
                    let alpha_t = Tensor::new(&[agent.alpha()], &[1, 1]);
                    let a_loss = (&probs * (&log_probs * alpha_t - &q_min))
                        .sum_axis(1)
                        .mean();
                    a_loss.forward()?;
                    let pv = probs.value()?.unwrap();
                    let lpv = log_probs.value()?.unwrap();
                    let avg_h =
                        -(&pv * &lpv).sum().get_data_number().unwrap() / pv.shape()[0] as f32;
                    actor_opt.zero_grad()?;
                    a_loss.backward()?;
                    actor_opt.step()?;

                    if smoke {
                        for (name, l) in [("c1", &c1_loss), ("c2", &c2_loss), ("a", &a_loss)] {
                            let v = l.value()?.unwrap()[[0, 0]];
                            assert!(v.is_finite(), "SMOKE: {name}_loss={v} 非有限");
                        }
                    }

                    agent.log_alpha =
                        update_alpha(agent.log_alpha, agent.alpha_lr, avg_h, agent.target_entropy);
                    agent.soft_update_targets();
                }

                if terminated || truncated {
                    break;
                }
                obs = next_obs;
            }

            total_steps += ep_len as u64;
            ep_rewards.push_back(ep_r);
            if ep_rewards.len() > 20 {
                ep_rewards.pop_front();
            }
            let avg = ep_rewards.iter().sum::<f32>() / ep_rewards.len() as f32;
            println!(
                "Ep {:3}: R={:5.1} len={:3} avg20={:5.1} α={:.3} t={:.2}s",
                ep + 1,
                ep_r,
                ep_len,
                avg,
                agent.alpha(),
                t0.elapsed().as_secs_f32()
            );
            // 每 25 局跑 greedy eval（固定 seed）；达标即记录样本量并停。
            if !smoke && (ep + 1) % 25 == 0 {
                let (eval_mean, _) = eval_cartpole(&env, &agent, obs_dim, 20, Some(12345))?;
                println!(
                    "  📊 greedy eval@ep{}: mean={:.1}（env_steps={}）",
                    ep + 1,
                    eval_mean,
                    total_steps
                );
                if eval_mean >= solved {
                    hit_solved = Some((ep + 1, total_steps));
                    println!(
                        "✅ CartPole-v1 达标！greedy eval={:.1} ≥ {solved}（ep={} env_steps={}）",
                        eval_mean,
                        ep + 1,
                        total_steps
                    );
                    break;
                }
            }
        }

        // eval：固定 seed 下 deterministic 跑 20 局
        if !smoke {
            let eval_env = GymEnv::new(py, "CartPole-v1");
            let (eval_mean, eval_max) = eval_cartpole(&eval_env, &agent, obs_dim, 20, Some(12345))?;
            println!("📊 Eval 20 局: mean={eval_mean:.1} max={eval_max:.0}");
            if eval_mean >= solved {
                println!("✅ Eval 达标 mean={eval_mean:.1} ≥ {solved}");
            } else {
                println!("⚠️ Eval 未达标 mean={eval_mean:.1} < {solved}");
            }
            eval_env.close();
            let eff = hit_solved
                .map(|(e, s)| format!("ep{e} / {s} env-steps"))
                .unwrap_or_else(|| "未达到".to_string());
            println!("📈 [样本效率] CartPole-v1 SAC 到 {solved}: {eff}");
        }

        if smoke {
            println!("[SMOKE] 通过");
        }
        env.close();
        Ok(())
    })
}
