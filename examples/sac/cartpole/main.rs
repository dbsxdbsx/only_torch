//! CartPole SAC-Discrete 示例
//!
//! ```bash
//! cargo run --example cartpole_sac
//! SMOKE=1 cargo run --example cartpole_sac  # 管线验证（3 ep，不验收敛）
//! ```

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
use std::collections::VecDeque;

fn main() -> Result<(), GraphError> {
    let smoke = std::env::var("SMOKE").is_ok();
    let (max_ep, start_after, batch_size) = if smoke { (3, 32, 32) } else { (500, 1000, 64) };
    let (lr, gamma) = (0.001, 0.99);

    Python::attach(|py| {
        let env = GymEnv::new(py, "CartPole-v0");
        let obs_dim = env.get_flatten_observation_len();
        let action_dim = 2;

        let graph = Graph::new_with_seed(42);
        let mut agent = SacAgent::new(&graph, obs_dim, action_dim)?;
        agent.target_critic1.hard_update_from(&agent.critic1);
        agent.target_critic2.hard_update_from(&agent.critic2);

        let mut actor_opt = Adam::new(&graph, &agent.actor.parameters(), lr);
        let mut c1_opt = Adam::new(&graph, &agent.critic1.parameters(), lr);
        let mut c2_opt = Adam::new(&graph, &agent.critic2.parameters(), lr);
        let mut buffer = ReplayBuffer::new(100_000);
        let mut rng = rand::thread_rng();
        let mut ep_rewards: VecDeque<f32> = VecDeque::with_capacity(100);
        for ep in 0..max_ep {
            let t0 = std::time::Instant::now();
            let mut obs = env.reset(None)[0].clone();
            let (mut ep_r, mut ep_len) = (0.0f32, 0);

            loop {
                let (action, _) = agent.actor.sample_action(&Tensor::new(&obs, &[1, obs_dim]))?;
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
                    c1_opt.zero_grad()?; c1_loss.backward()?; c1_opt.step()?;
                    let q2 = agent.critic2.forward(&graph.input_named(&b.obs, "o")?)?;
                    let c2_loss = q2.gather(1, &b.actions)?.mse_loss(&target)?;
                    c2_opt.zero_grad()?; c2_loss.backward()?; c2_opt.step()?;

                    // Actor 更新
                    let q_min = agent.critic1.get_q_values(&b.obs)?
                        .minimum(&agent.critic2.get_q_values(&b.obs)?);
                    let logits = agent.actor.forward(&graph.input_named(&b.obs, "o")?)?;
                    let dist = Categorical::new(logits);
                    let (probs, log_probs) = (dist.probs(), dist.log_probs());
                    let alpha_t = Tensor::new(&[agent.alpha()], &[1, 1]);
                    let a_loss = (&probs * (&log_probs * alpha_t - &q_min)).sum_axis(1).mean();
                    a_loss.forward()?;
                    let pv = probs.value()?.unwrap();
                    let lpv = log_probs.value()?.unwrap();
                    let avg_h = -(&pv * &lpv).sum().get_data_number().unwrap() / pv.shape()[0] as f32;
                    actor_opt.zero_grad()?; a_loss.backward()?; actor_opt.step()?;

                    if smoke {
                        for (name, l) in [("c1", &c1_loss), ("c2", &c2_loss), ("a", &a_loss)] {
                            let v = l.value()?.unwrap()[[0, 0]];
                            assert!(v.is_finite(), "SMOKE: {name}_loss={v} 非有限");
                        }
                    }

                    agent.log_alpha = update_alpha(agent.log_alpha, agent.alpha_lr, avg_h, agent.target_entropy);
                    agent.soft_update_targets();
                }

                if terminated || truncated { break; }
                obs = next_obs;
            }

            ep_rewards.push_back(ep_r);
            if ep_rewards.len() > 100 { ep_rewards.pop_front(); }
            let avg = ep_rewards.iter().sum::<f32>() / ep_rewards.len() as f32;
            println!("Ep {:3}: R={:5.1} len={:3} avg={:5.1} α={:.3} t={:.2}s",
                ep+1, ep_r, ep_len, avg, agent.alpha(), t0.elapsed().as_secs_f32());
            if ep_r >= 195.0 { println!("✅ 达标 R={ep_r:.0}"); break; }
        }

        if smoke { println!("[SMOKE] 通过"); }
        env.close();
        Ok(())
    })
}
