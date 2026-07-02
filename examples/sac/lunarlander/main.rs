//! LunarLander-v3 SAC-Discrete（`pip install "gymnasium[box2d]"`）
//!
//! ```bash
//! cargo run --example lunarlander_sac --release
//! SMOKE=1 cargo run --example lunarlander_sac  # 管线验证（3 ep 截短，不验收敛）
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
    let max_ep = if smoke { 3 } else { 1000 };
    let (start_after, batch_sz) = if smoke { (64, 32) } else { (1000, 64) };
    Python::attach(|py| {
        let env = GymEnv::new(py, "LunarLander-v3");
        let (obs_dim, act_dim, gamma) = (env.get_flatten_observation_len(), 4, 0.99);
        let graph = Graph::new_with_seed(42);
        let mut ag = SacAgent::new(&graph, obs_dim, act_dim)?;
        ag.target_critic1.hard_update_from(&ag.critic1);
        ag.target_critic2.hard_update_from(&ag.critic2);
        let mut a_opt = Adam::new(&graph, &ag.actor.parameters(), 3e-4);
        let mut c1_opt = Adam::new(&graph, &ag.critic1.parameters(), 3e-4);
        let mut c2_opt = Adam::new(&graph, &ag.critic2.parameters(), 3e-4);
        let mut buf = ReplayBuffer::new(100_000);
        let mut rng = rand::thread_rng();
        let mut hist: VecDeque<f32> = VecDeque::with_capacity(100);

        for ep in 0..max_ep {
            let mut obs = env.reset(None)[0].clone();
            let mut ep_r = 0.0f32;
            loop {
                let (action, _) = ag.actor.sample_action(&Tensor::new(&obs, &[1, obs_dim]))?;
                let (nobs, r, term, trunc) = env.step(&[action as f32]);
                let nxt = nobs[0].clone();
                ep_r += r;
                buf.push(Transition {
                    obs: obs.clone(),
                    action: vec![action as f32],
                    reward: r,
                    next_obs: nxt.clone(),
                    terminated: term,
                    truncated: trunc,
                });
                if buf.len() >= start_after {
                    let batch = buf.sample(batch_sz, &mut rng);
                    let b = transitions_to_batch(&batch, obs_dim);
                    let (np, nlp) = ag.actor.get_action_probs(&b.next_obs)?;
                    let tq = ag
                        .target_critic1
                        .get_q_values(&b.next_obs)?
                        .minimum(&ag.target_critic2.get_q_values(&b.next_obs)?);
                    let tgt = compute_td_target(
                        &b.rewards,
                        &compute_v_discrete(&np, &tq, &nlp, ag.alpha()),
                        &b.not_terminated,
                        gamma,
                    );
                    let q1 = ag.critic1.forward(&graph.input_named(&b.obs, "o")?)?;
                    c1_opt.zero_grad()?;
                    q1.gather(1, &b.actions)?.mse_loss(&tgt)?.backward()?;
                    c1_opt.step()?;
                    let q2 = ag.critic2.forward(&graph.input_named(&b.obs, "o")?)?;
                    c2_opt.zero_grad()?;
                    q2.gather(1, &b.actions)?.mse_loss(&tgt)?.backward()?;
                    c2_opt.step()?;
                    let qm = ag
                        .critic1
                        .get_q_values(&b.obs)?
                        .minimum(&ag.critic2.get_q_values(&b.obs)?);
                    let logits = ag.actor.forward(&graph.input_named(&b.obs, "o")?)?;
                    let d = Categorical::new(logits);
                    let (p, lp) = (d.probs(), d.log_probs());
                    let al = (&p * (&lp * Tensor::new(&[ag.alpha()], &[1, 1]) - &qm))
                        .sum_axis(1)
                        .mean();
                    al.forward()?;
                    let (pv, lpv) = (p.value()?.unwrap(), lp.value()?.unwrap());
                    let h = -(&pv * &lpv).sum().get_data_number().unwrap() / pv.shape()[0] as f32;
                    a_opt.zero_grad()?;
                    al.backward()?;
                    a_opt.step()?;
                    ag.log_alpha = update_alpha(ag.log_alpha, ag.alpha_lr, h, ag.target_entropy);
                    ag.soft_update_targets();

                    if smoke {
                        let v = al.value()?.unwrap()[[0, 0]];
                        assert!(v.is_finite(), "SMOKE: actor_loss={v} 非有限");
                    }
                }
                if term || trunc {
                    break;
                }
                obs = nxt;
            }
            hist.push_back(ep_r);
            if hist.len() > 100 {
                hist.pop_front();
            }
            let avg = hist.iter().sum::<f32>() / hist.len() as f32;
            if (ep + 1) % 10 == 0 {
                println!(
                    "Ep {:4}: R={:6.1} avg={:6.1} α={:.3}",
                    ep + 1,
                    ep_r,
                    avg,
                    ag.alpha()
                );
            }
            if !smoke && avg >= 200.0 {
                println!("✅ avg={avg:.1}");
                break;
            }
        }
        if smoke {
            println!("[SMOKE] 通过");
        }
        env.close();
        Ok(())
    })
}
