//! Pendulum SAC-Continuous 示例
//!
//! ```bash
//! cargo run --example pendulum_sac
//! ```

mod model;

use model::SacAgent;
use only_torch::nn::{
    Adam, Graph, GraphError, Module, Optimizer, VarActivationOps, VarLossOps, VarReduceOps,
};
use only_torch::rl::algo::sac::{
    compute_td_target, compute_v_continuous, transitions_to_batch, update_alpha,
};
use only_torch::rl::{GymEnv, ReplayBuffer, Transition};
use only_torch::tensor::Tensor;
use pyo3::Python;
use std::collections::VecDeque;

fn main() -> Result<(), GraphError> {
    let (actor_lr, critic_lr, gamma, hidden, batch_sz) = (3e-4, 1e-3, 0.99, 32, 256);

    Python::attach(|py| {
        let env = GymEnv::new(py, "Pendulum-v1");
        let obs_dim = env.get_flatten_observation_len();
        let ranges = env.get_all_action_valid_range();
        let action_dim = ranges.len();
        let (lo, hi) = ranges[0].get_continuous_action_low_high();

        let graph = Graph::new_with_seed(42);
        let mut agent = SacAgent::new(&graph, obs_dim, action_dim, hidden, lo, hi)?;
        agent.target_critic1.hard_update_from(&agent.critic1);
        agent.target_critic2.hard_update_from(&agent.critic2);

        let mut a_opt = Adam::new(&graph, &agent.actor.parameters(), actor_lr);
        let mut c1_opt = Adam::new(&graph, &agent.critic1.parameters(), critic_lr);
        let mut c2_opt = Adam::new(&graph, &agent.critic2.parameters(), critic_lr);
        let mut buffer = ReplayBuffer::new(100_000);
        let mut rng = rand::thread_rng();
        let mut ep_rewards: VecDeque<f32> = VecDeque::with_capacity(100);
        for ep in 0..300 {
            let t0 = std::time::Instant::now();
            let mut obs = env.reset(None)[0].clone();
            let (mut ep_r, mut ep_len) = (0.0f32, 0);

            loop {
                let (tanh_act, _) = agent
                    .actor
                    .sample_action(&Tensor::new(&obs, &[1, obs_dim]))?;
                let env_act = agent.scale_action(&tanh_act);
                let act_vec: Vec<f32> = (0..action_dim).map(|i| env_act[[0, i]]).collect();
                let (nobs, reward, terminated, truncated) = env.step(&act_vec);
                let next_obs = nobs[0].clone();
                ep_r += reward;
                ep_len += 1;

                buffer.push(Transition {
                    obs: obs.clone(),
                    action: act_vec,
                    reward,
                    next_obs: next_obs.clone(),
                    terminated,
                    truncated,
                });

                if buffer.len() >= 500 {
                    let batch = buffer.sample(batch_sz, &mut rng);
                    let b = transitions_to_batch(&batch, obs_dim);
                    let act_norm = agent.unscale_action(&b.actions);

                    // Target V
                    let (nta, nlp) = agent.actor.sample_action(&b.next_obs)?;
                    let nlp_sum = nlp.sum_axis_keepdims(1);
                    let tq1 = agent.target_critic1.get_q_value(&b.next_obs, &nta)?;
                    let tq2 = agent.target_critic2.get_q_value(&b.next_obs, &nta)?;
                    let v_next = compute_v_continuous(&tq1.minimum(&tq2), &nlp_sum, agent.alpha());
                    let target = compute_td_target(&b.rewards, &v_next, &b.not_terminated, gamma);

                    // Critic 更新
                    let ov1 = graph.input_named(&b.obs, "o")?;
                    let av1 = graph.input_named(&act_norm, "a")?;
                    let c1_loss = agent.critic1.forward_q(&ov1, &av1)?.mse_loss(&target)?;
                    c1_opt.zero_grad()?;
                    c1_loss.backward()?;
                    c1_opt.step()?;
                    let ov2 = graph.input_named(&b.obs, "o")?;
                    let av2 = graph.input_named(&act_norm, "a")?;
                    let c2_loss = agent.critic2.forward_q(&ov2, &av2)?.mse_loss(&target)?;
                    c2_opt.zero_grad()?;
                    c2_loss.backward()?;
                    c2_opt.step()?;

                    // Actor 更新
                    let ov_a = graph.input_named(&b.obs, "o")?;
                    let (act_var, lp_var) = agent.actor.sample_for_update(&ov_a)?;
                    let lp_sum = lp_var.sum_axis(1);
                    let q1a = agent.critic1.forward_q(&ov_a, &act_var)?;
                    let q2a = agent.critic2.forward_q(&ov_a, &act_var)?;
                    let q_diff = &q1a - &q2a;
                    let half = Tensor::new(&[0.5], &[1, 1]);
                    let q_min = (&q1a + &q2a - q_diff.abs()) * half;
                    let alpha_t = Tensor::new(&[agent.alpha()], &[1, 1]);
                    let a_loss = (&lp_sum * alpha_t - &q_min).mean();
                    a_loss.forward()?;
                    let avg_h = -(lp_sum.value()?.unwrap().sum().get_data_number().unwrap())
                        / batch.len() as f32;
                    a_opt.zero_grad()?;
                    a_loss.backward()?;
                    a_opt.step()?;

                    agent.log_alpha =
                        update_alpha(agent.log_alpha, agent.alpha_lr, avg_h, agent.target_entropy);
                    agent.soft_update_targets();
                }

                if terminated || truncated {
                    break;
                }
                obs = next_obs;
            }

            ep_rewards.push_back(ep_r);
            if ep_rewards.len() > 100 {
                ep_rewards.pop_front();
            }
            let avg = ep_rewards.iter().sum::<f32>() / ep_rewards.len() as f32;
            println!(
                "Ep {:3}: R={:7.1} len={:3} avg={:7.1} α={:.4} t={:.2}s",
                ep + 1,
                ep_r,
                ep_len,
                avg,
                agent.alpha(),
                t0.elapsed().as_secs_f32()
            );
            if ep_r >= -300.0 {
                println!("✅ 达标 R={ep_r:.1}");
                break;
            }
        }

        env.close();
        Ok(())
    })
}
