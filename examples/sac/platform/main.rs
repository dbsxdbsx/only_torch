//! Platform-v0 Hybrid SAC 示例
//!
//! ```bash
//! pip install hybrid-platform
//! cargo run --example platform_sac
//! ```

mod model;

use model::SacAgent;
use only_torch::nn::{
    Adam, Graph, GraphError, Module, Optimizer, Var, VarLossOps, VarReduceOps, VarShapeOps,
};
use only_torch::rl::algo::sac::{
    compute_td_target, compute_v_hybrid, transitions_to_batch, update_alpha,
};
use only_torch::rl::{GymEnv, ReplayBuffer, Transition};
use only_torch::tensor::Tensor;
use pyo3::Python;
use std::collections::VecDeque;

fn main() -> Result<(), GraphError> {
    let (actor_lr, critic_lr, gamma, tau, hidden, batch_sz) = (3e-4, 1e-3, 0.99, 0.005, 128, 128);

    Python::attach(|py| {
        let env = GymEnv::new(py, "Platform-v0");
        let obs_dim = env.get_flatten_observation_len();

        let graph = Graph::new_with_seed(42);
        let mut agent = SacAgent::new(&graph, obs_dim, hidden)?;
        agent.target_critic1.soft_update_from(&agent.critic1, 1.0);
        agent.target_critic2.soft_update_from(&agent.critic2, 1.0);

        let mut a_opt = Adam::new(&graph, &agent.actor.parameters(), actor_lr);
        let mut c1_opt = Adam::new(&graph, &agent.critic1.parameters(), critic_lr);
        let mut c2_opt = Adam::new(&graph, &agent.critic2.parameters(), critic_lr);
        let mut buffer = ReplayBuffer::new(100_000);
        let mut rng = rand::thread_rng();
        let mut ep_rewards: VecDeque<f32> = VecDeque::with_capacity(50);
        let mut steps = 0usize;

        for ep in 0..2000 {
            let mut obs = env.flatten_obs(&env.reset(None));
            let mut ep_r = 0.0f32;

            loop {
                let (_, act_vec) = agent
                    .actor
                    .select_action(&Tensor::new(&obs, &[1, obs_dim]))?;
                let (nobs, reward, terminated, truncated) = env.step(&act_vec);
                let next_obs = env.flatten_obs(&nobs);
                ep_r += reward;
                steps += 1;

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

                    // 分离 action 编码：[discrete_idx, c0, c1, c2]
                    let bs = batch.len();
                    let disc_idx: Vec<f32> = batch.iter().map(|t| t.action[0]).collect();
                    let disc_t = Tensor::new(&disc_idx, &[bs, 1]);
                    let cont_data: Vec<f32> = batch
                        .iter()
                        .flat_map(|t| t.action[1..4].iter().copied())
                        .collect();
                    let cont_t = Tensor::new(&cont_data, &[bs, 3]);

                    // Target V
                    let (np, nlp) = agent.actor.get_discrete_probs(&b.next_obs)?;
                    let (nc, nclp) = agent.actor.sample_cont(&b.next_obs)?;
                    let tq1 = agent.target_critic1.get_q_values(&b.next_obs, &nc)?;
                    let tq2 = agent.target_critic2.get_q_values(&b.next_obs, &nc)?;
                    let clp_sum = nclp.sum_axis_keepdims(1);
                    let v_next = compute_v_hybrid(
                        &np,
                        &tq1.minimum(&tq2),
                        &nlp,
                        agent.alpha_d(),
                        &clp_sum,
                        agent.alpha_c(),
                    );
                    let target = compute_td_target(&b.rewards, &v_next, &b.not_terminated, gamma);

                    // Critic 更新
                    let ov1 = graph.input_named(&b.obs, "o")?;
                    let cv1 = graph.input_named(&cont_t, "c")?;
                    let c1_loss = agent
                        .critic1
                        .forward_q(&ov1, &cv1)?
                        .gather(1, &disc_t)?
                        .mse_loss(&target)?;
                    c1_opt.zero_grad()?;
                    c1_loss.backward()?;
                    c1_opt.step()?;
                    let ov2 = graph.input_named(&b.obs, "o")?;
                    let cv2 = graph.input_named(&cont_t, "c")?;
                    let c2_loss = agent
                        .critic2
                        .forward_q(&ov2, &cv2)?
                        .gather(1, &disc_t)?
                        .mse_loss(&target)?;
                    c2_opt.zero_grad()?;
                    c2_loss.backward()?;
                    c2_opt.step()?;

                    // Actor 更新
                    let ov_a = graph.input_named(&b.obs, "o")?;
                    let (probs, lp_var, cont_act, cont_lp) = agent.actor.forward_train(&ov_a)?;
                    let q1a = agent.critic1.forward_q(&ov_a, &cont_act)?;
                    let q2a = agent.critic2.forward_q(&ov_a, &cont_act)?;
                    let q_min = Var::minimum(&q1a, &q2a)?;
                    let wq = (&probs * &q_min).sum_axis(1);
                    let d_ent = (&probs * &lp_var).sum_axis(1);
                    let c_lp_sum = cont_lp.sum_axis(1);
                    let a_loss =
                        (&(&d_ent * agent.alpha_d()) + &(&c_lp_sum * agent.alpha_c()) - &wq).mean();
                    a_opt.zero_grad()?;
                    a_loss.backward()?;
                    a_opt.step()?;

                    // Alpha 更新
                    let avg_h_d = -(&np * &nlp).sum_axis_keepdims(1).mean()[[0, 0]];
                    agent.log_alpha_d = update_alpha(
                        agent.log_alpha_d,
                        agent.alpha_lr,
                        avg_h_d,
                        agent.target_entropy_d,
                    );
                    let avg_clp = nclp.mean()[[0, 0]];
                    agent.log_alpha_c = update_alpha(
                        agent.log_alpha_c,
                        agent.alpha_lr,
                        -avg_clp,
                        agent.target_entropy_c,
                    );

                    agent.soft_update_targets(tau);
                }

                if terminated || truncated {
                    break;
                }
                obs = next_obs;
            }

            ep_rewards.push_back(ep_r);
            if ep_rewards.len() > 50 {
                ep_rewards.pop_front();
            }
            if (ep + 1) % 50 == 0 {
                let avg = ep_rewards.iter().sum::<f32>() / ep_rewards.len() as f32;
                println!(
                    "Ep {:>4} | steps {:>6} | R={:.2} | avg50={:.2} | αd={:.4} αc={:.4}",
                    ep + 1,
                    steps,
                    ep_r,
                    avg,
                    agent.alpha_d(),
                    agent.alpha_c()
                );
                if avg >= 0.0 {
                    println!("[达标]");
                    break;
                }
            }
        }

        env.close();
        Ok(())
    })
}
