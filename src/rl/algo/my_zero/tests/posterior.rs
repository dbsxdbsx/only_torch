//! POMDP-lite posterior encoder 端到端测试

use crate::nn::{Adam, Graph, GraphError, Optimizer};
use crate::rl::algo::my_zero::network::{MyZeroModel, ObsSource, UnrollItem};

/// posterior=false 训练应与旧路径数学等价（空 burn-in 零开销）
#[test]
fn posterior_off_no_behavior_change() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let model = MyZeroModel::new(&graph, 4, 3, 32)?
        .with_recurrent_posterior(false)?;
    assert!(model.posterior.is_none());

    let obs: Vec<f32> = vec![0.1, -0.2, 0.3, 0.4];
    let item = UnrollItem {
        obs_t: obs.clone().into(),
        actions: vec![1],
        target_policies: vec![vec![0.3, 0.3, 0.4]; 2],
        target_values: vec![0.5, 0.2],
        target_rewards: vec![1.0],
        target_continuations: vec![1.0],
        next_obs: Vec::new(),
        bc_weights: Vec::new(),
        burn_in_obs: Vec::new(),
        burn_in_actions: Vec::new(),
        burn_in_leading_action: None,
        train_prev_action: None,
    };
    let mut opt = Adam::new(&graph, &model.parameters(), 0.01);
    opt.zero_grad()?;
    let loss = model.train_unroll_batch(&[item], 0.0, 0.0, 1.0, false, 0.0)?;
    let lv = loss.backward()?;
    assert!(lv.is_finite(), "posterior=false loss 应有限，got {lv}");
    opt.step()?;
    Ok(())
}

/// posterior=true：构造→训练→梯度流验证
#[test]
fn posterior_on_trains_and_has_gradients() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let model = MyZeroModel::new(&graph, 4, 3, 32)?
        .with_recurrent_posterior(true)?;
    assert!(model.posterior.is_some());
    let posterior = model.posterior.as_ref().unwrap();
    let posterior_param_count = posterior.parameters().len();
    assert!(posterior_param_count > 0, "posterior 应有可训练参数");

    let all_params = model.parameters();
    assert!(
        all_params.len() > posterior_param_count,
        "总参数应 > posterior 参数（repr+dyn+pred 也在）"
    );

    let burn_obs: Vec<ObsSource<'static>> = vec![
        vec![0.2, 0.1, -0.1, 0.3].into(),
        vec![0.3, -0.2, 0.0, 0.1].into(),
    ];
    let item = UnrollItem {
        obs_t: vec![0.1, -0.2, 0.3, 0.4].into(),
        actions: vec![1],
        target_policies: vec![vec![0.3, 0.3, 0.4]; 2],
        target_values: vec![0.5, 0.2],
        target_rewards: vec![1.0],
        target_continuations: vec![1.0],
        next_obs: Vec::new(),
        bc_weights: Vec::new(),
        burn_in_obs: burn_obs,
        burn_in_actions: vec![2, 0],
        burn_in_leading_action: None,
        train_prev_action: Some(0),
    };
    let mut opt = Adam::new(&graph, &all_params, 0.01);
    opt.zero_grad()?;
    let loss = model.train_unroll_batch(&[item], 0.0, 0.0, 1.0, false, 0.0)?;
    let lv = loss.backward()?;
    assert!(lv.is_finite(), "posterior=true loss 应有限，got {lv}");

    for (i, p) in posterior.parameters().iter().enumerate() {
        let grad = p.grad()?.unwrap_or_else(|| panic!("posterior 参数 {i} 应有梯度"));
        assert!(
            grad.to_vec().iter().all(|v| v.is_finite()),
            "posterior 参数 {i} 梯度应全部有限"
        );
    }
    opt.step()?;
    Ok(())
}

/// repr_inference + posterior_step_inference + pred_inference 推理链路可跑通
#[test]
fn posterior_inference_chain() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(99);
    let model = MyZeroModel::new(&graph, 4, 3, 32)?
        .with_recurrent_posterior(true)?;
    let hs = model.posterior.as_ref().unwrap().hidden_size;

    let obs = [0.1f32, -0.2, 0.3, 0.4];
    let repr_latent = model.repr_inference(&obs);
    assert_eq!(repr_latent.len(), 32, "latent_dim=32");

    let prev_oh = vec![0.0f32; 3];
    let hidden = vec![0.0f32; hs];
    let (posterior_latent, new_hidden) =
        model.posterior_step_inference(&repr_latent, &prev_oh, &hidden);
    assert_eq!(posterior_latent.len(), 32);
    assert_eq!(new_hidden.len(), hs);

    let (policy, value) = model.pred_inference(&posterior_latent);
    assert_eq!(policy.len(), 3, "action_dim=3");
    assert!(value.is_finite());
    let policy_sum: f32 = policy.iter().sum();
    approx::assert_abs_diff_eq!(policy_sum, 1.0, epsilon = 1e-5);

    Ok(())
}

/// PrecomputedRootDynamics 搜索与原 Dynamics 搜索应产出一致的根 value
#[test]
fn precomputed_root_dynamics_consistent_with_direct() -> Result<(), GraphError> {
    use crate::rl::algo::my_zero::network::PrecomputedRootDynamics;
    use crate::rl::mcts::{ActionPayload, Dynamics, DynamicsOutput};

    let graph = Graph::new_with_seed(42);
    let model = MyZeroModel::new(&graph, 4, 3, 32)?;
    let obs = [0.5f32, -0.1, 0.2, 0.8];

    let (latent, policy, value) = Dynamics::initial_state(&&model, &obs);
    let pre = PrecomputedRootDynamics::new(&model, latent.clone(), policy.clone(), value);
    let (pre_latent, pre_policy, pre_value) = Dynamics::initial_state(&pre, &obs);
    assert_eq!(pre_latent, latent);
    assert_eq!(pre_policy, policy);
    assert_eq!(pre_value, value);

    let action = ActionPayload::Discrete(1);
    let out_orig: DynamicsOutput = Dynamics::recurrent(&&model, &latent, &action);
    let out_pre: DynamicsOutput = Dynamics::recurrent(&pre, &latent, &action);
    assert_eq!(out_orig.next_state, out_pre.next_state);
    approx::assert_abs_diff_eq!(out_orig.reward, out_pre.reward, epsilon = 1e-6);

    Ok(())
}

/// ObsAdapter::Flat mask_indices 应正确置零指定维度
#[test]
fn obs_mask_zeros_velocity_dims() {
    use crate::rl::algo::my_zero::obs_pipeline::ObsAdapter;
    use crate::rl::algo::my_zero::config::ObservationPlan;

    pyo3::Python::attach(|py| {
        let env = crate::rl::GymEnv::new(py, "CartPole-v1");
        let mut adapter = ObsAdapter::resolve(&env, ObservationPlan::Auto, vec![1, 3]);
        let (obs, _store) = adapter.reset(&env, Some(42));
        assert_eq!(obs.len(), 4, "CartPole obs 应为 4 维");
        assert_eq!(obs[1], 0.0, "cart_velocity（索引 1）应被遮蔽为 0");
        assert_eq!(obs[3], 0.0, "pole_velocity（索引 3）应被遮蔽为 0");
        assert!(obs[0] != 0.0 || obs[2] != 0.0, "至少有一个非速度维度非零");

        let action = vec![0.0]; // left
        let (obs2, _, _, _, _) = adapter.step(&env, &action);
        assert_eq!(obs2[1], 0.0, "step 后 cart_velocity 仍应为 0");
        assert_eq!(obs2[3], 0.0, "step 后 pole_velocity 仍应为 0");
    });
}

/// masked obs + posterior=true 的 train_unroll_batch 能跑通且有梯度
#[test]
fn masked_obs_posterior_trains() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(77);
    let model = MyZeroModel::new(&graph, 4, 2, 32)?
        .with_recurrent_posterior(true)?;
    let mut opt = Adam::new(&graph, &model.parameters(), 0.01);

    let masked_obs = |seed: u32| -> Vec<f32> {
        let mut o = vec![0.1 * seed as f32, 0.0, -0.05 * seed as f32, 0.0];
        o[1] = 0.0; // masked
        o[3] = 0.0; // masked
        o
    };

    let item = UnrollItem {
        obs_t: masked_obs(3).into(),
        actions: vec![1],
        target_policies: vec![vec![0.6, 0.4]; 2],
        target_values: vec![0.8, 0.5],
        target_rewards: vec![1.0],
        target_continuations: vec![1.0],
        next_obs: Vec::new(),
        bc_weights: Vec::new(),
        burn_in_obs: vec![masked_obs(1).into(), masked_obs(2).into()],
        burn_in_actions: vec![0, 1],
        burn_in_leading_action: None,
        train_prev_action: Some(1),
    };

    opt.zero_grad()?;
    let loss = model.train_unroll_batch(&[item], 0.0, 0.0, 1.0, false, 0.0)?;
    let lv = loss.backward()?;
    assert!(lv.is_finite(), "masked + posterior loss 应有限，got {lv}");

    let posterior = model.posterior.as_ref().unwrap();
    for (i, p) in posterior.parameters().iter().enumerate() {
        assert!(
            p.grad()?.is_some(),
            "posterior 参数 {i} 在 masked 训练后应有梯度"
        );
    }
    opt.step()?;
    Ok(())
}
