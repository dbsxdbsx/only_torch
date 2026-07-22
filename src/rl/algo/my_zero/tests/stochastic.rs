//! Stochastic MuZero 端到端测试（S9）

use crate::nn::{Adam, Graph, GraphError, Module, Optimizer};
use crate::rl::algo::my_zero::network::{MyZeroModel, ObsSource, UnrollItem};

/// K=1（默认）时 stochastic 路径不构建组件，训练行为与旧路径等价
#[test]
fn stochastic_off_no_behavior_change() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let model = MyZeroModel::new(&graph, 4, 3, 32)?.with_stochastic(1)?;
    assert!(!model.is_stochastic(), "K=1 应为确定性快路径");
    assert_eq!(model.num_chance_outcomes, 1);

    let item = UnrollItem {
        obs_t: vec![0.1, -0.2, 0.3, 0.4].into(),
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
    assert!(lv.is_finite(), "K=1 loss 应有限，got {lv}");

    // 核心网络（repr + dyn + pred）应有梯度；辅助网络（proj/recon/lstm）不参与此路径
    let core_params: Vec<_> = [
        model.repr.parameters(),
        model.dyn_net.parameters(),
        model.pred.parameters(),
    ]
    .concat();
    for (i, p) in core_params.iter().enumerate() {
        let grad = p.grad()?.unwrap_or_else(|| panic!("核心参数 {i} 应有梯度"));
        assert!(
            grad.to_vec().iter().all(|v| v.is_finite()),
            "核心参数 {i} 梯度应全部有限"
        );
    }
    opt.step()?;
    Ok(())
}

/// K=8 时应构建全部 4 个 stochastic 子网络，且参数数量多于 K=1 模型
#[test]
fn stochastic_on_constructs_components() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let model_k1 = MyZeroModel::new(&graph, 4, 3, 32)?.with_stochastic(1)?;
    let params_k1 = model_k1.parameters().len();

    let graph8 = Graph::new_with_seed(42);
    let model_k8 = MyZeroModel::new(&graph8, 4, 3, 32)?.with_stochastic(8)?;
    assert!(model_k8.is_stochastic(), "K=8 应为 stochastic 模式");
    assert_eq!(model_k8.num_chance_outcomes, 8);

    let params_k8 = model_k8.parameters().len();
    assert!(
        params_k8 > params_k1,
        "K=8 参数数量 ({params_k8}) 应 > K=1 ({params_k1})"
    );
    Ok(())
}

/// K=8 训练：next_obs 填充后 stochastic 附加 loss 正常反传，各组件有梯度
#[test]
fn stochastic_on_trains_and_has_gradients() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let model = MyZeroModel::new(&graph, 4, 3, 32)?.with_stochastic(8)?;

    let next_obs_data: Vec<ObsSource<'static>> = vec![vec![0.2, -0.1, 0.4, 0.3].into()];
    let item = UnrollItem {
        obs_t: vec![0.1, -0.2, 0.3, 0.4].into(),
        actions: vec![1],
        target_policies: vec![vec![0.3, 0.3, 0.4]; 2],
        target_values: vec![0.5, 0.2],
        target_rewards: vec![1.0],
        target_continuations: vec![1.0],
        next_obs: next_obs_data,
        bc_weights: Vec::new(),
        burn_in_obs: Vec::new(),
        burn_in_actions: Vec::new(),
        burn_in_leading_action: None,
        train_prev_action: None,
    };

    let all_params = model.parameters();
    let mut opt = Adam::new(&graph, &all_params, 0.01);
    opt.zero_grad()?;
    let loss = model.train_unroll_batch(&[item], 0.0, 0.0, 1.0, false, 0.0)?;
    let lv = loss.backward()?;
    assert!(lv.is_finite(), "K=8 loss 应有限，got {lv}");

    let mut has_grad_count = 0;
    for p in &all_params {
        if let Some(grad_data) = p.grad()? {
            assert!(
                grad_data.to_vec().iter().all(|v| v.is_finite()),
                "梯度含非有限值"
            );
            has_grad_count += 1;
        }
    }
    assert!(
        has_grad_count > 0,
        "至少部分参数应有梯度"
    );
    opt.step()?;
    Ok(())
}

/// afterstate + stochastic dynamics 推理链路：维度与数值契约
#[test]
fn afterstate_inference_chain() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(99);
    let k = 8;
    let latent_dim = 32;
    let action_dim = 3;
    let model = MyZeroModel::new(&graph, 4, action_dim, latent_dim)?.with_stochastic(k)?;

    let obs = [0.1f32, -0.2, 0.3, 0.4];
    let (latent, _policy, _value) =
        crate::rl::mcts::Dynamics::initial_state(&&model, &obs);
    assert_eq!(latent.len(), latent_dim, "latent 维度应为 {latent_dim}");

    // afterstate_dynamics_inference
    let (afterstate, chance_prior, as_value) = model.afterstate_dynamics_inference(&latent, 1);
    assert_eq!(afterstate.len(), latent_dim, "afterstate 维度应为 {latent_dim}");
    assert_eq!(chance_prior.len(), k, "chance_prior 长度应为 K={k}");
    let prior_sum: f32 = chance_prior.iter().sum();
    approx::assert_abs_diff_eq!(prior_sum, 1.0, epsilon = 1e-4);
    assert!(as_value.is_finite(), "afterstate value 应有限");
    assert!(
        chance_prior.iter().all(|p| *p >= 0.0),
        "chance_prior 各分量应 >= 0"
    );

    // stochastic_dynamics_inference
    let (next_latent, reward, policy, value, continuation) =
        model.stochastic_dynamics_inference(&afterstate, 3);
    assert_eq!(next_latent.len(), latent_dim, "next_latent 维度应为 {latent_dim}");
    assert!(reward.is_finite(), "reward 应有限");
    assert_eq!(policy.len(), action_dim, "policy 维度应为 {action_dim}");
    assert!(value.is_finite(), "value 应有限");
    assert!(
        (0.0..=1.0).contains(&continuation),
        "continuation 应在 [0,1]，got {continuation}"
    );
    let policy_sum: f32 = policy.iter().sum();
    approx::assert_abs_diff_eq!(policy_sum, 1.0, epsilon = 1e-4);

    Ok(())
}

/// Stochastic MCTS 搜索：K=3 mock model，验证树结构与推荐动作有效性
#[test]
fn mcts_chance_node_tree_structure() {
    use crate::rl::mcts::{
        ActionId, ActionPayload, CandidateSet, DecisionRecurrentOut, MctsConfig, MctsModel,
        PuctPolicy, RecurrentOut, RootOut, mcts_search,
    };
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const K: usize = 3;
    const NUM_ACTIONS: usize = 2;

    /// Mock：2 个离散动作 + 3 chance outcomes，单步就终止
    #[derive(Clone)]
    struct StochasticMock;

    impl MctsModel for StochasticMock {
        type State = u32;

        fn root(&self, _obs: &[f32]) -> RootOut<Self::State> {
            RootOut {
                state: 0,
                value: 0.5,
                candidates: CandidateSet::from_actions_and_priors(
                    (0..NUM_ACTIONS).map(ActionPayload::Discrete).collect(),
                    vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS],
                ),
                to_play: 0,
            }
        }

        fn recurrent(
            &self,
            _state: &Self::State,
            _action_id: ActionId,
            _action: &ActionPayload,
        ) -> RecurrentOut<Self::State> {
            unreachable!("stochastic 模式不应调用 recurrent")
        }

        fn num_chance_outcomes(&self) -> usize {
            K
        }

        fn decision_recurrent(
            &self,
            _state: &Self::State,
            _action_id: ActionId,
            _action: &ActionPayload,
        ) -> DecisionRecurrentOut<Self::State> {
            DecisionRecurrentOut {
                afterstate: 1,
                chance_prior: vec![1.0 / K as f32; K],
                afterstate_value: 0.3,
                to_play: 0,
            }
        }

        fn chance_recurrent(
            &self,
            _afterstate: &Self::State,
            _chance_id: usize,
        ) -> RecurrentOut<Self::State> {
            RecurrentOut {
                state: 2,
                reward: 1.0,
                value: 0.0,
                candidates: CandidateSet::from_actions_and_priors(
                    (0..NUM_ACTIONS).map(ActionPayload::Discrete).collect(),
                    vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS],
                ),
                terminal: false,
                to_play: 0,
                discount: 0.99,
            }
        }
    }

    let model = StochasticMock;
    let policy = PuctPolicy::new();
    let cfg = MctsConfig {
        num_simulations: 30,
        temperature: 1.0,
        num_chance_outcomes: K,
        ..MctsConfig::default()
    };
    let mut rng = StdRng::seed_from_u64(42);
    let result = mcts_search(&model, &policy, &[0.0], &cfg, &mut rng);

    assert_eq!(
        result.children.len(),
        NUM_ACTIONS,
        "根节点应有 {NUM_ACTIONS} 个子节点"
    );
    let total_visits: u32 = result.children.iter().map(|c| c.visit_count).sum();
    assert!(
        total_visits >= cfg.num_simulations,
        "总访问 {total_visits} 应 >= {}", cfg.num_simulations
    );
    assert!(
        matches!(result.recommended, ActionPayload::Discrete(_)),
        "推荐动作应为离散类型"
    );
    let policy_sum: f32 = result.learn_policy.iter().sum();
    approx::assert_abs_diff_eq!(policy_sum, 1.0, epsilon = 1e-4);
    assert!(
        result.network_value.is_finite(),
        "network_value 应有限"
    );
}
