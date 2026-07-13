//! Phase 3A0：逐转移 world-model 误差诊断契约。

use super::super::component::Components;
use super::super::model_error::{
    TransitionDiagnostics, categorical_kl, categorical_nll, jensen_shannon_divergence,
    score_transition, spearman_rank_correlation, top_fraction_event_lift,
};
use super::super::model_error_audit::{
    audit_gomoku_proxy, collect_gomoku_audit_block, score_gomoku_audit_block,
};
use super::super::network::{MyZeroModel, UnrollItem};
use crate::nn::{Adam, Graph, Optimizer};
use crate::rl::GymEnv;
use crate::rl::mcts::{
    ActionId, ActionPayload, DynamicsModel, MctsConfig, PuctPolicy, mcts_search,
};
use approx::assert_abs_diff_eq;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serial_test::serial;

fn raw_diagnostics() -> TransitionDiagnostics {
    TransitionDiagnostics {
        reward_probs: vec![0.1, 0.8, 0.1],
        continuation: 0.8,
        imagined_next_policy: vec![0.7, 0.2, 0.1],
        imagined_next_value: 0.4,
        reencoded_next_policy: Some(vec![0.6, 0.3, 0.1]),
        reencoded_next_value: Some(0.1),
    }
}

#[test]
fn categorical_nll_and_brier_follow_proper_loss_definitions() {
    let raw = raw_diagnostics();
    let score = score_transition(&raw, &[0.0, 1.0, 0.0], 1.0, None);
    assert_abs_diff_eq!(score.reward_kl, -0.8_f32.ln(), epsilon = 1e-6);
    assert_abs_diff_eq!(score.continuation_brier, 0.04, epsilon = 1e-6);
    assert!(score.is_finite_nonnegative());

    assert_abs_diff_eq!(
        categorical_nll(&[0.25, 0.75], &[0.4, 0.6]),
        -(0.25 * 0.4_f32.ln() + 0.75 * 0.6_f32.ln()),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        categorical_kl(&[0.25, 0.75], &[0.25, 0.75]),
        0.0,
        epsilon = 1e-7
    );
    assert!(
        categorical_kl(&[f32::NAN, 0.0], &[0.5, 0.5]).is_nan(),
        "非有限输入必须暴露给上层有限性检查，不能静默截成 0"
    );
}

#[test]
fn jsd_is_symmetric_bounded_and_respects_shared_mask() {
    let p = [0.9, 0.1, 0.0];
    let q = [0.1, 0.9, 0.0];
    let pq = jensen_shannon_divergence(&p, &q, None).unwrap();
    let qp = jensen_shannon_divergence(&q, &p, None).unwrap();
    assert_abs_diff_eq!(pq, qp, epsilon = 1e-6);
    assert!(pq > 0.0 && pq <= std::f32::consts::LN_2);

    let same = jensen_shannon_divergence(&p, &p, None).unwrap();
    assert_abs_diff_eq!(same, 0.0, epsilon = 1e-7);

    // 差异只在被 mask 的动作 0；共同合法支撑 {1,2} 上应完全一致。
    let masked = jensen_shannon_divergence(
        &[0.8, 0.1, 0.1],
        &[0.2, 0.4, 0.4],
        Some(&[false, true, true]),
    )
    .unwrap();
    assert_abs_diff_eq!(masked, 0.0, epsilon = 1e-7);
    assert!(
        jensen_shannon_divergence(&p, &q, Some(&[false, false, false])).is_none(),
        "空合法支撑必须按缺失分量处理"
    );
}

#[test]
fn missing_real_next_excludes_decision_equivalence_components() {
    let mut raw = raw_diagnostics();
    raw.reencoded_next_policy = None;
    raw.reencoded_next_value = None;
    let score = score_transition(&raw, &[0.0, 1.0, 0.0], 0.0, None);
    assert!(score.policy_jsd.is_none());
    assert!(score.value_abs_diff.is_none());
    assert!(score.is_finite_nonnegative());
}

#[test]
fn model_transition_diagnostics_are_finite_and_rng_free() {
    let graph = Graph::new_with_seed(42);
    let model = MyZeroModel::new(&graph, 4, 3, 8).unwrap();
    let obs = [0.1, -0.2, 0.3, 0.4];
    let next_obs = [0.2, -0.1, 0.35, 0.45];
    let mask = [true, false, true];

    let first = model.transition_diagnostics(&obs, ActionId(2), Some(&next_obs));
    let second = model.transition_diagnostics(&obs, ActionId(2), Some(&next_obs));
    assert_eq!(first, second, "诊断前向不得消耗 RNG 或改变模型状态");
    assert_eq!(first.reward_probs.len(), 41);
    assert_abs_diff_eq!(first.reward_probs.iter().sum::<f32>(), 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(
        first.imagined_next_policy.iter().sum::<f32>(),
        1.0,
        epsilon = 1e-5
    );

    let score = model.transition_error_components(
        &obs,
        ActionId(2),
        1.0,
        1.0,
        Some(&next_obs),
        Some(&mask),
    );
    assert!(score.is_finite_nonnegative(), "score={score:?}");
    assert!(score.policy_jsd.is_some());
    assert!(score.value_abs_diff.is_some());
}

fn training_item() -> UnrollItem<'static> {
    UnrollItem {
        obs_t: vec![0.1, -0.2, 0.3, 0.4].into(),
        actions: vec![2],
        target_policies: vec![vec![0.2, 0.3, 0.5], vec![0.1, 0.2, 0.7]],
        target_values: vec![0.4, -0.2],
        target_rewards: vec![1.0],
        target_continuations: vec![1.0],
        next_obs: Vec::new(),
        bc_weights: Vec::new(),
    }
}

#[test]
fn diagnostics_do_not_change_mcts_or_next_training_update() {
    let graph_a = Graph::new_with_seed(123);
    let graph_b = Graph::new_with_seed(123);
    let model_a = MyZeroModel::new(&graph_a, 4, 3, 8).unwrap();
    let model_b = MyZeroModel::new(&graph_b, 4, 3, 8).unwrap();
    let obs = [0.1, -0.2, 0.3, 0.4];
    let actions = (0..3).map(ActionPayload::Discrete).collect();
    let mcts_model = DynamicsModel::new(&model_a, actions, 0.99);
    let cfg = MctsConfig {
        num_simulations: 16,
        temperature: 0.0,
        root_exploration_fraction: 0.0,
        ..MctsConfig::default()
    };
    let policy = PuctPolicy::new();
    let mut before_rng = StdRng::seed_from_u64(7);
    let before = mcts_search(&mcts_model, &policy, &obs, &cfg, &mut before_rng);

    let _ = model_a.transition_diagnostics(&obs, ActionId(2), Some(&[0.2, -0.1, 0.4, 0.5]));

    let mut after_rng = StdRng::seed_from_u64(7);
    let after = mcts_search(&mcts_model, &policy, &obs, &cfg, &mut after_rng);
    assert_eq!(before.recommended_id, after.recommended_id);
    assert_eq!(before.learn_policy, after.learn_policy);
    assert_eq!(before.children.len(), after.children.len());
    for (before, after) in before.children.iter().zip(&after.children) {
        assert_eq!(before.action_id, after.action_id);
        assert_eq!(before.visit_count, after.visit_count);
        assert_eq!(before.value_sum.to_bits(), after.value_sum.to_bits());
        assert_eq!(before.reward.to_bits(), after.reward.to_bits());
    }

    let mut optimizer_a = Adam::new(&graph_a, &model_a.parameters(), 1e-3);
    let mut optimizer_b = Adam::new(&graph_b, &model_b.parameters(), 1e-3);
    for (model, optimizer) in [(&model_a, &mut optimizer_a), (&model_b, &mut optimizer_b)] {
        let loss = model
            .train_unroll_batch(&[training_item()], 0.0, 0.0, 1.0, false, 0.0)
            .unwrap();
        optimizer.zero_grad().unwrap();
        loss.backward().unwrap();
        optimizer.step().unwrap();
    }
    for (a, b) in model_a.parameters().iter().zip(model_b.parameters()) {
        let a = a.value().unwrap().unwrap().to_vec();
        let b = b.value().unwrap().unwrap().to_vec();
        assert_eq!(
            a.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            "插入 diagnostics 不得改变下一次训练更新"
        );
    }
}

#[test]
fn audit_statistics_handle_ranks_ties_and_sparse_events() {
    assert_abs_diff_eq!(
        spearman_rank_correlation(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]).unwrap(),
        1.0,
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        spearman_rank_correlation(&[1.0, 1.0, 2.0, 2.0], &[4.0, 3.0, 2.0, 1.0]).unwrap(),
        -0.894_427_2,
        epsilon = 1e-6
    );
    assert!(
        spearman_rank_correlation(&[1.0, 1.0], &[2.0, 3.0]).is_none(),
        "常量 rank 没有可定义相关性"
    );

    let scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let events = [
        false, false, false, false, false, false, false, false, true, true,
    ];
    assert_abs_diff_eq!(
        top_fraction_event_lift(&scores, &events, 0.2).unwrap(),
        5.0,
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        top_fraction_event_lift(&[0.0, 1.0, 1.0, 1.0], &[false, true, false, true], 0.25,).unwrap(),
        4.0 / 3.0,
        epsilon = 1e-6
    );
    assert!(top_fraction_event_lift(&scores, &[false; 10], 0.2).is_none());
}

#[test]
#[serial]
fn gomoku_proxy_audit_runs_from_real_reset_without_training() {
    pyo3::Python::attach(|py| {
        let env = GymEnv::new(py, "Gomoku-selfplay-v0");
        env.reset(Some(7));
        let action_dim = env.legal_mask().len();
        let obs_dim = env.board_observation_flat().len();
        let graph = Graph::new_with_seed(7);
        let model = MyZeroModel::new(&graph, obs_dim, action_dim, 16).unwrap();
        let block = collect_gomoku_audit_block(&env, &model, &Components::base(), 4, 1, 1000);
        let other_block = collect_gomoku_audit_block(&env, &model, &Components::base(), 4, 1, 2000);
        let report = score_gomoku_audit_block(&model, &block);
        let direct_report = audit_gomoku_proxy(&env, &model, &Components::base(), 4, 1, 1000);
        env.close();

        assert!(!report.records.is_empty());
        assert_ne!(
            block
                .iter()
                .map(|transition| transition.action)
                .collect::<Vec<_>>(),
            other_block
                .iter()
                .map(|transition| transition.action)
                .collect::<Vec<_>>(),
            "不同 temporal block 必须通过 self-play 噪声产生不同真实轨迹"
        );
        assert_eq!(report.records.len(), direct_report.records.len());
        assert_eq!(report.components.len(), 4);
        assert!(report.tactical_positions <= report.records.len());
        assert!(
            report
                .records
                .iter()
                .all(|record| record.error.is_finite_nonnegative())
        );
        assert!(report.components.iter().all(|component| {
            !component.name.is_empty()
                && component.count > 0
                && component.mean.is_finite()
                && component
                    .spearman_to_reference_policy
                    .is_none_or(f32::is_finite)
                && component
                    .tactical_position_top_decile_lift
                    .is_none_or(f32::is_finite)
        }));
        for (first, second) in report.components.iter().zip(&direct_report.components) {
            assert_eq!(first.name, second.name);
            assert_abs_diff_eq!(first.mean, second.mean, epsilon = 1e-7);
        }
    });
}
