//! MCTS 类型契约测试：ActionId 与 payload 分离、MctsConfig recipe 往返

use crate::rl::mcts::{ActionCandidate, ActionId, ActionPayload, CandidateSet, MctsConfig};

#[test]
fn candidate_set_keeps_action_id_separate_from_payload() {
    let candidates = CandidateSet {
        candidates: vec![ActionCandidate::new(
            ActionId(7),
            ActionPayload::Continuous(vec![0.25]),
            0.8,
        )],
    };
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates.candidates[0].id, ActionId(7));
    assert_eq!(candidates.policy_priors(), vec![0.8]);
    assert!(matches!(
        candidates.candidates[0].payload,
        ActionPayload::Continuous(_)
    ));
}

#[test]
fn mcts_recipe_roundtrip_preserves_legacy_config() {
    let cfg = MctsConfig {
        num_simulations: 32,
        pb_c_base: 100.0,
        pb_c_init: 2.0,
        root_dirichlet_alpha: 0.2,
        root_exploration_fraction: 0.15,
        temperature: 0.5,
        discount: 0.97,
        sampled_k: Some(5),
    };
    let roundtrip = MctsConfig::from_recipe(cfg.recipe());
    assert_eq!(roundtrip, cfg);
}
