//! Sampled MuZero 候选采样测试：全覆盖退化 / 子采样 π̂_β prior / 根 Dirichlet 组合

use crate::rl::mcts::sampled::{
    sample_for_expansion, sample_root_for_expansion, sampled_puct_priors,
};
use crate::rl::mcts::{ActionCandidate, ActionId, ActionPayload, CandidateSet, MctsConfig};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn candidate_set(priors: Vec<f32>) -> CandidateSet {
    CandidateSet {
        candidates: priors
            .into_iter()
            .enumerate()
            .map(|(idx, prior)| {
                ActionCandidate::new(ActionId(idx), ActionPayload::Discrete(idx), prior)
            })
            .collect(),
    }
}

#[test]
fn full_coverage_returns_original_prior() {
    let candidates = candidate_set(vec![0.7, 0.3]);
    let mut rng = StdRng::seed_from_u64(0);
    let out = sample_for_expansion(&candidates, 2, &mut rng);
    let priors = out.policy_priors();
    assert_eq!(out.len(), 2);
    assert!((priors[0] - 0.7).abs() < 1e-5);
    assert!((priors[1] - 0.3).abs() < 1e-5);
}

#[test]
fn subsample_k1_has_valid_prior() {
    let candidates = candidate_set(vec![0.1, 0.1, 0.2, 0.3, 0.3]);
    let mut rng = StdRng::seed_from_u64(42);
    let priors = sample_for_expansion(&candidates, 1, &mut rng).policy_priors();
    assert_eq!(priors.len(), 1);
    assert!((priors[0] - 1.0).abs() < 1e-5);
}

#[test]
fn subsample_k3_prior_sums_to_one() {
    let candidates = candidate_set(vec![0.05; 6]);
    let mut rng = StdRng::seed_from_u64(7);
    let out = sample_for_expansion(&candidates, 3, &mut rng);
    let priors = out.policy_priors();
    assert_eq!(out.len(), 3);
    assert_eq!(priors.len(), 3);
    let s: f32 = priors.iter().sum();
    assert!((s - 1.0).abs() < 1e-5);
}

#[test]
fn beta_equals_pi_gives_uniform_prior_on_sampled_subset() {
    // 论文 remark：β=π（τ=1）时 π̂_β = β̂，即子集内近似 uniform；
    // 网络 prior 只影响“采到谁”，不应在采到后再重复偏置一次。
    let beta = vec![0.1, 0.2, 0.7];
    let pi = beta.clone();
    let priors = sampled_puct_priors(&beta, &pi, &[0, 2]);
    assert_eq!(priors.len(), 2);
    assert!((priors[0] - 0.5).abs() < 1e-5, "got {priors:?}");
    assert!((priors[1] - 0.5).abs() < 1e-5, "got {priors:?}");
}

#[test]
fn uniform_beta_preserves_network_prior_within_subset() {
    let beta = vec![1.0 / 3.0; 3];
    let pi = vec![0.1, 0.2, 0.7];
    let priors = sampled_puct_priors(&beta, &pi, &[0, 2]);
    let expected0 = 0.1 / (0.1 + 0.7);
    let expected1 = 0.7 / (0.1 + 0.7);
    assert!((priors[0] - expected0).abs() < 1e-5, "got {priors:?}");
    assert!((priors[1] - expected1).abs() < 1e-5, "got {priors:?}");
}

#[test]
fn sample_root_applies_dirichlet_then_expands() {
    let candidates = candidate_set(vec![0.5, 0.5]);
    let cfg = MctsConfig::default();
    let mut rng = StdRng::seed_from_u64(99);
    let out = sample_root_for_expansion(&candidates, &cfg, 2, &mut rng);
    let priors = out.policy_priors();
    assert_eq!(out.len(), 2);
    let s: f32 = priors.iter().sum();
    assert!((s - 1.0).abs() < 1e-5);
}
