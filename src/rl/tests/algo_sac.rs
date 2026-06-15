//! SAC helper 单元测试
//!
//! 覆盖 `transitions_to_batch`、TD target、V 值计算、alpha 更新

use crate::rl::Transition;
use crate::rl::algo::sac::{
    compute_td_target, compute_v_continuous, compute_v_discrete, compute_v_hybrid,
    transitions_to_batch, update_alpha,
};
use crate::tensor::Tensor;

// ============================================================================
// transitions_to_batch
// ============================================================================

fn make_transition(obs: &[f32], action: &[f32], reward: f32, terminated: bool) -> Transition {
    Transition {
        obs: obs.to_vec(),
        action: action.to_vec(),
        reward,
        next_obs: obs.iter().map(|x| x + 1.0).collect(),
        terminated,
        truncated: false,
    }
}

#[test]
fn batch_basic_shape() {
    let ts = vec![
        make_transition(&[1.0, 2.0], &[0.0], 1.0, false),
        make_transition(&[3.0, 4.0], &[1.0], -1.0, true),
    ];
    let b = transitions_to_batch(&ts, 2);
    assert_eq!(b.obs.shape(), &[2, 2]);
    assert_eq!(b.actions.shape(), &[2, 1]);
    assert_eq!(b.rewards.shape(), &[2, 1]);
    assert_eq!(b.next_obs.shape(), &[2, 2]);
    assert_eq!(b.not_terminated.shape(), &[2, 1]);
}

#[test]
fn batch_obs_content() {
    let ts = vec![
        make_transition(&[1.0, 2.0, 3.0], &[0.0], 0.5, false),
        make_transition(&[4.0, 5.0, 6.0], &[1.0], -0.5, true),
    ];
    let b = transitions_to_batch(&ts, 3);
    assert_eq!(b.obs[[0, 0]], 1.0);
    assert_eq!(b.obs[[1, 2]], 6.0);
    assert_eq!(b.next_obs[[0, 0]], 2.0);
    assert_eq!(b.next_obs[[1, 2]], 7.0);
}

#[test]
fn batch_rewards_content() {
    let ts = vec![
        make_transition(&[0.0], &[0.0], 1.5, false),
        make_transition(&[0.0], &[0.0], -2.0, false),
    ];
    let b = transitions_to_batch(&ts, 1);
    assert_eq!(b.rewards[[0, 0]], 1.5);
    assert_eq!(b.rewards[[1, 0]], -2.0);
}

#[test]
fn batch_not_terminated_mask() {
    let ts = vec![
        make_transition(&[0.0], &[0.0], 0.0, false),
        make_transition(&[0.0], &[0.0], 0.0, true),
        make_transition(&[0.0], &[0.0], 0.0, false),
    ];
    let b = transitions_to_batch(&ts, 1);
    assert_eq!(b.not_terminated[[0, 0]], 1.0);
    assert_eq!(b.not_terminated[[1, 0]], 0.0);
    assert_eq!(b.not_terminated[[2, 0]], 1.0);
}

#[test]
fn batch_multi_dim_action() {
    let ts = vec![
        make_transition(&[0.0], &[1.0, 0.5, -0.3], 0.0, false),
        make_transition(&[0.0], &[2.0, 0.1, 0.8], 0.0, false),
    ];
    let b = transitions_to_batch(&ts, 1);
    assert_eq!(b.actions.shape(), &[2, 3]);
    assert_eq!(b.actions[[0, 0]], 1.0);
    assert_eq!(b.actions[[1, 2]], 0.8);
}

#[test]
#[should_panic(expected = "空 batch")]
fn batch_empty_panics() {
    transitions_to_batch(&[], 1);
}

// ============================================================================
// compute_td_target
// ============================================================================

#[test]
fn td_target_basic() {
    let rewards = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let v_next = Tensor::new(&[10.0, 20.0], &[2, 1]);
    let not_term = Tensor::new(&[1.0, 1.0], &[2, 1]);
    let target = compute_td_target(&rewards, &v_next, &not_term, 0.99);
    assert!((target[[0, 0]] - (1.0 + 0.99 * 10.0)).abs() < 1e-5);
    assert!((target[[1, 0]] - (2.0 + 0.99 * 20.0)).abs() < 1e-5);
}

#[test]
fn td_target_terminated_no_bootstrap() {
    let rewards = Tensor::new(&[1.0], &[1, 1]);
    let v_next = Tensor::new(&[100.0], &[1, 1]);
    let not_term = Tensor::new(&[0.0], &[1, 1]); // terminated
    let target = compute_td_target(&rewards, &v_next, &not_term, 0.99);
    assert!((target[[0, 0]] - 1.0).abs() < 1e-5);
}

#[test]
fn td_target_gamma_zero() {
    let rewards = Tensor::new(&[5.0], &[1, 1]);
    let v_next = Tensor::new(&[100.0], &[1, 1]);
    let not_term = Tensor::new(&[1.0], &[1, 1]);
    let target = compute_td_target(&rewards, &v_next, &not_term, 0.0);
    assert!((target[[0, 0]] - 5.0).abs() < 1e-5);
}

// ============================================================================
// compute_v_discrete
// ============================================================================

#[test]
fn v_discrete_uniform_probs() {
    // 均匀分布，2 个动作
    let probs = Tensor::new(&[0.5, 0.5], &[1, 2]);
    let q = Tensor::new(&[1.0, 3.0], &[1, 2]);
    let lp = Tensor::new(&[0.5_f32.ln(), 0.5_f32.ln()], &[1, 2]);
    let v = compute_v_discrete(&probs, &q, &lp, 1.0);
    // V = 0.5 * (1 - 1*ln(0.5)) + 0.5 * (3 - 1*ln(0.5))
    //   = 0.5 * (1 + 0.693) + 0.5 * (3 + 0.693)
    //   = 0.5*1.693 + 0.5*3.693 = 0.847 + 1.847 = 2.693
    let expected = 0.5 * (1.0 - 0.5_f32.ln()) + 0.5 * (3.0 - 0.5_f32.ln());
    assert!((v[[0, 0]] - expected).abs() < 1e-4);
}

#[test]
fn v_discrete_alpha_zero_equals_expected_q() {
    // α=0 → V = E_π[Q]
    let probs = Tensor::new(&[0.3, 0.7], &[1, 2]);
    let q = Tensor::new(&[2.0, 4.0], &[1, 2]);
    let lp = Tensor::new(&[0.3_f32.ln(), 0.7_f32.ln()], &[1, 2]);
    let v = compute_v_discrete(&probs, &q, &lp, 0.0);
    let expected = 0.3 * 2.0 + 0.7 * 4.0;
    assert!((v[[0, 0]] - expected).abs() < 1e-4);
}

#[test]
fn v_discrete_batch() {
    let probs = Tensor::new(&[0.5, 0.5, 0.2, 0.8], &[2, 2]);
    let q = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let lp = probs.ln();
    let v = compute_v_discrete(&probs, &q, &lp, 0.0);
    assert_eq!(v.shape(), &[2, 1]);
    assert!((v[[0, 0]] - 1.0).abs() < 1e-4);
    assert!((v[[1, 0]] - 1.0).abs() < 1e-4);
}

// ============================================================================
// compute_v_continuous
// ============================================================================

#[test]
fn v_continuous_basic() {
    let q_min = Tensor::new(&[5.0, 3.0], &[2, 1]);
    let lp_sum = Tensor::new(&[-1.0, -2.0], &[2, 1]);
    let v = compute_v_continuous(&q_min, &lp_sum, 0.2);
    // V = Q - α * lp → [5 - 0.2*(-1), 3 - 0.2*(-2)] = [5.2, 3.4]
    assert!((v[[0, 0]] - 5.2).abs() < 1e-5);
    assert!((v[[1, 0]] - 3.4).abs() < 1e-5);
}

#[test]
fn v_continuous_alpha_zero() {
    let q = Tensor::new(&[7.0], &[1, 1]);
    let lp = Tensor::new(&[-3.0], &[1, 1]);
    let v = compute_v_continuous(&q, &lp, 0.0);
    assert!((v[[0, 0]] - 7.0).abs() < 1e-5);
}

// ============================================================================
// compute_v_hybrid
// ============================================================================

#[test]
fn v_hybrid_basic() {
    let probs = Tensor::new(&[0.5, 0.5], &[1, 2]);
    let q = Tensor::new(&[2.0, 4.0], &[1, 2]);
    let lp = Tensor::new(&[0.5_f32.ln(), 0.5_f32.ln()], &[1, 2]);
    let cont_lp_sum = Tensor::new(&[-1.0], &[1, 1]);
    let v = compute_v_hybrid(&probs, &q, &lp, 0.1, &cont_lp_sum, 0.2);
    // discrete V = Σ π(d)(Q(d) - 0.1*ln π(d))
    //   = 0.5*(2-0.1*ln0.5) + 0.5*(4-0.1*ln0.5)
    //   = 0.5*(2+0.0693) + 0.5*(4+0.0693) = 3.0693
    // hybrid V = discrete_V - 0.2 * (-1) = 3.0693 + 0.2 = 3.2693
    let dv = 0.5 * (2.0 - 0.1 * 0.5_f32.ln()) + 0.5 * (4.0 - 0.1 * 0.5_f32.ln());
    let expected = dv - 0.2 * (-1.0);
    assert!((v[[0, 0]] - expected).abs() < 1e-3);
}

#[test]
fn v_hybrid_alpha_zero_both() {
    let probs = Tensor::new(&[0.3, 0.7], &[1, 2]);
    let q = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let lp = Tensor::new(&[0.3_f32.ln(), 0.7_f32.ln()], &[1, 2]);
    let cont_lp = Tensor::new(&[-5.0], &[1, 1]);
    let v = compute_v_hybrid(&probs, &q, &lp, 0.0, &cont_lp, 0.0);
    let expected = 0.3 * 1.0 + 0.7 * 2.0;
    assert!((v[[0, 0]] - expected).abs() < 1e-4);
}

// ============================================================================
// update_alpha
// ============================================================================

#[test]
fn alpha_update_entropy_below_target_increases_alpha() {
    let log_alpha = 0.0; // α = 1.0
    let new = update_alpha(log_alpha, 0.01, 0.3, 0.5);
    // 实际熵 < 目标 → grad 为负 → log_alpha 增大 → α 增大
    assert!(new > log_alpha);
}

#[test]
fn alpha_update_entropy_above_target_decreases_alpha() {
    let log_alpha = 0.0;
    let new = update_alpha(log_alpha, 0.01, 0.7, 0.5);
    assert!(new < log_alpha);
}

#[test]
fn alpha_update_at_target_no_change() {
    let log_alpha = 0.5;
    let new = update_alpha(log_alpha, 0.01, 1.0, 1.0);
    assert!((new - log_alpha).abs() < 1e-6);
}

#[test]
fn alpha_update_clamp_lower() {
    let new = update_alpha(-30.0, 0.01, 10.0, 0.0);
    assert!(new >= -20.0);
}

#[test]
fn alpha_update_clamp_upper() {
    let new = update_alpha(10.0, 0.01, 0.0, 100.0);
    assert!(new <= 2.0);
}

#[test]
fn alpha_update_lr_zero_no_change() {
    let log_alpha = 0.5;
    let new = update_alpha(log_alpha, 0.0, 0.0, 1.0);
    assert!((new - log_alpha).abs() < 1e-6);
}
