//! `component.rs` 组件开关测试：默认全关 = base、cq/loss 系数默认值

use super::super::component::Components;

#[test]
fn default_is_all_off() {
    let c = Components::default();
    assert!(!c.consistency);
    assert!(!c.reconstruction);
    assert!(!c.value_prefix);
    assert!(!c.reanalyze);
    assert!(!c.target_net);
    assert!(!c.sve_enabled());
    assert!(!c.gumbel);
    assert!(!c.completed_q_target);
    assert!(!c.sampled);
}

#[test]
fn cq_defaults_match_reference_qtransform() {
    let c = Components::default();
    assert!((c.cq_c_visit - 50.0).abs() < 1e-6);
    assert!((c.cq_c_scale - 1.0).abs() < 1e-6);
}

#[test]
fn loss_coef_defaults_match_constants() {
    let c = Components::default();
    assert!(
        (c.consistency_coef - 2.0).abs() < 1e-6,
        "consistency 默认 2.0"
    );
    assert!(
        (c.reconstruction_coef - 16.0).abs() < 1e-6,
        "reconstruction 默认 16.0（v0.26 P0 重标定；论文 lg=1.0 已被消融证伪）"
    );
    assert!(
        (c.continuation_coef - 1.0).abs() < 1e-6,
        "continuation 默认 1.0"
    );
}
