//! `checkpoint.rs` BestTracker 测试：min_delta 判据、smoke 禁用、多 seed 路径解析

use super::super::checkpoint::BestTracker;
use super::super::config::CheckpointSettings;
use std::path::PathBuf;

fn enabled_checkpoint(base: PathBuf) -> CheckpointSettings {
    CheckpointSettings {
        enabled: true,
        best_base: Some(base),
        min_delta: 0.0,
        save_last: false,
    }
}

#[test]
fn should_update_first_eval_always() {
    let t = BestTracker::new(
        &enabled_checkpoint(PathBuf::from("/tmp/x/best")),
        42,
        1,
        4,
        false,
    );
    assert!(t.should_update(9.4));
}

#[test]
fn min_delta_zero_requires_strict_improvement_or_equal() {
    let mut t = BestTracker::new(
        &enabled_checkpoint(PathBuf::from("/tmp/x/best")),
        42,
        1,
        4,
        false,
    );
    t.best_score = 100.0;
    assert!(t.should_update(100.0));
    assert!(!t.should_update(99.9));
    assert!(t.should_update(100.1));
}

#[test]
fn min_delta_one_uses_gte_threshold() {
    let mut t = BestTracker::new(
        &CheckpointSettings {
            enabled: true,
            best_base: Some(PathBuf::from("/tmp/x/best")),
            min_delta: 1.0,
            save_last: false,
        },
        42,
        1,
        4,
        false,
    );
    t.best_score = 499.0;
    assert!(t.should_update(500.0));
    assert!(!t.should_update(499.5));
}

#[test]
fn disabled_when_smoke() {
    let t = BestTracker::new(
        &enabled_checkpoint(PathBuf::from("/tmp/x/best")),
        42,
        1,
        4,
        true,
    );
    assert!(!t.should_update(500.0));
}

#[test]
fn disabled_by_default() {
    let t = BestTracker::new(&CheckpointSettings::default(), 7, 1, 4, false);
    assert!(!t.should_update(500.0));
    assert!(t.model_path().is_none());
}

#[test]
fn uses_explicit_path_single_seed() {
    let base = PathBuf::from("/tmp/my_cartpole_best");
    let t = BestTracker::new(&enabled_checkpoint(base.clone()), 42, 1, 4, false);
    assert!(t.should_update(500.0));
    assert_eq!(t.best_base, base);
}

#[test]
fn multi_seed_inserts_seed_subdir() {
    let t = BestTracker::new(
        &enabled_checkpoint(PathBuf::from("models/foo/best")),
        43,
        3,
        4,
        false,
    );
    assert_eq!(t.best_base, PathBuf::from("models/foo/seed_43/best"));
}
