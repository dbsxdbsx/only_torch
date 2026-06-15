//! RolloutBuffer + RolloutStep 单元测试（纯 Rust，无 pyo3）

use crate::rl::buffer::{RolloutBuffer, RolloutStep};

fn make_step(id: usize, terminated: bool, truncated: bool) -> RolloutStep {
    RolloutStep {
        obs: vec![id as f32; 4],
        action: vec![id as f32],
        log_prob: -(id as f32 + 1.0).ln(),
        value: id as f32 * 0.5,
        reward: id as f32 * 0.1,
        terminated,
        truncated,
    }
}

// ============================================================================
// 创建与容量
// ============================================================================

#[test]
fn test_new_creates_correct_capacity() {
    let buf = RolloutBuffer::new(128);
    assert_eq!(buf.capacity(), 128);
    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());
}

// ============================================================================
// push + len
// ============================================================================

#[test]
fn test_push_and_len() {
    let mut buf = RolloutBuffer::new(10);
    buf.push(make_step(0, false, false));
    buf.push(make_step(1, false, false));
    assert_eq!(buf.len(), 2);
    assert!(!buf.is_empty());
}

// ============================================================================
// is_full
// ============================================================================

#[test]
fn test_is_full_at_capacity() {
    let mut buf = RolloutBuffer::new(3);
    for i in 0..3 {
        assert!(!buf.is_full());
        buf.push(make_step(i, false, false));
    }
    assert!(buf.is_full());
}

// ============================================================================
// push 满时 panic
// ============================================================================

#[test]
#[should_panic(expected = "RolloutBuffer 已满")]
fn test_push_panics_when_full() {
    let mut buf = RolloutBuffer::new(2);
    buf.push(make_step(0, false, false));
    buf.push(make_step(1, false, false));
    buf.push(make_step(2, false, false)); // 应 panic
}

// ============================================================================
// clear
// ============================================================================

#[test]
fn test_clear_resets_buffer() {
    let mut buf = RolloutBuffer::new(5);
    for i in 0..5 {
        buf.push(make_step(i, false, false));
    }
    assert!(buf.is_full());

    buf.clear();
    assert!(buf.is_empty());
    assert_eq!(buf.len(), 0);
    assert!(!buf.is_full());
}

// ============================================================================
// steps() 引用切片
// ============================================================================

#[test]
fn test_steps_returns_correct_slice() {
    let mut buf = RolloutBuffer::new(10);
    buf.push(make_step(0, false, false));
    buf.push(make_step(1, true, false));
    buf.push(make_step(2, false, true));

    let steps = buf.steps();
    assert_eq!(steps.len(), 3);
    assert_eq!(steps[0].obs, vec![0.0; 4]);
    assert_eq!(steps[1].obs, vec![1.0; 4]);
    assert_eq!(steps[2].obs, vec![2.0; 4]);
}

// ============================================================================
// 字段保真：terminated / truncated / log_prob / value 不串位
// ============================================================================

#[test]
fn test_field_fidelity() {
    let mut buf = RolloutBuffer::new(10);

    buf.push(RolloutStep {
        obs: vec![1.0, 2.0, 3.0],
        action: vec![0.5],
        log_prob: -0.693,
        value: 1.25,
        reward: 0.1,
        terminated: false,
        truncated: true,
    });
    buf.push(RolloutStep {
        obs: vec![4.0, 5.0, 6.0],
        action: vec![-0.3],
        log_prob: -1.386,
        value: 2.50,
        reward: -0.5,
        terminated: true,
        truncated: false,
    });

    let steps = buf.steps();

    // 第一步
    assert_eq!(steps[0].obs, vec![1.0, 2.0, 3.0]);
    assert_eq!(steps[0].action, vec![0.5]);
    assert!((steps[0].log_prob - (-0.693)).abs() < 1e-6);
    assert!((steps[0].value - 1.25).abs() < 1e-6);
    assert!((steps[0].reward - 0.1).abs() < 1e-6);
    assert!(!steps[0].terminated);
    assert!(steps[0].truncated);

    // 第二步
    assert_eq!(steps[1].obs, vec![4.0, 5.0, 6.0]);
    assert_eq!(steps[1].action, vec![-0.3]);
    assert!((steps[1].log_prob - (-1.386)).abs() < 1e-6);
    assert!((steps[1].value - 2.50).abs() < 1e-6);
    assert!((steps[1].reward - (-0.5)).abs() < 1e-6);
    assert!(steps[1].terminated);
    assert!(!steps[1].truncated);
}
