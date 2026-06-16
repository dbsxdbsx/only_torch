//! ReplayBuffer + Transition 单元测试（纯 Rust，无 pyo3）

use crate::rl::buffer::{BufferItem, ReplayBuffer, Transition};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn make_transition(id: usize, terminated: bool, truncated: bool) -> Transition {
    Transition {
        obs: vec![id as f32; 4],
        action: vec![id as f32],
        reward: id as f32 * 0.1,
        next_obs: vec![(id + 1) as f32; 4],
        terminated,
        truncated,
    }
}

// ============================================================================
// 容量与 FIFO
// ============================================================================

#[test]
fn test_fifo_eviction() {
    let mut buf = ReplayBuffer::new(3);
    for i in 0..5 {
        buf.push(make_transition(i, false, false));
    }
    // 容量 3，push 5 次 → 前 2 个被淘汰，剩 [2,3,4]
    assert_eq!(buf.len(), 3);
    assert_eq!(buf.capacity(), 3);
}

#[test]
fn test_push_within_capacity() {
    let mut buf = ReplayBuffer::new(10);
    buf.push(make_transition(0, false, false));
    buf.push(make_transition(1, false, false));
    assert_eq!(buf.len(), 2);
    assert!(!buf.is_empty());
}

#[test]
fn test_clear() {
    let mut buf = ReplayBuffer::new(10);
    buf.push(make_transition(0, false, false));
    buf.clear();
    assert!(buf.is_empty());
    assert_eq!(buf.len(), 0);
}

// ============================================================================
// 采样语义
// ============================================================================

#[test]
fn test_sample_empty_batch() {
    let buf: ReplayBuffer<Transition> = ReplayBuffer::new(10);
    let mut rng = StdRng::seed_from_u64(0);
    let batch = buf.sample(0, &mut rng);
    assert!(batch.is_empty());
}

#[test]
fn test_sample_from_empty_buffer() {
    let buf: ReplayBuffer<Transition> = ReplayBuffer::new(10);
    let mut rng = StdRng::seed_from_u64(0);
    let batch = buf.sample(5, &mut rng);
    assert!(batch.is_empty());
}

#[test]
fn test_sample_with_replacement() {
    // len == 1 时 sample(5) 应返回 5 个（有放回允许重复）
    let mut buf = ReplayBuffer::new(10);
    buf.push(make_transition(42, false, false));
    let mut rng = StdRng::seed_from_u64(0);
    let batch = buf.sample(5, &mut rng);
    assert_eq!(batch.len(), 5);
    for item in &batch {
        assert_eq!(item.obs, vec![42.0; 4]);
    }
}

#[test]
fn test_sample_batch_larger_than_buffer() {
    let mut buf = ReplayBuffer::new(10);
    buf.push(make_transition(0, false, false));
    buf.push(make_transition(1, false, false));
    let mut rng = StdRng::seed_from_u64(0);
    // batch_size(10) > len(2)，有放回仍返回 10 条
    let batch = buf.sample(10, &mut rng);
    assert_eq!(batch.len(), 10);
}

#[test]
fn test_sample_reproducible_with_same_seed() {
    let mut buf = ReplayBuffer::new(100);
    for i in 0..50 {
        buf.push(make_transition(i, false, false));
    }
    let mut rng1 = StdRng::seed_from_u64(123);
    let mut rng2 = StdRng::seed_from_u64(123);
    let batch1 = buf.sample(16, &mut rng1);
    let batch2 = buf.sample(16, &mut rng2);
    for (a, b) in batch1.iter().zip(batch2.iter()) {
        assert_eq!(a.obs, b.obs);
        assert_eq!(a.reward, b.reward);
    }
}

// ============================================================================
// Transition action shape × 3
// ============================================================================

#[test]
fn test_action_shape_discrete() {
    let t = Transition {
        obs: vec![0.0; 4],
        action: vec![1.0], // Discrete: idx as f32
        reward: 1.0,
        next_obs: vec![0.0; 4],
        terminated: false,
        truncated: false,
    };
    let mut buf = ReplayBuffer::new(10);
    buf.push(t);
    let mut rng = StdRng::seed_from_u64(0);
    let batch = buf.sample(1, &mut rng);
    assert_eq!(batch[0].action, vec![1.0]);
}

#[test]
fn test_action_shape_continuous() {
    let t = Transition {
        obs: vec![0.0; 3],
        action: vec![0.5, -0.3], // Box(2,)
        reward: -1.5,
        next_obs: vec![0.1; 3],
        terminated: false,
        truncated: true,
    };
    let mut buf = ReplayBuffer::new(10);
    buf.push(t);
    let mut rng = StdRng::seed_from_u64(0);
    let batch = buf.sample(1, &mut rng);
    assert_eq!(batch[0].action, vec![0.5, -0.3]);
}

#[test]
fn test_action_shape_hybrid_tuple() {
    let t = Transition {
        obs: vec![0.0; 10],
        action: vec![2.0, 15.0, 360.0, 200.0], // [discrete, c0, c1, c2]
        reward: 0.08,
        next_obs: vec![0.1; 10],
        terminated: true,
        truncated: false,
    };
    let mut buf = ReplayBuffer::new(10);
    buf.push(t);
    let mut rng = StdRng::seed_from_u64(0);
    let batch = buf.sample(1, &mut rng);
    assert_eq!(batch[0].action.len(), 4);
    assert_eq!(batch[0].action[0], 2.0);
}

// ============================================================================
// 终止字段保真
// ============================================================================

#[test]
fn test_terminated_truncated_preserved() {
    let mut buf = ReplayBuffer::new(10);
    // terminated=false, truncated=true（CartPole 撞步数上限场景）
    buf.push(Transition {
        obs: vec![1.0; 4],
        action: vec![0.0],
        reward: 1.0,
        next_obs: vec![1.1; 4],
        terminated: false,
        truncated: true,
    });
    let mut rng = StdRng::seed_from_u64(0);
    let batch = buf.sample(1, &mut rng);
    assert!(!batch[0].terminated);
    assert!(batch[0].truncated);
    assert!(batch[0].is_episode_end());
}

#[test]
fn test_is_episode_end() {
    let t1 = make_transition(0, false, false);
    assert!(!t1.is_episode_end());

    let t2 = make_transition(1, true, false);
    assert!(t2.is_episode_end());

    let t3 = make_transition(2, false, true);
    assert!(t3.is_episode_end());

    let t4 = make_transition(3, true, true);
    assert!(t4.is_episode_end());
}

// ============================================================================
// Clone 语义
// ============================================================================

#[test]
fn test_clone_independence() {
    let mut buf = ReplayBuffer::new(10);
    let t = make_transition(7, false, false);
    buf.push(t);
    let mut rng = StdRng::seed_from_u64(0);
    let mut batch = buf.sample(1, &mut rng);
    // 修改 batch 不影响 buffer
    batch[0].obs[0] = 999.0;
    let batch2 = buf.sample(1, &mut rng);
    assert_eq!(batch2[0].obs[0], 7.0);
}

// ============================================================================
// BufferItem trait 可扩展性
// ============================================================================

#[test]
fn test_custom_buffer_item() {
    #[derive(Clone)]
    struct MyItem {
        data: Vec<f32>,
    }
    impl BufferItem for MyItem {}

    let mut buf = ReplayBuffer::new(5);
    buf.push(MyItem {
        data: vec![1.0, 2.0],
    });
    buf.push(MyItem {
        data: vec![3.0, 4.0],
    });
    let mut rng = StdRng::seed_from_u64(0);
    let batch = buf.sample(2, &mut rng);
    assert_eq!(batch.len(), 2);
    assert!(!batch[0].data.is_empty(), "采样的自定义条目应保真");
}

// ============================================================================
// sample_indexed + update_at（v0.24 Phase 0a 接缝：reanalyze 回写 / PER 预留）
// ============================================================================

#[test]
fn test_sample_indexed_returns_valid_indices() {
    let mut buf = ReplayBuffer::new(10);
    for i in 0..5 {
        buf.push(make_transition(i, false, false)); // obs[0] = i = 存储下标
    }
    let mut rng = StdRng::seed_from_u64(7);
    let batch = buf.sample_indexed(8, &mut rng); // 有放回，可超过 len
    assert_eq!(batch.len(), 8);
    for (idx, item) in &batch {
        assert!(*idx < buf.len(), "下标应在 buffer 范围内");
        // 未发生淘汰时存储顺序 == push 顺序 == id == obs[0]
        assert_eq!(item.obs[0] as usize, *idx, "返回的下标应对应该条目");
    }
}

#[test]
fn test_sample_indexed_empty_and_zero() {
    let buf: ReplayBuffer<Transition> = ReplayBuffer::new(4);
    let mut rng = StdRng::seed_from_u64(1);
    assert!(
        buf.sample_indexed(3, &mut rng).is_empty(),
        "空 buffer 返回空"
    );

    let mut buf2 = ReplayBuffer::new(4);
    buf2.push(make_transition(0, false, false));
    assert!(
        buf2.sample_indexed(0, &mut rng).is_empty(),
        "batch=0 返回空"
    );
}

#[test]
fn test_update_at_writes_back() {
    let mut buf = ReplayBuffer::new(4);
    for i in 0..3 {
        buf.push(make_transition(i, false, false));
    }
    // 回写下标 1 → reward 改成 99.0
    let mut replacement = make_transition(1, false, false);
    replacement.reward = 99.0;
    buf.update_at(1, replacement);

    let mut rng = StdRng::seed_from_u64(0);
    let all = buf.sample_indexed(20, &mut rng);
    let got = all
        .iter()
        .find(|(idx, _)| *idx == 1)
        .expect("应能采到下标 1");
    assert_eq!(got.1.reward, 99.0, "update_at 应原地回写");

    // 越界 update_at 是 no-op，不 panic、不改变长度
    buf.update_at(999, make_transition(0, false, false));
    assert_eq!(buf.len(), 3);
}
