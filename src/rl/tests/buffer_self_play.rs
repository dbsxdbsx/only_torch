//! SelfPlayGame + ReplayBuffer 单元测试（纯 Rust，无 pyo3）

use crate::rl::buffer::{GameOutcome, ReplayBuffer, SelfPlayGame, SelfPlayStep};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn make_step(id: usize, player: u8) -> SelfPlayStep {
    let action_dim = 9;
    let mut policy = vec![0.0; action_dim];
    policy[id % action_dim] = 1.0;
    SelfPlayStep {
        obs: vec![id as f32; 4],
        action: vec![(id % action_dim) as f32],
        policy_target: policy,
        player,
        reward: 0.0,
        root_value: None,
        terminated: false,
    }
}

fn make_game(n_steps: usize, outcome: GameOutcome) -> SelfPlayGame {
    let steps = (0..n_steps).map(|i| make_step(i, (i % 2) as u8)).collect();
    SelfPlayGame { steps, outcome }
}

// ============================================================================
// BufferItem 兼容性
// ============================================================================

#[test]
fn test_self_play_game_as_buffer_item() {
    let mut buf = ReplayBuffer::new(10);
    buf.push(make_game(5, GameOutcome::Win(0)));
    assert_eq!(buf.len(), 1);
}

// ============================================================================
// push + sample 内容保真
// ============================================================================

#[test]
fn test_sample_steps_preserved() {
    let mut buf = ReplayBuffer::new(10);
    let game = make_game(3, GameOutcome::Win(1));
    buf.push(game.clone());

    let mut rng = StdRng::seed_from_u64(42);
    let batch = buf.sample(1, &mut rng);
    assert_eq!(batch.len(), 1);

    let sampled = &batch[0];
    assert_eq!(sampled.steps.len(), 3);
    for (orig, got) in game.steps.iter().zip(sampled.steps.iter()) {
        assert_eq!(orig.obs, got.obs);
        assert_eq!(orig.policy_target, got.policy_target);
        assert_eq!(orig.player, got.player);
    }
}

// ============================================================================
// outcome 保真
// ============================================================================

#[test]
fn test_outcome_win_preserved() {
    let mut buf = ReplayBuffer::new(10);
    buf.push(make_game(2, GameOutcome::Win(0)));

    let mut rng = StdRng::seed_from_u64(0);
    let batch = buf.sample(1, &mut rng);
    assert_eq!(batch[0].outcome, GameOutcome::Win(0));
}

#[test]
fn test_outcome_draw_preserved() {
    let mut buf = ReplayBuffer::new(10);
    buf.push(make_game(4, GameOutcome::Draw));

    let mut rng = StdRng::seed_from_u64(0);
    let batch = buf.sample(1, &mut rng);
    assert_eq!(batch[0].outcome, GameOutcome::Draw);
}

#[test]
fn test_outcome_in_progress() {
    let mut buf = ReplayBuffer::new(10);
    buf.push(make_game(1, GameOutcome::InProgress));

    let mut rng = StdRng::seed_from_u64(0);
    let batch = buf.sample(1, &mut rng);
    assert_eq!(batch[0].outcome, GameOutcome::InProgress);
}

// ============================================================================
// 空局也能正常 push/sample
// ============================================================================

#[test]
fn test_empty_game_push_sample() {
    let mut buf = ReplayBuffer::new(10);
    buf.push(make_game(0, GameOutcome::Draw));
    assert_eq!(buf.len(), 1);

    let mut rng = StdRng::seed_from_u64(0);
    let batch = buf.sample(1, &mut rng);
    assert_eq!(batch.len(), 1);
    assert!(batch[0].steps.is_empty());
    assert_eq!(batch[0].outcome, GameOutcome::Draw);
}

// ============================================================================
// 容量满 FIFO 淘汰
// ============================================================================

#[test]
fn test_fifo_eviction() {
    let mut buf = ReplayBuffer::new(3);
    for i in 0..5 {
        buf.push(make_game(i + 1, GameOutcome::Win((i % 2) as u8)));
    }
    // 容量 3，push 5 次 → 前 2 个被淘汰，剩下 steps 长度分别为 3, 4, 5
    assert_eq!(buf.len(), 3);

    let mut rng = StdRng::seed_from_u64(99);
    let all = buf.sample(3, &mut rng);
    for game in &all {
        assert!(game.steps.len() >= 3);
    }
}
