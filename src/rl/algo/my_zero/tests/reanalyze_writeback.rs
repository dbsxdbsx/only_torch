//! Reanalyze 写回闭环：sample_indexed → reanalyze → train → update_at

use crate::nn::{Adam, Graph};
use crate::rl::algo::my_zero::action::ActionAdapter;
use crate::rl::algo::my_zero::network::MyZeroModel;
use crate::rl::algo::my_zero::runner::{
    TrainBatchItem, prepare_train_batch, writeback_reanalyzed_samples,
};
use crate::rl::buffer::{GameOutcome, ReplayBuffer, SelfPlayGame, SelfPlayStep};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn make_step(obs_tag: f32, root_value: f32) -> SelfPlayStep {
    SelfPlayStep {
        obs: vec![obs_tag, 0.0, 0.0, 0.0],
        action: vec![0.0],
        policy_target: vec![1.0, 0.0],
        player: 0,
        reward: 1.0,
        root_value: Some(root_value),
        terminated: false,
        extras: Default::default(),
    }
}

fn make_game(steps: Vec<SelfPlayStep>) -> SelfPlayGame {
    SelfPlayGame {
        steps,
        outcome: GameOutcome::InProgress,
    }
}

#[test]
fn writeback_skips_items_without_buffer_idx() {
    let mut buf = ReplayBuffer::new(4);
    buf.push(make_game(vec![make_step(0.0, 10.0), make_step(1.0, 10.0)]));

    let mut updated = make_game(vec![make_step(0.0, 99.0), make_step(1.0, 99.0)]);
    updated.steps[0].root_value = Some(99.0);

    writeback_reanalyzed_samples(
        &mut buf,
        vec![TrainBatchItem {
            buffer_idx: None,
            game: updated,
            start: 0,
        }],
    );

    let stored = buf.get_at(0).expect("下标 0 应存在");
    assert_eq!(
        stored.steps[0].root_value,
        Some(10.0),
        "无 buffer_idx 时不应写回"
    );
}

#[test]
fn writeback_last_wins_on_duplicate_index() {
    let mut buf = ReplayBuffer::new(4);
    buf.push(make_game(vec![make_step(0.0, 1.0), make_step(1.0, 1.0)]));

    let mut first = make_game(vec![make_step(0.0, 11.0), make_step(1.0, 11.0)]);
    first.steps[0].root_value = Some(11.0);
    let mut second = make_game(vec![make_step(0.0, 22.0), make_step(1.0, 22.0)]);
    second.steps[0].root_value = Some(22.0);

    writeback_reanalyzed_samples(
        &mut buf,
        vec![
            TrainBatchItem {
                buffer_idx: Some(0),
                game: first,
                start: 0,
            },
            TrainBatchItem {
                buffer_idx: Some(0),
                game: second,
                start: 1,
            },
        ],
    );

    let stored = buf.get_at(0).expect("下标 0 应存在");
    assert_eq!(
        stored.steps[0].root_value,
        Some(22.0),
        "同 batch 重复 idx 应以后写者为准"
    );
}

#[test]
fn prepare_reanalyze_then_writeback_updates_buffer() {
    let graph = Graph::new_with_seed(0);
    let model = MyZeroModel::new(&graph, 4, 2, 32).unwrap();
    let adapter = ActionAdapter::discrete_for_test(2);
    let mut opt = Adam::new(&graph, &model.parameters(), 1e-3);

    let mut buf = ReplayBuffer::new(8);
    buf.push(make_game(vec![
        make_step(0.0, 999.0),
        make_step(1.0, 999.0),
        make_step(2.0, 999.0),
    ]));

    let mut rng = StdRng::seed_from_u64(42);
    let batch = prepare_train_batch(&buf, 1, 1, true, &model, &adapter, 0.99, 8, &mut rng);
    assert_eq!(batch.len(), 1);
    let idx = batch[0]
        .buffer_idx
        .expect("reanalyze 路径应携带 buffer 下标");
    let before = buf.get_at(idx).unwrap().steps[0].root_value;

    // train 只借 ref，不消费 batch
    let train_view: Vec<_> = batch.iter().map(|item| (&item.game, item.start)).collect();
    crate::rl::algo::my_zero::runner::train_batch(
        &model,
        &mut opt,
        &train_view,
        1,
        5,
        0.99,
        &Default::default(),
        &mut rng,
    )
    .unwrap();

    let refreshed_in_batch = batch[0].game.steps[batch[0].start]
        .root_value
        .expect("reanalyze 应刷新 root_value");
    assert!(
        (refreshed_in_batch - 999.0).abs() > 1.0,
        "副本上 reanalyze 应改变 root_value，got {refreshed_in_batch}"
    );

    let start = batch[0].start;
    writeback_reanalyzed_samples(&mut buf, batch);

    let after = buf.get_at(idx).unwrap();
    assert_eq!(
        after.steps[start].root_value,
        Some(refreshed_in_batch),
        "写回后 buffer 应与 train 前 reanalyze 结果一致"
    );
    assert_eq!(before, Some(999.0), "写回前 buffer 仍应为旧标签");
}

#[test]
fn prepare_without_reanalyze_does_not_attach_buffer_idx() {
    let graph = Graph::new_with_seed(1);
    let model = MyZeroModel::new(&graph, 4, 2, 32).unwrap();
    let adapter = ActionAdapter::discrete_for_test(2);

    let mut buf = ReplayBuffer::new(4);
    buf.push(make_game(vec![make_step(0.0, 5.0), make_step(1.0, 5.0)]));

    let mut rng = StdRng::seed_from_u64(3);
    let batch = prepare_train_batch(&buf, 1, 1, false, &model, &adapter, 0.99, 4, &mut rng);
    assert_eq!(batch.len(), 1);
    assert!(batch[0].buffer_idx.is_none(), "未开 reanalyze 时不应写回");
}
