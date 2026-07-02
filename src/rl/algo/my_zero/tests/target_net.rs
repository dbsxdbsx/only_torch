//! `target_net.rs` 同步测试：EMA 混合 / hard copy / 同步节拍 / 非连续视图回归

use super::super::target_net::{ema_update, hard_update, is_hard_sync_step};
use crate::nn::{Graph, Var};
use crate::tensor::Tensor;

fn vars(graph: &Graph, data: &[f32]) -> Vec<Var> {
    vec![graph.input(&Tensor::new(data, &[1, data.len()])).unwrap()]
}

#[test]
fn ema_blends_halfway() {
    let graph = Graph::new_with_seed(0);
    let online = vars(&graph, &[1.0, 1.0]);
    let target = vars(&graph, &[0.0, 0.0]);
    ema_update(&online, &target, 0.5);
    let s = target[0].value().unwrap().unwrap();
    let s = s.data_as_slice();
    assert!((s[0] - 0.5).abs() < 1e-6 && (s[1] - 0.5).abs() < 1e-6);
}

#[test]
fn hard_copies_online() {
    let graph = Graph::new_with_seed(0);
    let online = vars(&graph, &[2.0, 3.0]);
    let target = vars(&graph, &[0.0, 0.0]);
    hard_update(&online, &target);
    let s = target[0].value().unwrap().unwrap();
    let s = s.data_as_slice();
    assert!((s[0] - 2.0).abs() < 1e-6 && (s[1] - 3.0).abs() < 1e-6);
}

#[test]
fn hard_sync_step_schedule() {
    assert!(!is_hard_sync_step(0, 5));
    assert!(!is_hard_sync_step(3, 5));
    assert!(is_hard_sync_step(5, 5));
    assert!(is_hard_sync_step(10, 5));
    assert!(!is_hard_sync_step(7, 0), "interval=0 永不 hard（走 EMA）");
}

/// **回归测试**：online 参数被塞入**非连续**视图（`permute` 产物）时，
/// `ema_update` 必须按逻辑行主序混合、不得 panic（此前 `data_as_slice()` 会 panic）。
#[test]
fn ema_update_noncontiguous_online_no_panic() {
    let graph = Graph::new_with_seed(0);
    // base [2,2]=[1,2,3,4] → permute[1,0] → 非连续，逻辑行主序为 [1,3,2,4]
    let base = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let nc = base.permute(&[1, 0]);
    assert!(!nc.is_contiguous(), "permute 结果应为非连续");

    let online = vec![graph.input(&Tensor::zeros(&[2, 2])).unwrap()];
    online[0].set_value(&nc).unwrap();
    let target = vec![graph.input(&Tensor::zeros(&[2, 2])).unwrap()];

    ema_update(&online, &target, 0.5);

    // target = 0.5·online_logical + 0.5·0 = [0.5, 1.5, 1.0, 2.0]（逻辑行主序）
    let out = target[0].value().unwrap().unwrap().to_vec();
    let expected = [0.5, 1.5, 1.0, 2.0];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "EMA 应按逻辑序混合：{out:?} vs {expected:?}"
        );
    }
}
