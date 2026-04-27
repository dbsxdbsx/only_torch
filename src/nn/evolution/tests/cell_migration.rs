//! Cell 类型迁移（MutateCellType 权重搬运）测试。
//!
//! - 原语测试：直接调用 `migrate_cell_weights` 验证目标门的权重来自正确的
//!   旧门；饱和门的 bias 接近 ±6。
//! - 集成测试：真实 NodeLevel 基因组上走完
//!   capture → MutateCellType → rebuild → restore 流程，断言迁移写入的参数
//!   被 `restore_weights` 正确复原（inherited ≥ 新 cell 的参数张量数）。

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::nn::descriptor::NodeTypeDescriptor as NT;
use crate::nn::evolution::cell_migration::{CellKind, migrate_cell_weights};
use crate::nn::evolution::gene::*;
use crate::nn::evolution::mutation::{MutateCellTypeMutation, Mutation, SizeConstraints};
use crate::nn::evolution::node_ops::{NodeBlockKind, node_main_path};
use crate::tensor::Tensor;

fn is_cell_node(n: &crate::nn::evolution::node_gene::NodeGene) -> bool {
    matches!(
        n.node_type,
        NT::CellRnn { .. } | NT::CellLstm { .. } | NT::CellGru { .. }
    )
}

// ==================== 原语单元测试 ====================

/// 构造一组可识别的旧快照：`W_ih` 全 `gate_val`，`W_hh` 全 `gate_val + 0.5`，
/// `b` 全 `gate_val + 1.0`，方便断言目标门来源。
fn make_gate_snapshot(in_dim: usize, hidden: usize, gate_val: f32) -> Vec<Option<Tensor>> {
    vec![
        Some(Tensor::full(gate_val, &[in_dim, hidden])),
        Some(Tensor::full(gate_val + 0.5, &[hidden, hidden])),
        Some(Tensor::full(gate_val + 1.0, &[1, hidden])),
    ]
}

/// 将多个门的快照拼接成完整列表。
fn concat_gates(gates: Vec<Vec<Option<Tensor>>>) -> Vec<Option<Tensor>> {
    gates.into_iter().flatten().collect()
}

#[test]
fn test_rnn_to_lstm_copies_rnn_into_g_gate_and_saturates_others() {
    let (in_dim, hidden) = (3usize, 4usize);
    let old = make_gate_snapshot(in_dim, hidden, 0.7);
    let new_ids: Vec<u64> = (100..112).collect(); // 12 个参数
    let out = migrate_cell_weights(CellKind::Rnn, &old, CellKind::Lstm, &new_ids, hidden)
        .expect("迁移应成功");

    // 门序 [i(0..3), f(3..6), g(6..9), o(9..12)]
    // g 门应继承 RNN 的权重
    let g_w_ih = out.get(&new_ids[6]).expect("W_ig");
    let g_w_hh = out.get(&new_ids[7]).expect("W_hg");
    let g_b = out.get(&new_ids[8]).expect("b_g");
    assert_eq!(g_w_ih.shape(), &[in_dim, hidden]);
    assert!((g_w_ih.to_vec()[0] - 0.7).abs() < 1e-6);
    assert!((g_w_hh.to_vec()[0] - 1.2).abs() < 1e-6);
    assert!((g_b.to_vec()[0] - 1.7).abs() < 1e-6);

    // i 门应饱和高（bias ≈ +6，W 全零）
    let i_w_ih = out.get(&new_ids[0]).expect("W_ii");
    let i_b = out.get(&new_ids[2]).expect("b_i");
    assert!(i_w_ih.to_vec().iter().all(|&v| v.abs() < 1e-6));
    assert!(i_b.to_vec().iter().all(|&v| (v - 6.0).abs() < 1e-3));

    // f 门应饱和低（bias ≈ -6）
    let f_b = out.get(&new_ids[5]).expect("b_f");
    assert!(f_b.to_vec().iter().all(|&v| (v + 6.0).abs() < 1e-3));

    // o 门应饱和高
    let o_b = out.get(&new_ids[11]).expect("b_o");
    assert!(o_b.to_vec().iter().all(|&v| (v - 6.0).abs() < 1e-3));
}

#[test]
fn test_lstm_to_rnn_extracts_g_gate() {
    let (in_dim, hidden) = (3usize, 4usize);
    // i=1.0, f=2.0, g=3.0, o=4.0
    let old = concat_gates(vec![
        make_gate_snapshot(in_dim, hidden, 1.0),
        make_gate_snapshot(in_dim, hidden, 2.0),
        make_gate_snapshot(in_dim, hidden, 3.0),
        make_gate_snapshot(in_dim, hidden, 4.0),
    ]);
    let new_ids: Vec<u64> = vec![200, 201, 202];
    let out = migrate_cell_weights(CellKind::Lstm, &old, CellKind::Rnn, &new_ids, hidden)
        .expect("迁移应成功");

    // 新 RNN 应取 g 门（gate_val = 3.0）
    assert!((out[&new_ids[0]].to_vec()[0] - 3.0).abs() < 1e-6);
    assert!((out[&new_ids[1]].to_vec()[0] - 3.5).abs() < 1e-6);
    assert!((out[&new_ids[2]].to_vec()[0] - 4.0).abs() < 1e-6);
}

#[test]
fn test_gru_to_rnn_extracts_n_gate() {
    let (in_dim, hidden) = (3usize, 4usize);
    // r=1.0, z=2.0, n=3.0
    let old = concat_gates(vec![
        make_gate_snapshot(in_dim, hidden, 1.0),
        make_gate_snapshot(in_dim, hidden, 2.0),
        make_gate_snapshot(in_dim, hidden, 3.0),
    ]);
    let new_ids: Vec<u64> = vec![300, 301, 302];
    let out = migrate_cell_weights(CellKind::Gru, &old, CellKind::Rnn, &new_ids, hidden)
        .expect("迁移应成功");
    assert!((out[&new_ids[0]].to_vec()[0] - 3.0).abs() < 1e-6);
}

#[test]
fn test_rnn_to_gru_copies_rnn_into_n_and_saturates_r_z() {
    let (in_dim, hidden) = (3usize, 4usize);
    let old = make_gate_snapshot(in_dim, hidden, 0.9);
    let new_ids: Vec<u64> = (400..409).collect();
    let out = migrate_cell_weights(CellKind::Rnn, &old, CellKind::Gru, &new_ids, hidden)
        .expect("迁移应成功");

    // 门序 [r(0..3), z(3..6), n(6..9)]
    assert!(
        out[&new_ids[2]]
            .to_vec()
            .iter()
            .all(|&v| (v - 6.0).abs() < 1e-3)
    ); // b_r ≈ +6
    assert!(
        out[&new_ids[5]]
            .to_vec()
            .iter()
            .all(|&v| (v + 6.0).abs() < 1e-3)
    ); // b_z ≈ -6

    // n 门继承 RNN
    assert!((out[&new_ids[6]].to_vec()[0] - 0.9).abs() < 1e-6);
    assert!((out[&new_ids[7]].to_vec()[0] - 1.4).abs() < 1e-6);
    assert!((out[&new_ids[8]].to_vec()[0] - 1.9).abs() < 1e-6);
}

#[test]
fn test_lstm_to_gru_remaps_f_to_z_and_g_to_n() {
    let (in_dim, hidden) = (3usize, 4usize);
    let old = concat_gates(vec![
        make_gate_snapshot(in_dim, hidden, 1.0), // i
        make_gate_snapshot(in_dim, hidden, 2.0), // f
        make_gate_snapshot(in_dim, hidden, 3.0), // g
        make_gate_snapshot(in_dim, hidden, 4.0), // o
    ]);
    let new_ids: Vec<u64> = (500..509).collect();
    let out = migrate_cell_weights(CellKind::Lstm, &old, CellKind::Gru, &new_ids, hidden)
        .expect("迁移应成功");

    // r 门应饱和为 1
    assert!(
        out[&new_ids[2]]
            .to_vec()
            .iter()
            .all(|&v| (v - 6.0).abs() < 1e-3)
    );
    // z 门 ← f 门 (2.0)
    assert!((out[&new_ids[3]].to_vec()[0] - 2.0).abs() < 1e-6);
    // n 门 ← g 门 (3.0)
    assert!((out[&new_ids[6]].to_vec()[0] - 3.0).abs() < 1e-6);
}

#[test]
fn test_gru_to_lstm_remaps_z_to_f_and_n_to_g() {
    let (in_dim, hidden) = (3usize, 4usize);
    let old = concat_gates(vec![
        make_gate_snapshot(in_dim, hidden, 1.0), // r
        make_gate_snapshot(in_dim, hidden, 2.0), // z
        make_gate_snapshot(in_dim, hidden, 3.0), // n
    ]);
    let new_ids: Vec<u64> = (600..612).collect();
    let out = migrate_cell_weights(CellKind::Gru, &old, CellKind::Lstm, &new_ids, hidden)
        .expect("迁移应成功");

    // i 门饱和为 1
    assert!(
        out[&new_ids[2]]
            .to_vec()
            .iter()
            .all(|&v| (v - 6.0).abs() < 1e-3)
    );
    // f 门 ← z 门 (2.0)
    assert!((out[&new_ids[3]].to_vec()[0] - 2.0).abs() < 1e-6);
    // g 门 ← n 门 (3.0)
    assert!((out[&new_ids[6]].to_vec()[0] - 3.0).abs() < 1e-6);
    // o 门饱和为 1
    assert!(
        out[&new_ids[11]]
            .to_vec()
            .iter()
            .all(|&v| (v - 6.0).abs() < 1e-3)
    );
}

#[test]
fn test_migrate_cell_weights_refuses_identity() {
    let old = make_gate_snapshot(2, 3, 0.5);
    let ids: Vec<u64> = vec![0, 1, 2];
    let out = migrate_cell_weights(CellKind::Rnn, &old, CellKind::Rnn, &ids, 3);
    assert!(out.is_none(), "同类型迁移应返回 None");
}

#[test]
fn test_migrate_cell_weights_refuses_bad_snapshot_count() {
    let old: Vec<Option<Tensor>> = vec![Some(Tensor::full(1.0, &[2, 3]))]; // 少
    let ids: Vec<u64> = (0..12).collect();
    let out = migrate_cell_weights(CellKind::Rnn, &old, CellKind::Lstm, &ids, 3);
    assert!(out.is_none(), "快照数量不对应返回 None");
}

// ==================== 端到端集成测试 ====================

#[test]
fn test_mutate_cell_type_rnn_to_lstm_migrates_weights_to_g_gate() {
    // 构造 RNN(hidden=4) → Linear(1) 的 NodeLevel 基因组
    let mut genome = NetworkGenome::minimal_sequential(2, 1);
    genome.seq_len = Some(5);
    genome.migrate_to_node_level().unwrap();

    let mut rng = StdRng::seed_from_u64(20260420);
    let build1 = genome.build(&mut rng).unwrap();
    genome.capture_weights(&build1).unwrap();

    // 记录 RNN 三个参数的"标记"值：直接修改快照为常数，便于后续识别
    let blocks = node_main_path(&genome);
    let rnn_block = blocks
        .iter()
        .find(|b| matches!(&b.kind, NodeBlockKind::Rnn { .. }))
        .cloned()
        .expect("应有 RNN 块");

    // 定位 Cell 节点的 3 个参数 id（按 expand_rnn 顺序）
    let rnn_param_ids: Vec<u64> = {
        let cell = genome
            .nodes()
            .iter()
            .find(|n| rnn_block.node_ids.contains(&n.innovation_number) && is_cell_node(n))
            .unwrap();
        cell.parents.iter().skip(1).copied().collect()
    };
    assert_eq!(rnn_param_ids.len(), 3);

    // 把 RNN W_ih/W_hh/b 标记为 11.0/12.0/13.0
    {
        let snaps = genome.node_weight_snapshots_mut();
        let shapes: Vec<Vec<usize>> = rnn_param_ids
            .iter()
            .map(|pid| snaps[pid].shape().to_vec())
            .collect();
        snaps.insert(rnn_param_ids[0], Tensor::full(11.0, &shapes[0]));
        snaps.insert(rnn_param_ids[1], Tensor::full(12.0, &shapes[1]));
        snaps.insert(rnn_param_ids[2], Tensor::full(13.0, &shapes[2]));
    }

    // 强制多次尝试，直到成功切换到 LSTM（50% 几率切到 GRU）
    let constraints = SizeConstraints::default();
    let mut attempts = 0;
    let target_lstm_cell_id = loop {
        attempts += 1;
        assert!(attempts < 64, "64 次尝试仍未切到 LSTM");
        let mut trial = genome.clone();
        let mut r = StdRng::seed_from_u64(attempts);
        MutateCellTypeMutation
            .apply(&mut trial, &constraints, &mut r)
            .unwrap();
        // 检查是否变成了 LSTM
        let new_blocks = node_main_path(&trial);
        if let Some(lstm_block) = new_blocks
            .iter()
            .find(|b| matches!(&b.kind, NodeBlockKind::Lstm { .. }))
        {
            // 成功：替换 genome
            genome = trial;
            break lstm_block.clone();
        }
    };

    // 定位 LSTM 12 个参数 id
    let lstm_param_ids: Vec<u64> = {
        let cell = genome
            .nodes()
            .iter()
            .find(|n| {
                target_lstm_cell_id.node_ids.contains(&n.innovation_number) && is_cell_node(n)
            })
            .unwrap();
        cell.parents.iter().skip(1).copied().collect()
    };
    assert_eq!(lstm_param_ids.len(), 12);

    // 断言 g 门（索引 6/7/8）取自 RNN 的标记值
    let snaps = genome.node_weight_snapshots();
    let g_w_ih = snaps.get(&lstm_param_ids[6]).expect("g 门 W_ih 应有快照");
    let g_w_hh = snaps.get(&lstm_param_ids[7]).expect("g 门 W_hh 应有快照");
    let g_b = snaps.get(&lstm_param_ids[8]).expect("g 门 b 应有快照");
    assert!((g_w_ih.to_vec()[0] - 11.0).abs() < 1e-6);
    assert!((g_w_hh.to_vec()[0] - 12.0).abs() < 1e-6);
    assert!((g_b.to_vec()[0] - 13.0).abs() < 1e-6);

    // f 门 bias ≈ -6, i/o 门 bias ≈ +6
    assert!(
        snaps[&lstm_param_ids[5]]
            .to_vec()
            .iter()
            .all(|&v| (v + 6.0).abs() < 1e-3),
        "LSTM f 门 bias 未饱和"
    );
    assert!(
        snaps[&lstm_param_ids[2]]
            .to_vec()
            .iter()
            .all(|&v| (v - 6.0).abs() < 1e-3),
        "LSTM i 门 bias 未饱和"
    );
    assert!(
        snaps[&lstm_param_ids[11]]
            .to_vec()
            .iter()
            .all(|&v| (v - 6.0).abs() < 1e-3),
        "LSTM o 门 bias 未饱和"
    );

    // 重建 + restore，应全部 inherited（不走部分继承/重新初始化）
    let build2 = genome.build(&mut rng).unwrap();
    let report = genome.restore_weights(&build2).unwrap();
    assert_eq!(
        report.reinitialized, 0,
        "LSTM 全部参数都应从迁移快照继承，实际 report={:?}",
        report
    );
}

#[test]
fn test_mutate_cell_type_preserves_shapes_after_rebuild() {
    // 冒烟：多次切换后 build + restore 不会报错
    let mut genome = NetworkGenome::minimal_sequential(3, 2);
    genome.seq_len = Some(4);
    genome.migrate_to_node_level().unwrap();

    let mut rng = StdRng::seed_from_u64(7);
    let build = genome.build(&mut rng).unwrap();
    genome.capture_weights(&build).unwrap();

    let constraints = SizeConstraints::default();
    for seed in 0..10u64 {
        let mut trial = genome.clone();
        let mut r = StdRng::seed_from_u64(seed);
        if MutateCellTypeMutation
            .apply(&mut trial, &constraints, &mut r)
            .is_ok()
        {
            let rebuilt = trial.build(&mut r).unwrap();
            let report = trial.restore_weights(&rebuilt).unwrap();
            // 总张量数 = inherited + partial + reinit，必须等于新图参数节点数
            let total = report.inherited + report.partially_inherited + report.reinitialized;
            assert_eq!(total, rebuilt.layer_params.len());
        }
    }
}
