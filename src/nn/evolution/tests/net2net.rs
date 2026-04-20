//! Net2Net 集成测试：验证在真实 Genome 上 resize + Net2Net + build + restore 后
//! 前向输出与扩宽前函数等价。

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::nn::evolution::gene::*;
use crate::nn::evolution::net2net::apply_widen_to_snapshots;
use crate::nn::evolution::node_ops::{node_main_path, resize_linear_out, NodeBlockKind};
use crate::tensor::Tensor;

/// 构造 NodeLevel 基因组：Input(d_in) → Linear(h) → Linear(d_out)（无激活，便于函数等价验证）
fn linear_mlp_nodegenome(d_in: usize, h: usize, d_out: usize) -> NetworkGenome {
    let mut g = NetworkGenome::minimal(d_in, d_out);
    let inn_hidden = g.next_innovation_number();
    g.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: inn_hidden,
            layer_config: LayerConfig::Linear { out_features: h },
            enabled: true,
        },
    );
    g.migrate_to_node_level().expect("迁移 NodeLevel 失败");
    g
}

/// 前向传播并返回输出张量
fn forward_value(build: &super::super::builder::BuildResult, x: &Tensor) -> Tensor {
    build.input.set_value(x).unwrap();
    build.graph.forward(&build.output).unwrap();
    build.output.value().unwrap().unwrap()
}

#[test]
fn test_net2net_linear_widen_end_to_end_function_preserving() {
    // 1. Linear(4) → Linear(1)
    let mut genome = linear_mlp_nodegenome(2, 4, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build1 = genome.build(&mut rng).expect("build1 失败");

    let x = Tensor::new(&[0.7, -0.3], &[1, 2]);
    let y_old = forward_value(&build1, &x);

    // 2. 捕获权重
    genome.capture_weights(&build1).expect("capture 失败");

    // 3. 找到隐藏 Linear(4) 块，resize 到 7
    let blocks_before = node_main_path(&genome);
    let hidden_block = blocks_before
        .iter()
        .find(|b| matches!(&b.kind, NodeBlockKind::Linear { out_features: 4 }))
        .cloned()
        .expect("应找到 Linear(4) 块");

    resize_linear_out(&mut genome, &hidden_block, 7).expect("resize 失败");

    // 4. 对快照应用 Net2Net
    let applied = apply_widen_to_snapshots(&mut genome, &hidden_block, 4, 7, &mut rng)
        .expect("net2net 内部错误");
    assert!(applied, "Net2Net 应成功应用于 Linear→Linear 路径");

    // 5. 重新构建 + 恢复权重
    let build2 = genome.build(&mut rng).expect("build2 失败");
    let report = genome.restore_weights(&build2).expect("restore 失败");
    assert_eq!(
        report.reinitialized, 0,
        "Net2Net 生效后不应有被重新初始化的参数，实际 {:?}",
        report
    );
    assert!(
        report.partially_inherited == 0,
        "Net2Net 生效后不应走部分继承路径，实际 {:?}",
        report
    );

    // 6. 再次前向传播，输出应与扩宽前等价
    let y_new = forward_value(&build2, &x);
    assert_eq!(y_new.shape(), y_old.shape());
    let diff: f32 = y_old
        .to_vec()
        .iter()
        .zip(y_new.to_vec().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        diff < 1e-4,
        "Linear widen 端到端非函数保持: max |Δ| = {diff}"
    );
}

#[test]
fn test_net2net_rnn_widen_end_to_end_function_preserving() {
    // RNN(hidden=4) → Linear(1)；seq_len=5
    let mut genome = NetworkGenome::minimal_sequential(2, 1);
    genome.seq_len = Some(5);
    genome.migrate_to_node_level().expect("迁移失败");

    let mut rng = StdRng::seed_from_u64(7);
    let build1 = genome.build(&mut rng).expect("build1 失败");

    let x = Tensor::new(
        &[
            0.1, -0.2, 0.3, -0.4, 0.5, -0.1, 0.2, -0.3, 0.4, -0.5,
        ],
        &[1, 5, 2],
    );
    let y_old = forward_value(&build1, &x);

    genome.capture_weights(&build1).expect("capture 失败");

    // 找 RNN 块
    let blocks = node_main_path(&genome);
    let rnn_block = blocks
        .iter()
        .find(|b| matches!(&b.kind, NodeBlockKind::Rnn { .. }))
        .cloned()
        .expect("应有 RNN 块");
    let old_h = rnn_block.kind.current_size().unwrap();

    // 调用底层 resize_recurrent_out 将 hidden 从 4 → 7
    use crate::nn::evolution::node_ops::resize_recurrent_out;
    let new_h = old_h + 3;
    resize_recurrent_out(&mut genome, &rnn_block, new_h).expect("resize 失败");

    // Net2Net
    let applied = apply_widen_to_snapshots(&mut genome, &rnn_block, old_h, new_h, &mut rng)
        .expect("net2net 内部错误");
    assert!(applied, "Net2Net 应成功应用于 RNN→Linear");

    // 重建 + 恢复
    let build2 = genome.build(&mut rng).expect("build2 失败");
    let report = genome.restore_weights(&build2).expect("restore 失败");
    assert_eq!(report.reinitialized, 0, "不应有参数被重新初始化，实际 {:?}", report);
    assert_eq!(
        report.partially_inherited, 0,
        "不应走部分继承路径，实际 {:?}",
        report
    );

    let y_new = forward_value(&build2, &x);
    assert_eq!(y_new.shape(), y_old.shape());
    let diff: f32 = y_old
        .to_vec()
        .iter()
        .zip(y_new.to_vec().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        diff < 1e-4,
        "RNN widen 端到端非函数保持: max |Δ| = {diff}"
    );
}

#[test]
fn test_net2net_widen_mapping_identity_when_new_eq_old() {
    // new_size == old_size 时应退出（返回 false），不修改快照
    let mut genome = linear_mlp_nodegenome(2, 4, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();
    genome.capture_weights(&build).unwrap();

    let blocks = node_main_path(&genome);
    let hidden = blocks
        .iter()
        .find(|b| matches!(&b.kind, NodeBlockKind::Linear { out_features: 4 }))
        .cloned()
        .unwrap();

    // 快照备份用于比对
    let snap_before: std::collections::HashMap<u64, Tensor> =
        genome.node_weight_snapshots().clone();

    let applied = apply_widen_to_snapshots(&mut genome, &hidden, 4, 4, &mut rng).unwrap();
    assert!(!applied, "new == old 时应返回 false");

    // 快照保持不变
    let snap_after = genome.node_weight_snapshots();
    for (k, v) in &snap_before {
        let a = snap_after.get(k).expect("应存在");
        assert_eq!(a.shape(), v.shape());
        assert_eq!(a.to_vec(), v.to_vec());
    }
}
