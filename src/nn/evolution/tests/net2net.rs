//! Net2Net 单元 + 集成测试。
//!
//! - 前半部分验证原语：`widening_mapping`、`counts_of`、`gather_along_axis`
//!   （含 scaled 变体与 Flatten 跨域）以及 Linear / RNN / Conv2d→Flatten→Linear
//!   的手工搭建前向函数保持性（不走 genome build）。
//! - 后半部分验证端到端：构造真实 NodeLevel genome，走
//!   `resize → apply_widen_to_snapshots → restore_weights → forward`，
//!   断言扩宽前后输出函数等价。

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::nn::evolution::gene::*;
use crate::nn::evolution::net2net::{
    apply_widen_to_snapshots, counts_of, gather_along_axis, gather_along_axis_scaled,
    gather_linear_in_with_flatten, widening_mapping,
};
use crate::nn::evolution::node_ops::{node_main_path, resize_linear_out, NodeBlockKind};
use crate::tensor::Tensor;

fn seeded_rng() -> StdRng {
    StdRng::seed_from_u64(12345)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    a.to_vec()
        .iter()
        .zip(b.to_vec().iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

// ==================== 原语单元测试 ====================

#[test]
fn test_widening_mapping_identity_prefix() {
    let mut r = seeded_rng();
    let m = widening_mapping(4, 7, &mut r);
    assert_eq!(m.len(), 7);
    for i in 0..4 {
        assert_eq!(m[i], i, "prefix should be identity");
    }
    for j in 4..7 {
        assert!(m[j] < 4, "suffix indices must be in [0, 4)");
    }
}

#[test]
fn test_widening_mapping_no_growth() {
    let mut r = seeded_rng();
    let m = widening_mapping(5, 5, &mut r);
    assert_eq!(m, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_counts_of_basic() {
    let m = vec![0, 1, 2, 0, 1, 0];
    let c = counts_of(&m, 3);
    assert_eq!(c, vec![3, 2, 1]);
}

#[test]
fn test_gather_along_axis_identity() {
    let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let g = gather_along_axis(&t, 1, &[0, 1, 2]);
    assert_eq!(g.shape(), &[2, 3]);
    assert_eq!(g.to_vec(), t.to_vec());
}

#[test]
fn test_gather_along_axis_duplicate() {
    let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let g = gather_along_axis(&t, 1, &[0, 1, 2, 0]);
    assert_eq!(g.shape(), &[2, 4]);
    assert_eq!(g.to_vec(), vec![1.0, 2.0, 3.0, 1.0, 4.0, 5.0, 6.0, 4.0]);
}

#[test]
fn test_gather_scaled_preserves_row_sum() {
    let v = Tensor::new(&[2.0, 4.0, 6.0], &[1, 3]);
    let mapping = vec![0, 1, 2, 0, 1];
    let counts = counts_of(&mapping, 3);
    assert_eq!(counts, vec![2, 2, 1]);
    let g = gather_along_axis_scaled(&v, 1, &mapping, &counts);
    let data = g.to_vec();
    for (x, y) in data.iter().zip(&[1.0, 2.0, 6.0, 1.0, 2.0]) {
        assert!((x - y).abs() < 1e-6, "got {x}, expected {y}");
    }
    let mut sums = vec![0.0_f32; 3];
    for (i, v) in data.iter().enumerate() {
        sums[mapping[i]] += *v;
    }
    assert!((sums[0] - 2.0).abs() < 1e-6);
    assert!((sums[1] - 4.0).abs() < 1e-6);
    assert!((sums[2] - 6.0).abs() < 1e-6);
}

#[test]
fn test_flatten_stride_expand() {
    let w = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[6, 2],
    );
    let mapping = vec![0, 1, 0];
    let counts = counts_of(&mapping, 2);
    let g = gather_linear_in_with_flatten(&w, &mapping, &counts, 3);
    assert_eq!(g.shape(), &[9, 2]);
    let expected = vec![
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
    ];
    for (x, y) in g.to_vec().iter().zip(&expected) {
        assert!((x - y).abs() < 1e-6, "got {x}, expected {y}");
    }
}

// ==================== 函数保持性（手工搭建前向）====================

#[test]
fn test_linear_widen_preserves_function_no_activation() {
    let (h_old, h_new, d_in, d_out) = (3usize, 5usize, 2usize, 1usize);
    let mut r = seeded_rng();
    let mapping = widening_mapping(h_old, h_new, &mut r);
    let counts = counts_of(&mapping, h_old);

    let w1 = Tensor::new(&[0.1, 0.2, 0.3, -0.4, 0.5, -0.6], &[d_in, h_old]);
    let b1 = Tensor::new(&[0.01, 0.02, 0.03], &[1, h_old]);
    let w2 = Tensor::new(&[1.0, -0.5, 0.25], &[h_old, d_out]);
    let b2 = Tensor::new(&[0.5], &[1, d_out]);

    let w1_new = gather_along_axis(&w1, 1, &mapping);
    let b1_new = gather_along_axis(&b1, 1, &mapping);
    let w2_new = gather_along_axis_scaled(&w2, 0, &mapping, &counts);

    let x = Tensor::new(&[0.7, -0.3], &[1, d_in]);
    let h_old_vec = x.mat_mul(&w1) + b1.clone();
    let y_old = h_old_vec.mat_mul(&w2) + b2.clone();

    let h_new_vec = x.mat_mul(&w1_new) + b1_new;
    let y_new = h_new_vec.mat_mul(&w2_new) + b2;

    assert_eq!(y_new.shape(), y_old.shape());
    let diff = max_abs_diff(&y_new, &y_old);
    assert!(diff < 1e-5, "Linear widen 非函数保持: max diff = {diff}");
}

#[test]
fn test_linear_bn_linear_widen_preserves_function() {
    let (h_old, h_new, d_in, d_out) = (3usize, 6usize, 2usize, 1usize);
    let mut r = seeded_rng();
    let mapping = widening_mapping(h_old, h_new, &mut r);
    let counts = counts_of(&mapping, h_old);

    let w1 = Tensor::new(&[0.1, 0.2, 0.3, -0.4, 0.5, -0.6], &[d_in, h_old]);
    let b1 = Tensor::new(&[0.01, 0.02, 0.03], &[1, h_old]);
    let gamma = Tensor::new(&[1.1, 0.9, 1.2], &[1, h_old]);
    let beta = Tensor::new(&[0.05, -0.1, 0.0], &[1, h_old]);
    let w2 = Tensor::new(&[1.0, -0.5, 0.25], &[h_old, d_out]);
    let b2 = Tensor::new(&[0.5], &[1, d_out]);

    let w1_new = gather_along_axis(&w1, 1, &mapping);
    let b1_new = gather_along_axis(&b1, 1, &mapping);
    let gamma_new = gather_along_axis(&gamma, 1, &mapping);
    let beta_new = gather_along_axis(&beta, 1, &mapping);
    let w2_new = gather_along_axis_scaled(&w2, 0, &mapping, &counts);

    let x = Tensor::new(&[0.7, -0.3], &[1, d_in]);

    let h_old_vec = x.mat_mul(&w1) + b1.clone();
    let h_old_bn = &h_old_vec * &gamma + beta.clone();
    let y_old = h_old_bn.mat_mul(&w2) + b2.clone();

    let h_new_vec = x.mat_mul(&w1_new) + b1_new;
    let h_new_bn = &h_new_vec * &gamma_new + beta_new;
    let y_new = h_new_bn.mat_mul(&w2_new) + b2;

    let diff = max_abs_diff(&y_new, &y_old);
    assert!(
        diff < 1e-5,
        "Linear->BN->Linear widen 非函数保持: max diff = {diff}"
    );
}

#[test]
fn test_rnn_hidden_widen_preserves_function_single_step() {
    let (h_old, h_new, d_in) = (3usize, 5usize, 2usize);
    let mut r = seeded_rng();
    let mapping = widening_mapping(h_old, h_new, &mut r);
    let counts = counts_of(&mapping, h_old);

    let w_ix = Tensor::new(&[0.1, 0.2, 0.3, -0.4, 0.5, -0.6], &[d_in, h_old]);
    let w_hx = Tensor::new(
        &[0.01, 0.02, 0.03, 0.04, -0.05, 0.06, -0.07, 0.08, 0.09],
        &[h_old, h_old],
    );
    let b_x = Tensor::new(&[0.1, -0.1, 0.0], &[1, h_old]);

    let w_ix_new = gather_along_axis(&w_ix, 1, &mapping);
    let w_hx_tmp = gather_along_axis_scaled(&w_hx, 0, &mapping, &counts);
    let w_hx_new = gather_along_axis(&w_hx_tmp, 1, &mapping);
    let b_x_new = gather_along_axis(&b_x, 1, &mapping);

    let x = Tensor::new(&[0.7, -0.3], &[1, d_in]);
    let h_prev = Tensor::new(&[0.2, -0.1, 0.4], &[1, h_old]);
    let h_prev_new = gather_along_axis(&h_prev, 1, &mapping);

    let pre_old = x.mat_mul(&w_ix) + h_prev.mat_mul(&w_hx) + b_x.clone();
    let pre_new = x.mat_mul(&w_ix_new) + h_prev_new.mat_mul(&w_hx_new) + b_x_new;

    let old_slice = pre_old.to_vec();
    let new_slice = pre_new.to_vec();
    for (i_new, &i_old) in mapping.iter().enumerate() {
        let a = new_slice[i_new];
        let b = old_slice[i_old];
        assert!(
            (a - b).abs() < 1e-5,
            "RNN hidden widen mismatch at new idx {i_new} (src {i_old}): got {a}, expected {b}"
        );
    }
}

#[test]
fn test_conv2d_flatten_linear_widen_preserves_function() {
    let (c_old, c_new, h, w, d_out) = (2usize, 4usize, 2usize, 2usize, 1usize);
    let mut r = seeded_rng();
    let mapping = widening_mapping(c_old, c_new, &mut r);
    let counts = counts_of(&mapping, c_old);

    let feat = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, c_old, h, w]);
    let feat_new = gather_along_axis(&feat, 1, &mapping);
    assert_eq!(feat_new.shape(), &[1, c_new, h, w]);

    let feat_old_flat = feat.reshape(&[1, c_old * h * w]);
    let feat_new_flat = feat_new.reshape(&[1, c_new * h * w]);

    let w_down = Tensor::new(
        &[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
        &[c_old * h * w, d_out],
    );
    let w_down_new = gather_linear_in_with_flatten(&w_down, &mapping, &counts, h * w);
    assert_eq!(w_down_new.shape(), &[c_new * h * w, d_out]);

    let y_old = feat_old_flat.mat_mul(&w_down);
    let y_new = feat_new_flat.mat_mul(&w_down_new);
    let diff = max_abs_diff(&y_new, &y_old);
    assert!(
        diff < 1e-5,
        "Conv2d->Flatten->Linear widen 非函数保持: max diff = {diff}"
    );
}

// ==================== 端到端集成测试 ====================

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
