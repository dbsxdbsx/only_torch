/*
 * @Author       : 老董
 * @Description  : MultiHeadAttention 层单元测试
 */

use crate::nn::{Graph, GraphError, Module, MultiHeadAttention, VarLossOps, VarReduceOps};
use crate::tensor::Tensor;

/// MultiHeadAttention 前向传播形状测试
#[test]
fn test_attention_forward_shape() -> Result<(), GraphError> {
    let graph = Graph::new();

    // embed_dim=8, num_heads=2 → head_dim=4
    let attn = MultiHeadAttention::new(&graph, 8, 2, "attn")?;

    // self-attention: x [N=2, T=3, D=8]
    let x = graph.input(&Tensor::ones(&[2, 3, 8]))?;
    let y = attn.forward(&x, &x, &x);
    y.forward()?;

    let output = y.value()?.unwrap();
    assert_eq!(output.shape(), &[2, 3, 8]);

    Ok(())
}

/// MultiHeadAttention 参数数量
#[test]
fn test_attention_parameters() -> Result<(), GraphError> {
    let graph = Graph::new();

    let attn = MultiHeadAttention::new(&graph, 16, 4, "attn")?;
    let params = attn.parameters();

    // 4 个 Linear 层，每个有 weight + bias = 2 参数
    assert_eq!(params.len(), 8);
    assert_eq!(attn.embed_dim(), 16);
    assert_eq!(attn.num_heads(), 4);

    Ok(())
}

/// MultiHeadAttention 反向传播
#[test]
fn test_attention_backward() -> Result<(), GraphError> {
    let graph = Graph::new();

    let attn = MultiHeadAttention::new(&graph, 8, 2, "attn")?;

    let x = graph.input(&Tensor::ones(&[1, 2, 8]))?;
    let y = attn.forward(&x, &x, &x);

    let target = graph.input(&Tensor::zeros(&[1, 2, 8]))?;
    let loss = y.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val >= 0.0);

    // 所有参数应有梯度
    for (i, p) in attn.parameters().iter().enumerate() {
        let grad = p.grad()?;
        assert!(grad.is_some(), "参数 {i} 应有梯度");
    }

    Ok(())
}

/// Cross-Attention: query 和 key/value 序列长度不同
#[test]
fn test_cross_attention_shape() -> Result<(), GraphError> {
    let graph = Graph::new();

    let attn = MultiHeadAttention::new(&graph, 8, 2, "attn")?;

    // query: [1, 3, 8], key/value: [1, 5, 8]
    let q = graph.input(&Tensor::ones(&[1, 3, 8]))?;
    let kv = graph.input(&Tensor::ones(&[1, 5, 8]))?;

    let y = attn.forward(&q, &kv, &kv);
    y.forward()?;

    let output = y.value()?.unwrap();
    // 输出应为 [1, T_q=3, D=8]
    assert_eq!(output.shape(), &[1, 3, 8]);

    Ok(())
}

/// 注意力输出是否会改变（非常量映射）
#[test]
fn test_attention_non_trivial() -> Result<(), GraphError> {
    let graph = Graph::new();

    let attn = MultiHeadAttention::new(&graph, 4, 2, "attn")?;

    // 不同的输入应产生不同的输出
    let x1 = graph.input(&Tensor::ones(&[1, 2, 4]))?;
    let y1 = attn.forward(&x1, &x1, &x1);
    y1.forward()?;
    let out1 = y1.value()?.unwrap().clone();

    // 在同一图中重新计算（不同输入）
    let x2_data: Vec<f32> = (1..=8).map(|i| i as f32 * 0.1).collect();
    let x2 = graph.input(&Tensor::new(&x2_data, &[1, 2, 4]))?;
    let y2 = attn.forward(&x2, &x2, &x2);
    y2.forward()?;
    let out2 = y2.value()?.unwrap();

    // 两次输出应不同
    let diff: f32 = out1
        .data_as_slice()
        .iter()
        .zip(out2.data_as_slice().iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 0.01, "不同输入应产生不同输出");

    Ok(())
}

// ==================== forward_masked 三个 mask 测试 ====================

/// 全 1 mask（[T_q, T_k] shape）下 forward_masked 与 forward 输出一致
#[test]
fn test_forward_masked_with_all_ones_matches_forward() -> Result<(), GraphError> {
    let graph = Graph::new();
    let attn = MultiHeadAttention::new(&graph, 8, 2, "attn")?;

    let x = graph.input(&Tensor::ones(&[1, 3, 8]))?;

    // 无 mask 路径
    let y1 = attn.forward(&x, &x, &x);
    let out1 = y1.value()?.unwrap().clone();

    // 全 1 mask 路径，应数值一致
    let mask = graph.input(&Tensor::ones(&[3, 3]))?;
    let y2 = attn.forward_masked(&x, &x, &x, &mask);
    let out2 = y2.value()?.unwrap();

    let max_diff = out1
        .data_as_slice()
        .iter()
        .zip(out2.data_as_slice().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "全 1 mask 应与无 mask 输出一致，max diff = {max_diff}"
    );

    Ok(())
}

/// causal mask：未来 token 不应影响当前 token 的输出
///
/// 验证方法：固定 head/feature 参数，比较 t=0 位置在 (1) 完整序列 (2) 仅有 t=0 的序列上的输出
/// 在 causal mask 下，t=0 位置只看自己，所以两种情况输出应一致。
#[test]
fn test_forward_masked_with_causal_mask() -> Result<(), GraphError> {
    let graph = Graph::new();
    let attn = MultiHeadAttention::new(&graph, 4, 2, "attn")?;

    // 用一个递增的非平凡输入，避免对称性掩盖问题
    let full_data: Vec<f32> = (1..=12).map(|i| i as f32 * 0.1).collect();
    let x_full = graph.input(&Tensor::new(&full_data, &[1, 3, 4]))?;
    let mask_full = MultiHeadAttention::causal_mask(&graph, 3)?;
    let y_full = attn.forward_masked(&x_full, &x_full, &x_full, &mask_full);
    let out_full = y_full.value()?.unwrap().clone();

    // 取 t=0 位置在完整序列上的输出
    let row_full = &out_full.data_as_slice()[0..4];

    // 仅 t=0 的 1×1×4 序列：causal mask 下 t=0 也只看 t=0，输出应一致
    let only0_data = &full_data[0..4];
    let x_only = graph.input(&Tensor::new(only0_data, &[1, 1, 4]))?;
    let mask_only = MultiHeadAttention::causal_mask(&graph, 1)?;
    let y_only = attn.forward_masked(&x_only, &x_only, &x_only, &mask_only);
    let out_only = y_only.value()?.unwrap();
    let row_only = out_only.data_as_slice();

    let max_diff = row_full
        .iter()
        .zip(row_only.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "causal mask 下 t=0 输出不应被未来 token 影响，max diff = {max_diff}"
    );

    Ok(())
}

/// padding mask：被屏蔽位置在 attention 权重上接近 0
///
/// 验证方法：构造 [N, 1, T_k] padding mask 并 broadcast 到 [N, T_q, T_k]，
/// 然后与全 1 mask 对比：屏蔽位置应让该位置上的 V 几乎不贡献到输出。
#[test]
fn test_forward_masked_with_padding_mask() -> Result<(), GraphError> {
    let graph = Graph::new();
    let attn = MultiHeadAttention::new(&graph, 4, 2, "attn")?;

    // 一个 batch，T=4，第 3、4 位置应被屏蔽（lengths=2）
    let x_data: Vec<f32> = (1..=16).map(|i| i as f32 * 0.1).collect();
    let x = graph.input(&Tensor::new(&x_data, &[1, 4, 4]))?;

    // 显式构造 [1, 4, 4] mask（所有 query 位置只能看 key 的前 2 列）
    let mut mask_data = vec![0.0f32; 4 * 4];
    for q in 0..4 {
        for k in 0..2 {
            mask_data[q * 4 + k] = 1.0;
        }
    }
    let mask = graph.input(&Tensor::new(&mask_data, &[1, 4, 4]))?;
    let y_masked = attn.forward_masked(&x, &x, &x, &mask);
    let out_masked = y_masked.value()?.unwrap().clone();

    // 对比：全 1 mask（不屏蔽）
    let mask_full = graph.input(&Tensor::ones(&[1, 4, 4]))?;
    let y_full = attn.forward_masked(&x, &x, &x, &mask_full);
    let out_full = y_full.value()?.unwrap();

    // 两者输出应有显著差异（屏蔽生效）
    let total_diff: f32 = out_masked
        .data_as_slice()
        .iter()
        .zip(out_full.data_as_slice().iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        total_diff > 1e-3,
        "padding mask 应让输出明显改变（区分前 2 / 全部），total diff = {total_diff}"
    );

    // 同时验证：屏蔽 mask 下，输出在数值上"等价于只看前 2 个位置"
    // 即与一个长度=2 的子序列做无 mask self-attention 的 t=0 输出应一致
    let x_short = graph.input(&Tensor::new(&x_data[0..8], &[1, 2, 4]))?;
    let y_short = attn.forward(&x_short, &x_short, &x_short);
    let out_short = y_short.value()?.unwrap();
    let head_short = &out_short.data_as_slice()[0..4]; // t=0 位置

    let head_masked = &out_masked.data_as_slice()[0..4];
    let max_diff = head_short
        .iter()
        .zip(head_masked.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "屏蔽前 2 个位置等价于子序列结果，max diff = {max_diff}"
    );

    Ok(())
}

/// causal_mask 工具方法生成正确的下三角形状
#[test]
fn test_causal_mask_shape_and_values() -> Result<(), GraphError> {
    let graph = Graph::new();
    let mask = MultiHeadAttention::causal_mask(&graph, 4)?;
    let v = mask.value()?.unwrap();
    let arr = v.data_as_slice();
    assert_eq!(v.shape(), &[4, 4]);
    // 下三角 1
    for i in 0..4 {
        for j in 0..=i {
            assert_eq!(arr[i * 4 + j], 1.0, "下三角 ({i},{j}) 应为 1");
        }
        for j in (i + 1)..4 {
            assert_eq!(arr[i * 4 + j], 0.0, "上三角 ({i},{j}) 应为 0");
        }
    }
    Ok(())
}

/// padding_mask 工具方法生成正确的 [N, 1, max_len]
#[test]
fn test_padding_mask_shape_and_values() -> Result<(), GraphError> {
    let graph = Graph::new();
    let lengths = vec![3, 1, 5];
    let mask = MultiHeadAttention::padding_mask(&graph, &lengths, 5)?;
    let v = mask.value()?.unwrap();
    assert_eq!(v.shape(), &[3, 1, 5]);
    let arr = v.data_as_slice();
    let expected: &[f32] = &[
        1.0, 1.0, 1.0, 0.0, 0.0, // 第 0 行 lengths=3
        1.0, 0.0, 0.0, 0.0, 0.0, // 第 1 行 lengths=1
        1.0, 1.0, 1.0, 1.0, 1.0, // 第 2 行 lengths=5
    ];
    for (i, &exp) in expected.iter().enumerate() {
        assert_eq!(arr[i], exp, "padding_mask[{i}] 应为 {exp}");
    }
    Ok(())
}

/// 验证 [N, T_q, T_k] mask 在 per-head loop 中按 batch_idx 正确寻址
///
/// 构造 N=2 的 batch，给两个 batch 分别用不同 mask（一个全 1 一个屏蔽尾部），
/// 验证两个 batch 的输出确实不同。
#[test]
fn test_forward_masked_per_batch_addressing() -> Result<(), GraphError> {
    let graph = Graph::new();
    let attn = MultiHeadAttention::new(&graph, 4, 2, "attn")?;

    // 两个 batch 用同样的输入数据
    let single = (1..=12).map(|i| i as f32 * 0.1).collect::<Vec<_>>();
    let mut x_data = Vec::with_capacity(24);
    x_data.extend_from_slice(&single);
    x_data.extend_from_slice(&single);
    let x = graph.input(&Tensor::new(&x_data, &[2, 3, 4]))?;

    // batch 0：全 1 mask（不屏蔽）
    // batch 1：尾部屏蔽（lengths=1）
    let mut mask_data = vec![0.0f32; 2 * 3 * 3];
    for q in 0..3 {
        for k in 0..3 {
            mask_data[q * 3 + k] = 1.0;
        }
    }
    for q in 0..3 {
        let base = 9 + q * 3;
        mask_data[base] = 1.0;
    }
    let mask = graph.input(&Tensor::new(&mask_data, &[2, 3, 3]))?;

    let y = attn.forward_masked(&x, &x, &x, &mask);
    let out = y.value()?.unwrap();
    assert_eq!(out.shape(), &[2, 3, 4]);

    let arr = out.data_as_slice();
    let batch0 = &arr[0..12];
    let batch1 = &arr[12..24];
    let diff: f32 = batch0
        .iter()
        .zip(batch1.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-3,
        "[N, T_q, T_k] mask 应让两个 batch 的输出不同，diff = {diff}"
    );

    // 同时确保有限性（mask 实现没让 softmax 出 NaN/Inf）
    let any_nan = arr.iter().any(|x| x.is_nan() || x.is_infinite());
    assert!(!any_nan, "输出不应含 NaN/Inf");

    let _ = VarReduceOps::sum_axis(&y, 0); // 仅检查 trait 可调
    Ok(())
}
