/*
 * @Author       : 老董
 * @Description  : MultiHeadAttention 层单元测试
 */

use crate::nn::{Graph, GraphError, Module, MultiHeadAttention, VarLossOps};
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
