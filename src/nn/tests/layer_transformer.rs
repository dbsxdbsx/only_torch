/*
 * @Author       : 老董
 * @Description  : Transformer Encoder 单元测试
 *   覆盖：单层 forward 形状、参数完整、反向梯度流通、mask 透传至底层 MHA、堆叠多层形状一致。
 */

use crate::nn::{
    Graph, GraphError, Module, MultiHeadAttention, TransformerEncoder, TransformerEncoderLayer,
    VarLossOps,
};
use crate::tensor::Tensor;

/// 单层 forward 形状正确
#[test]
fn transformer_layer_forward_shape() -> Result<(), GraphError> {
    let g = Graph::new();
    let layer = TransformerEncoderLayer::new(&g, 16, 4, 32, 0.0, "te")?;
    let x = g.input(&Tensor::ones(&[2, 5, 16]))?;
    let y = layer.forward(&x);
    y.forward()?;
    assert_eq!(y.value()?.unwrap().shape(), &[2, 5, 16]);
    Ok(())
}

/// 参数完整性：LN×2 + MHA(4 个 Linear，每个 weight+bias) + FFN×2(weight+bias)
#[test]
fn transformer_layer_parameters_complete() -> Result<(), GraphError> {
    let g = Graph::new();
    let layer = TransformerEncoderLayer::new(&g, 8, 2, 16, 0.0, "te")?;
    let params = layer.parameters();
    // LN1 (2) + MHA (8) + LN2 (2) + FFN1 (2) + FFN2 (2) = 16
    assert_eq!(params.len(), 16);
    Ok(())
}

/// 反向梯度流通：所有参数都应有梯度
#[test]
fn transformer_layer_backward_flows() -> Result<(), GraphError> {
    let g = Graph::new();
    let layer = TransformerEncoderLayer::new(&g, 8, 2, 16, 0.0, "te")?;
    let x = g.input(&Tensor::ones(&[1, 3, 8]))?;
    let y = layer.forward(&x);
    let target = g.input(&Tensor::zeros(&[1, 3, 8]))?;
    let loss = y.mse_loss(&target)?;
    g.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val >= 0.0);
    for (i, p) in layer.parameters().iter().enumerate() {
        let grad = p.grad()?;
        assert!(grad.is_some(), "参数 {i} 应有梯度");
    }
    Ok(())
}

/// mask 透传：causal mask 下 forward_masked 改变输出
#[test]
fn transformer_layer_mask_changes_output() -> Result<(), GraphError> {
    let g = Graph::new();
    let layer = TransformerEncoderLayer::new(&g, 8, 2, 16, 0.0, "te")?;

    let data: Vec<f32> = (1..=24).map(|i| i as f32 * 0.1).collect();
    let x = g.input(&Tensor::new(&data, &[1, 3, 8]))?;

    let y_full = layer.forward(&x);
    let out_full = y_full.value()?.unwrap().clone();

    let mask = MultiHeadAttention::causal_mask(&g, 3)?;
    let y_masked = layer.forward_masked(&x, &mask);
    let out_masked = y_masked.value()?.unwrap();

    // 注：Pre-LN 的残差路径会显著淡化 attention 内部差异，所以 transformer 整体输出
    // 差异通常远小于 attention 子层（attention 测试中 ~1e-3）。这里只要确认 mask
    // 路径走通即可，差异为 0 才说明 mask 没透传。
    let total_diff: f32 = out_full
        .data_as_slice()
        .iter()
        .zip(out_masked.data_as_slice().iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        total_diff > 1e-7,
        "causal mask 应让 forward_masked 输出与 forward 不同（实际 diff = {total_diff}）"
    );

    Ok(())
}

/// 堆叠多层 TransformerEncoder forward 形状正确
#[test]
fn transformer_encoder_stack_forward_shape() -> Result<(), GraphError> {
    let g = Graph::new();
    let enc = TransformerEncoder::new(&g, 3, 8, 2, 16, 0.0, "enc")?;
    let x = g.input(&Tensor::ones(&[2, 4, 8]))?;
    let y = enc.forward(&x);
    y.forward()?;
    assert_eq!(y.value()?.unwrap().shape(), &[2, 4, 8]);
    assert_eq!(enc.num_layers(), 3);
    Ok(())
}

/// 堆叠 Encoder 的参数 = N × 单层参数数
#[test]
fn transformer_encoder_parameters_match_n_layers() -> Result<(), GraphError> {
    let g = Graph::new();
    let enc = TransformerEncoder::new(&g, 3, 8, 2, 16, 0.0, "enc")?;
    let params = enc.parameters();
    assert_eq!(params.len(), 16 * 3);
    Ok(())
}

/// Encoder forward_masked 透传 mask 至所有层
#[test]
fn transformer_encoder_forward_masked_propagates() -> Result<(), GraphError> {
    let g = Graph::new();
    let enc = TransformerEncoder::new(&g, 2, 8, 2, 16, 0.0, "enc")?;

    let data: Vec<f32> = (1..=24).map(|i| i as f32 * 0.1).collect();
    let x = g.input(&Tensor::new(&data, &[1, 3, 8]))?;

    let y_full = enc.forward(&x);
    let out_full = y_full.value()?.unwrap().clone();

    let mask = MultiHeadAttention::causal_mask(&g, 3)?;
    let y_masked = enc.forward_masked(&x, &mask);
    let out_masked = y_masked.value()?.unwrap();

    let total_diff: f32 = out_full
        .data_as_slice()
        .iter()
        .zip(out_masked.data_as_slice().iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        total_diff > 1e-7,
        "堆叠 encoder 也应将 mask 透传给所有层（实际 diff = {total_diff}）"
    );

    Ok(())
}
