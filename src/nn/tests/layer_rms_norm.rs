/*
 * @Author       : 老董
 * @Description  : RMSNorm 层单元测试
 */

use crate::nn::{Graph, GraphError, Module, RMSNorm, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

/// RMSNorm 默认参数 (gamma=1)
#[test]
fn test_rms_norm_default_params() -> Result<(), GraphError> {
    let graph = Graph::new();

    let rn = RMSNorm::new(&graph, &[4], 1e-5, "rn")?;
    let x = graph.input(&Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4],
        &[2, 4],
    ))?;
    let y = rn.forward(&x);
    y.forward()?;

    let output = y.value()?.unwrap();
    assert_eq!(output.shape(), &[2, 4]);
    // gamma=1 → 等价于纯 RMSNormOp
    assert_abs_diff_eq!(output[[0, 0]], 0.3651, epsilon = 1e-3);

    Ok(())
}

/// RMSNorm 参数数量
#[test]
fn test_rms_norm_parameters() -> Result<(), GraphError> {
    let graph = Graph::new();
    let rn = RMSNorm::new(&graph, &[64], 1e-5, "rn")?;

    let params = rn.parameters();
    assert_eq!(params.len(), 1); // 只有 gamma

    let gamma_val = params[0].value()?.unwrap();
    assert_eq!(gamma_val.shape(), &[1, 64]);
    assert_abs_diff_eq!(gamma_val[[0, 0]], 1.0, epsilon = 1e-6);

    Ok(())
}

/// RMSNorm 反向传播
#[test]
fn test_rms_norm_backward() -> Result<(), GraphError> {
    let graph = Graph::new();

    let rn = RMSNorm::new(&graph, &[3], 1e-5, "rn")?;
    let x = graph.input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))?;
    let y = rn.forward(&x);

    let target = graph.input(&Tensor::zeros(&[2, 3]))?;
    let loss = y.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val >= 0.0);

    let gamma_grad = rn.gamma().grad()?.expect("gamma 应有 grad");
    assert_eq!(gamma_grad.shape(), &[1, 3]);

    Ok(())
}

/// RMSNorm vs LayerNorm — RMSNorm 不减均值
///
/// 对于均值为零的输入，RMSNorm ≈ LayerNorm
/// 对于均值非零的输入，两者输出不同
#[test]
fn test_rms_norm_vs_layer_norm() -> Result<(), GraphError> {
    let graph = Graph::new();

    let rn = RMSNorm::new(&graph, &[3], 1e-5, "rn")?;
    let ln = crate::nn::LayerNorm::new(&graph, &[3], 1e-5, "ln")?;

    // 均值非零的输入
    let x = graph.input(&Tensor::new(&[10.0, 11.0, 12.0], &[1, 3]))?;

    let y_rms = rn.forward(&x);
    let y_ln = ln.forward(&x);

    y_rms.forward()?;
    y_ln.forward()?;

    let out_rms = y_rms.value()?.unwrap();
    let out_ln = y_ln.value()?.unwrap();

    // 两者应该不同（均值非零时）
    let diff: f32 = out_rms
        .data_as_slice()
        .iter()
        .zip(out_ln.data_as_slice().iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 0.01, "均值非零时 RMSNorm 和 LayerNorm 应有差异");

    Ok(())
}
