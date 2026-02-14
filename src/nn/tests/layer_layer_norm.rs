/*
 * @Author       : 老董
 * @Description  : LayerNorm 层（含 gamma/beta）集成测试
 *
 * 验证 LayerNorm Layer（gamma * layer_norm(x) + beta）的完整行为：
 * 1. 默认参数（gamma=1, beta=0）等价于纯 LayerNormOp
 * 2. 参数可学习性（gamma, beta 出现在 parameters() 中）
 * 3. 前向传播形状正确性
 * 4. 反向传播梯度流至 gamma/beta
 *
 * Python 对照值:
 *   见 tests/layer_norm_reference.py
 */

use crate::nn::{Graph, GraphError, LayerNorm, Module, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 前向传播测试 ====================

/// LayerNorm 默认参数 (gamma=1, beta=0)
///
/// 输入 [2, 3]: [[1, 2, 3], [4, 5, 6]]
/// 预期输出 = 纯 LayerNormOp 输出:
///   [[-1.2247, 0.0, 1.2247], [-1.2247, 0.0, 1.2247]]
#[test]
fn test_layer_norm_default_params() -> Result<(), GraphError> {
    let graph = Graph::new();

    let ln = LayerNorm::new(&graph, &[3], 1e-5, "ln")?;

    let x = graph.input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))?;
    let y = ln.forward(&x);
    y.forward()?;

    let output = y.value()?.unwrap();
    assert_eq!(output.shape(), &[2, 3]);
    // gamma=1, beta=0 → 等价于纯 LayerNormOp
    assert_abs_diff_eq!(output[[0, 0]], -1.2247, epsilon = 1e-3);
    assert_abs_diff_eq!(output[[0, 1]], 0.0, epsilon = 1e-3);
    assert_abs_diff_eq!(output[[0, 2]], 1.2247, epsilon = 1e-3);
    assert_abs_diff_eq!(output[[1, 0]], -1.2247, epsilon = 1e-3);
    assert_abs_diff_eq!(output[[1, 1]], 0.0, epsilon = 1e-3);
    assert_abs_diff_eq!(output[[1, 2]], 1.2247, epsilon = 1e-3);

    Ok(())
}

// ==================== 参数测试 ====================

/// 测试 Module::parameters() 返回 gamma 和 beta
#[test]
fn test_layer_norm_parameters() -> Result<(), GraphError> {
    let graph = Graph::new();
    let ln = LayerNorm::new(&graph, &[64], 1e-5, "ln")?;

    let params = ln.parameters();
    assert_eq!(params.len(), 2); // gamma + beta

    // gamma 形状 [1, 64]（1D normalized_shape 前置一维），初始化为 1
    let gamma_val = params[0].value()?.unwrap();
    assert_eq!(gamma_val.shape(), &[1, 64]);
    assert_abs_diff_eq!(gamma_val[[0, 0]], 1.0, epsilon = 1e-6);

    // beta 形状 [1, 64]，初始化为 0
    let beta_val = params[1].value()?.unwrap();
    assert_eq!(beta_val.shape(), &[1, 64]);
    assert_abs_diff_eq!(beta_val[[0, 0]], 0.0, epsilon = 1e-6);

    Ok(())
}

// ==================== 3D 测试 ====================

/// LayerNorm 3D 输入 [N, T, D]
#[test]
fn test_layer_norm_3d() -> Result<(), GraphError> {
    let graph = Graph::new();

    let ln = LayerNorm::new(&graph, &[4], 1e-5, "ln")?;

    #[rustfmt::skip]
    let x = graph.input(&Tensor::new(&[
        1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,
        0.1, 0.2, 0.3, 0.4,  0.5, 0.6, 0.7, 0.8,
    ], &[2, 2, 4]))?;

    let y = ln.forward(&x);
    y.forward()?;

    let output = y.value()?.unwrap();
    assert_eq!(output.shape(), &[2, 2, 4]);
    assert_abs_diff_eq!(output[[0, 0, 0]], -1.3416, epsilon = 1e-3);
    assert_abs_diff_eq!(output[[0, 0, 3]], 1.3416, epsilon = 1e-3);

    Ok(())
}

// ==================== 反向传播测试 ====================

/// LayerNorm 反向传播 — gamma/beta 接收梯度
#[test]
fn test_layer_norm_backward() -> Result<(), GraphError> {
    let graph = Graph::new();

    let bn = LayerNorm::new(&graph, &[3], 1e-5, "ln")?;

    let x = graph.input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))?;
    let y = bn.forward(&x);

    // 用 mse loss
    let target = graph.input(&Tensor::zeros(&[2, 3]))?;
    let loss = y.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val >= 0.0);

    // gamma 应有梯度
    let gamma_grad = bn.gamma().grad()?.expect("gamma 应有 grad");
    assert_eq!(gamma_grad.shape(), &[1, 3]);

    // beta 应有梯度
    let beta_grad = bn.beta().grad()?.expect("beta 应有 grad");
    assert_eq!(beta_grad.shape(), &[1, 3]);

    Ok(())
}

// ==================== 自定义 Affine 测试 ====================

/// LayerNorm 自定义 gamma/beta 值
///
/// PyTorch 对照：gamma=[2, 0.5, 1], beta=[0.1, 0.2, 0.3]
/// 输入 [[1,2,3],[4,5,6]]
/// 预期输出:
///   [[-2.3495, 0.2, 1.5247], [-2.3495, 0.2, 1.5247]]
#[test]
fn test_layer_norm_custom_affine() -> Result<(), GraphError> {
    let graph = Graph::new();

    let ln = LayerNorm::new(&graph, &[3], 1e-5, "ln")?;

    // 手动设置 gamma 和 beta（参数形状为 [1, 3]）
    ln.gamma()
        .set_value(&Tensor::new(&[2.0, 0.5, 1.0], &[1, 3]))?;
    ln.beta()
        .set_value(&Tensor::new(&[0.1, 0.2, 0.3], &[1, 3]))?;

    let x = graph.input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))?;
    let y = ln.forward(&x);
    y.forward()?;

    let output = y.value()?.unwrap();
    assert_eq!(output.shape(), &[2, 3]);
    // PyTorch: [[-2.3495, 0.2, 1.5247], [-2.3495, 0.2, 1.5247]]
    assert_abs_diff_eq!(output[[0, 0]], -2.3495, epsilon = 1e-2);
    assert_abs_diff_eq!(output[[0, 1]], 0.2, epsilon = 1e-2);
    assert_abs_diff_eq!(output[[0, 2]], 1.5247, epsilon = 1e-2);
    assert_abs_diff_eq!(output[[1, 0]], -2.3495, epsilon = 1e-2);
    assert_abs_diff_eq!(output[[1, 1]], 0.2, epsilon = 1e-2);
    assert_abs_diff_eq!(output[[1, 2]], 1.5247, epsilon = 1e-2);

    Ok(())
}
