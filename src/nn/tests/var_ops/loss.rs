/*
 * @Description  : VarLossOps trait 测试
 *
 * 测试损失函数扩展 trait 的独立功能：
 * - mse_loss: 均方误差（回归）
 * - cross_entropy: 交叉熵（分类）
 * - 标量 LossTarget：数值类型自动广播
 */

use crate::nn::VarLossOps;
use crate::nn::graph::Graph;
use crate::tensor::Tensor;

#[test]
fn test_var_mse_loss() {
    let graph = Graph::new();
    let pred = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]))
        .unwrap();
    let loss = pred.mse_loss(&target).unwrap();
    loss.forward().unwrap();
    let result = loss.item().unwrap();
    assert!((result - 0.0).abs() < 1e-5);
}

#[test]
fn test_var_mse_loss_nonzero() {
    let graph = Graph::new();
    let pred = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[2.0, 3.0, 4.0], &[3, 1]))
        .unwrap();
    let loss = pred.mse_loss(&target).unwrap();
    loss.forward().unwrap();
    let result = loss.item().unwrap();
    // MSE = mean((1-2)^2 + (2-3)^2 + (3-4)^2) = mean(1+1+1) = 1.0
    assert!((result - 1.0).abs() < 1e-5);
}

#[test]
fn test_var_cross_entropy() {
    let graph = Graph::new();
    // logits: [1, 2, 3] -> softmax -> cross_entropy with [0, 0, 1]
    let logits = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let labels = graph
        .input(&Tensor::new(&[0.0, 0.0, 1.0], &[1, 3]))
        .unwrap();
    let loss = logits.cross_entropy(&labels).unwrap();
    loss.forward().unwrap();
    let result = loss.item().unwrap();
    // cross_entropy(softmax([1,2,3]), [0,0,1]) ≈ 0.4076
    assert!(result > 0.0 && result < 1.0);
}

// ==================== 标量 LossTarget 测试 ====================

/// 标量 i32 作为 mse_loss target：`mse_loss(0)` 等价于 `mse_loss(&Tensor::zeros(...))`
#[test]
fn test_scalar_mse_loss_i32_zero() {
    let graph = Graph::new();
    let pred = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]))
        .unwrap();

    // 标量写法
    let loss_scalar = pred.mse_loss(0).unwrap();
    loss_scalar.forward().unwrap();
    let result_scalar = loss_scalar.item().unwrap();

    // 等价的显式 Tensor 写法
    let loss_tensor = pred.mse_loss(&Tensor::zeros(&[3, 1])).unwrap();
    loss_tensor.forward().unwrap();
    let result_tensor = loss_tensor.item().unwrap();

    assert!((result_scalar - result_tensor).abs() < 1e-6);
    // MSE = mean((1-0)^2 + (2-0)^2 + (3-0)^2) / 3 = (1+4+9)/3 ≈ 4.6667
    assert!((result_scalar - 14.0 / 3.0).abs() < 1e-4);
}

/// 标量 i32 作为 mse_loss target：`mse_loss(1)` 等价于 `mse_loss(&Tensor::ones(...))`
#[test]
fn test_scalar_mse_loss_i32_one() {
    let graph = Graph::new();
    let pred = graph
        .input(&Tensor::new(&[1.0, 1.0, 1.0], &[3, 1]))
        .unwrap();

    // pred == target(全 1)，loss 应为 0
    let loss = pred.mse_loss(1).unwrap();
    loss.forward().unwrap();
    assert!(loss.item().unwrap().abs() < 1e-6);
}

/// 标量 f64 作为 mse_loss target
#[test]
fn test_scalar_mse_loss_f64() {
    let graph = Graph::new();
    let pred = graph
        .input(&Tensor::new(&[0.5, 0.5], &[2, 1]))
        .unwrap();

    // target = 0.5，pred = 0.5，loss 应为 0
    let loss = pred.mse_loss(0.5_f64).unwrap();
    loss.forward().unwrap();
    assert!(loss.item().unwrap().abs() < 1e-6);
}

/// 标量 f32 作为 mse_loss target
#[test]
fn test_scalar_mse_loss_f32() {
    let graph = Graph::new();
    let pred = graph
        .input(&Tensor::new(&[2.0, 3.0], &[2, 1]))
        .unwrap();

    // MSE((2,3), (1.5,1.5)) = mean((0.5)^2 + (1.5)^2) = mean(0.25 + 2.25) = 1.25
    let loss = pred.mse_loss(1.5_f32).unwrap();
    loss.forward().unwrap();
    assert!((loss.item().unwrap() - 1.25).abs() < 1e-5);
}

/// 标量 u32 作为 mse_loss target
#[test]
fn test_scalar_mse_loss_u32() {
    let graph = Graph::new();
    let pred = graph
        .input(&Tensor::new(&[2.0, 2.0], &[2, 1]))
        .unwrap();
    let loss = pred.mse_loss(2_u32).unwrap();
    loss.forward().unwrap();
    assert!(loss.item().unwrap().abs() < 1e-6);
}

/// 标量 i64 / u64 覆盖
#[test]
fn test_scalar_mse_loss_i64_u64() {
    let graph = Graph::new();
    let pred = graph
        .input(&Tensor::new(&[0.0, 0.0], &[2, 1]))
        .unwrap();

    let loss_i64 = pred.mse_loss(0_i64).unwrap();
    loss_i64.forward().unwrap();
    assert!(loss_i64.item().unwrap().abs() < 1e-6);

    let loss_u64 = pred.mse_loss(0_u64).unwrap();
    loss_u64.forward().unwrap();
    assert!(loss_u64.item().unwrap().abs() < 1e-6);
}

/// 标量 target 对 bce_loss 的覆盖
#[test]
fn test_scalar_bce_loss() {
    let graph = Graph::new();
    // 预测 logits 较大正值 -> sigmoid 接近 1 -> target=1 -> loss 接近 0
    let pred = graph
        .input(&Tensor::new(&[5.0, 5.0], &[2, 1]))
        .unwrap();
    let loss = pred.bce_loss(1).unwrap();
    loss.forward().unwrap();
    assert!(loss.item().unwrap() < 0.1);
}

/// 标量 target 对 mae_loss 的覆盖
#[test]
fn test_scalar_mae_loss() {
    let graph = Graph::new();
    let pred = graph
        .input(&Tensor::new(&[1.0, 3.0], &[2, 1]))
        .unwrap();
    // MAE((1,3), (2,2)) = mean(|1-2| + |3-2|) = mean(1 + 1) = 1.0
    let loss = pred.mae_loss(2).unwrap();
    loss.forward().unwrap();
    assert!((loss.item().unwrap() - 1.0).abs() < 1e-5);
}

/// 标量 target 对 huber_loss 的覆盖
#[test]
fn test_scalar_huber_loss() {
    let graph = Graph::new();
    let pred = graph
        .input(&Tensor::new(&[1.0, 1.0], &[2, 1]))
        .unwrap();
    // pred == target(全 1)，loss 应为 0
    let loss = pred.huber_loss(1).unwrap();
    loss.forward().unwrap();
    assert!(loss.item().unwrap().abs() < 1e-6);
}

/// 标量 target 对多维张量的广播正确性
#[test]
fn test_scalar_broadcast_2d() {
    let graph = Graph::new();
    // 2x3 的预测张量
    let pred = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();

    // 标量 0 -> 广播为 [[0,0,0],[0,0,0]]
    let loss = pred.mse_loss(0).unwrap();
    loss.forward().unwrap();
    // MSE = mean(1+4+9+16+25+36) / 6 = 91/6 ≈ 15.1667
    assert!((loss.item().unwrap() - 91.0 / 6.0).abs() < 1e-4);
}

/// 标量 target 支持反向传播
#[test]
fn test_scalar_mse_loss_backward() {
    use crate::nn::VarMatrixOps;
    let graph = Graph::new();
    let w = graph
        .parameter(&[1, 1], crate::nn::var::Init::Constant(0.5), "w")
        .unwrap();
    let x = graph.input(&Tensor::new(&[2.0], &[1, 1])).unwrap();
    let pred = x.matmul(&w).unwrap();

    // target = 1，pred = 2*0.5 = 1.0，loss 应为 0
    let loss = pred.mse_loss(1).unwrap();
    loss.forward().unwrap();
    assert!(loss.item().unwrap().abs() < 1e-6);

    // 反向传播不应出错
    loss.backward().unwrap();
}
