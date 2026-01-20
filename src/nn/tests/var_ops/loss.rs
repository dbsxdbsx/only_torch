/*
 * @Description  : VarLossOps trait 测试
 *
 * 测试损失函数扩展 trait 的独立功能：
 * - mse_loss: 均方误差（回归）
 * - cross_entropy: 交叉熵（分类）
 */

use crate::nn::graph::Graph;
use crate::nn::VarLossOps;
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
