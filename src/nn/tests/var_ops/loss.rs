/*
 * @Description  : VarLossOps trait 测试
 *
 * 测试损失函数扩展 trait 的独立功能：
 * - mse_loss, cross_entropy, perception_loss
 */

use crate::nn::graph::GraphHandle;
use crate::nn::VarLossOps;
use crate::tensor::Tensor;

#[test]
fn test_var_mse_loss() {
    let graph = GraphHandle::new();
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
    let graph = GraphHandle::new();
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
    let graph = GraphHandle::new();
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

#[test]
fn test_var_perception_loss() {
    let graph = GraphHandle::new();
    // 正确分类: label=1, output=0.5 -> loss = max(0, -0.5) = 0
    let output = graph.input(&Tensor::new(&[0.5], &[1, 1])).unwrap();
    let label = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();
    let combined = &output * &label;
    let loss = combined.perception_loss();
    loss.forward().unwrap();
    let result = loss.item().unwrap();
    assert!((result - 0.0).abs() < 1e-5);
}

#[test]
fn test_var_perception_loss_wrong_classification() {
    let graph = GraphHandle::new();
    // 错误分类: label=-1, output=0.5 -> loss = max(0, -(-0.5)) = 0.5
    let output = graph.input(&Tensor::new(&[0.5], &[1, 1])).unwrap();
    let label = graph.input(&Tensor::new(&[-1.0], &[1, 1])).unwrap();
    let combined = &output * &label;
    let loss = combined.perception_loss();
    loss.forward().unwrap();
    let result = loss.item().unwrap();
    assert!((result - 0.5).abs() < 1e-5);
}
