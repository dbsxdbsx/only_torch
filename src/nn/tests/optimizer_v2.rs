/*
 * @Description  : V2 Optimizer 测试
 *
 * 测试 PyTorch 风格的 V2 Optimizer API
 */

use crate::nn::graph::GraphHandle;
use crate::nn::layer::Linear;
use crate::nn::{Adamv2, Module, OptimizerV2, SGDv2, VarActivationOps, VarLossOps, VarMatrixOps};
use crate::tensor::Tensor;

#[test]
fn test_sgdv2_basic() {
    let graph = GraphHandle::new_with_seed(42);

    // 简单线性模型：y = x * w
    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let w = graph
        .parameter(
            &[2, 1],
            crate::nn::Init::Constant(0.5),
            "w",
        )
        .unwrap();
    let target = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();

    let y = x.matmul(&w).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    // 初始 loss
    loss.forward().unwrap();
    let initial_loss = loss.item().unwrap();

    // 创建 SGDv2 优化器
    let mut optimizer = SGDv2::new(&graph, &[w.clone()], 0.1);

    // 训练一步
    optimizer.zero_grad().unwrap();
    loss.backward().unwrap();
    optimizer.step().unwrap();

    // 重新计算 loss
    // 需要重新构建计算图，因为 w 已更新
    let y2 = x.matmul(&w).unwrap();
    let loss2 = y2.mse_loss(&target).unwrap();
    loss2.forward().unwrap();
    let new_loss = loss2.item().unwrap();

    // loss 应该下降
    assert!(new_loss < initial_loss, "Loss 应该下降: {} -> {}", initial_loss, new_loss);
}

#[test]
fn test_sgdv2_minimize() {
    let graph = GraphHandle::new_with_seed(42);

    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let w = graph
        .parameter(
            &[2, 1],
            crate::nn::Init::Constant(0.5),
            "w",
        )
        .unwrap();
    let target = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();

    let y = x.matmul(&w).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    let mut optimizer = SGDv2::new(&graph, &[w.clone()], 0.1);

    // 使用 minimize
    let loss_val = optimizer.minimize(&loss).unwrap();
    assert!(loss_val > 0.0, "Loss 应该是正数");
}

#[test]
fn test_adamv2_basic() {
    let graph = GraphHandle::new_with_seed(42);

    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let w = graph
        .parameter(
            &[2, 1],
            crate::nn::Init::Constant(0.5),
            "w",
        )
        .unwrap();
    let target = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();

    let y = x.matmul(&w).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    // 初始 loss
    loss.forward().unwrap();
    let initial_loss = loss.item().unwrap();

    // 创建 Adamv2 优化器
    let mut optimizer = Adamv2::new(&graph, &[w.clone()], 0.1);

    // 训练一步
    optimizer.zero_grad().unwrap();
    loss.backward().unwrap();
    optimizer.step().unwrap();

    // 重新计算 loss
    let y2 = x.matmul(&w).unwrap();
    let loss2 = y2.mse_loss(&target).unwrap();
    loss2.forward().unwrap();
    let new_loss = loss2.item().unwrap();

    // loss 应该下降
    assert!(new_loss < initial_loss, "Loss 应该下降: {} -> {}", initial_loss, new_loss);
}

#[test]
fn test_adamv2_minimize() {
    let graph = GraphHandle::new_with_seed(42);

    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let w = graph
        .parameter(
            &[2, 1],
            crate::nn::Init::Constant(0.5),
            "w",
        )
        .unwrap();
    let target = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();

    let y = x.matmul(&w).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    let mut optimizer = Adamv2::new(&graph, &[w.clone()], 0.1);

    // 使用 minimize
    let loss_val = optimizer.minimize(&loss).unwrap();
    assert!(loss_val > 0.0, "Loss 应该是正数");
}

#[test]
fn test_adamv2_reset() {
    let graph = GraphHandle::new_with_seed(42);

    let w = graph
        .parameter(
            &[2, 1],
            crate::nn::Init::Constant(0.5),
            "w",
        )
        .unwrap();

    let mut optimizer = Adamv2::new(&graph, &[w], 0.001);

    // 模拟一些更新
    // t 初始为 0，reset 后也应该是 0
    assert_eq!(optimizer.learning_rate(), 0.001);

    optimizer.set_learning_rate(0.01);
    assert_eq!(optimizer.learning_rate(), 0.01);

    optimizer.reset();
    // reset 不改变学习率，只清除动量
    assert_eq!(optimizer.learning_rate(), 0.01);
}

#[test]
fn test_sgdv2_with_linear_layer() {
    let graph = GraphHandle::new_with_seed(42);

    // 使用 Linear 层
    let fc = Linear::new(&graph, 3, 2, true, "fc").unwrap();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let target = graph.input(&Tensor::new(&[1.0, 0.0], &[1, 2])).unwrap();

    let y = fc.forward(&x);
    let loss = y.mse_loss(&target).unwrap();

    // 创建优化器，使用 Module 的 parameters()
    let params = fc.parameters();
    let mut optimizer = SGDv2::new(&graph, &params, 0.01);

    // 训练多步
    for _ in 0..10 {
        optimizer.zero_grad().unwrap();
        loss.backward().unwrap();
        optimizer.step().unwrap();
    }

    // 参数应该有梯度
    for p in &params {
        assert!(p.grad().unwrap().is_some());
    }
}

#[test]
fn test_adamv2_with_mlp() {
    let graph = GraphHandle::new_with_seed(42);

    // 两层 MLP
    let fc1 = Linear::new(&graph, 3, 4, true, "fc1").unwrap();
    let fc2 = Linear::new(&graph, 4, 2, true, "fc2").unwrap();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let target = graph.input(&Tensor::new(&[1.0, 0.0], &[1, 2])).unwrap();

    let h = fc1.forward(&x).relu();
    let y = fc2.forward(&h);
    let loss = y.mse_loss(&target).unwrap();

    // 收集所有参数
    let mut params = fc1.parameters();
    params.extend(fc2.parameters());

    let mut optimizer = Adamv2::new(&graph, &params, 0.01);

    // 训练
    let initial_loss = loss.backward().unwrap();
    optimizer.step().unwrap();

    // 训练更多步
    for _ in 0..5 {
        let _ = optimizer.minimize(&loss).unwrap();
    }

    // 最终 loss 应该下降
    loss.forward().unwrap();
    let final_loss = loss.item().unwrap();
    assert!(
        final_loss < initial_loss,
        "Loss 应该下降: {} -> {}",
        initial_loss,
        final_loss
    );
}
