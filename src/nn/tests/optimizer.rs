/*
 * @Description  : Optimizer 测试
 *
 * 测试 PyTorch 风格的 Optimizer API
 *
 * 测试覆盖：
 * - SGD/Adam 基本训练流程
 * - minimize() 一步完成
 * - learning_rate() / set_learning_rate() / reset()
 * - params() 参数列表查询
 * - Adam 状态查询：get_momentum() / get_velocity() / timestep()
 */

use crate::nn::graph::Graph;
use crate::nn::layer::Linear;
use crate::nn::{Adam, Module, Optimizer, SGD, VarActivationOps, VarLossOps, VarMatrixOps};
use crate::tensor::Tensor;

#[test]
fn test_sgd_basic() {
    let graph = Graph::new_with_seed(42);

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

    // 创建 SGD 优化器
    let mut optimizer = SGD::new(&graph, &[w.clone()], 0.1);

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
fn test_sgd_minimize() {
    let graph = Graph::new_with_seed(42);

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

    let mut optimizer = SGD::new(&graph, &[w.clone()], 0.1);

    // 使用 minimize
    let loss_val = optimizer.minimize(&loss).unwrap();
    assert!(loss_val > 0.0, "Loss 应该是正数");
}

#[test]
fn test_adam_basic() {
    let graph = Graph::new_with_seed(42);

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

    // 创建 Adam 优化器
    let mut optimizer = Adam::new(&graph, &[w.clone()], 0.1);

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
fn test_adam_minimize() {
    let graph = Graph::new_with_seed(42);

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

    let mut optimizer = Adam::new(&graph, &[w.clone()], 0.1);

    // 使用 minimize
    let loss_val = optimizer.minimize(&loss).unwrap();
    assert!(loss_val > 0.0, "Loss 应该是正数");
}

#[test]
fn test_adam_reset() {
    let graph = Graph::new_with_seed(42);

    let w = graph
        .parameter(
            &[2, 1],
            crate::nn::Init::Constant(0.5),
            "w",
        )
        .unwrap();

    let mut optimizer = Adam::new(&graph, &[w], 0.001);

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
fn test_sgd_with_linear_layer() {
    let graph = Graph::new_with_seed(42);

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
    let mut optimizer = SGD::new(&graph, &params, 0.01);

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
fn test_adam_with_mlp() {
    let graph = Graph::new_with_seed(42);

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

    let mut optimizer = Adam::new(&graph, &params, 0.01);

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

#[test]
fn test_optimizer_params_accessor() {
    let graph = Graph::new_with_seed(42);

    let w1 = graph
        .parameter(&[2, 3], crate::nn::Init::Constant(0.1), "w1")
        .unwrap();
    let w2 = graph
        .parameter(&[3, 1], crate::nn::Init::Constant(0.2), "w2")
        .unwrap();

    // SGD params() 测试
    let sgd = SGD::new(&graph, &[w1.clone(), w2.clone()], 0.01);
    let sgd_params = sgd.params();
    assert_eq!(sgd_params.len(), 2);
    assert_eq!(sgd_params[0].node_id(), w1.node_id());
    assert_eq!(sgd_params[1].node_id(), w2.node_id());

    // Adam params() 测试
    let adam = Adam::new(&graph, &[w1.clone(), w2.clone()], 0.001);
    let adam_params = adam.params();
    assert_eq!(adam_params.len(), 2);
    assert_eq!(adam_params[0].node_id(), w1.node_id());
    assert_eq!(adam_params[1].node_id(), w2.node_id());
}

#[test]
fn test_adam_state_accessors() {
    let graph = Graph::new_with_seed(42);

    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let w = graph
        .parameter(&[2, 1], crate::nn::Init::Constant(0.5), "w")
        .unwrap();
    let target = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();

    let y = x.matmul(&w).unwrap();
    let loss = y.mse_loss(&target).unwrap();

    let mut optimizer = Adam::new(&graph, &[w.clone()], 0.1);

    // 初始状态：timestep = 0，无动量/速度
    assert_eq!(optimizer.timestep(), 0);
    assert!(optimizer.get_momentum(&w).is_none());
    assert!(optimizer.get_velocity(&w).is_none());

    // 执行一步优化
    optimizer.zero_grad().unwrap();
    loss.backward().unwrap();
    optimizer.step().unwrap();

    // 优化后：timestep = 1，有动量/速度
    assert_eq!(optimizer.timestep(), 1);
    assert!(optimizer.get_momentum(&w).is_some());
    assert!(optimizer.get_velocity(&w).is_some());

    // 验证动量/速度的形状与参数一致
    let momentum = optimizer.get_momentum(&w).unwrap();
    let velocity = optimizer.get_velocity(&w).unwrap();
    assert_eq!(momentum.shape(), &[2, 1]);
    assert_eq!(velocity.shape(), &[2, 1]);

    // 再执行一步
    optimizer.minimize(&loss).unwrap();
    assert_eq!(optimizer.timestep(), 2);

    // reset 后状态清零
    optimizer.reset();
    assert_eq!(optimizer.timestep(), 0);
    assert!(optimizer.get_momentum(&w).is_none());
    assert!(optimizer.get_velocity(&w).is_none());
}
