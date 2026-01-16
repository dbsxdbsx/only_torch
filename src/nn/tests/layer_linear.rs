/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : Linear layer 单元测试（Batch-First 设计，含 PyTorch 数值对照）
 *
 * 参考值来源: tests/python/layer_reference/linear_layer_reference.py
 */

use crate::nn::layer::linear;
use crate::nn::{GraphInner, GraphError};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== PyTorch 参考常量 ====================

// 测试: 简单前向传播 (batch=2, in=3, out=4)
const PYTORCH_FWD_X: &[f32] = &[1.0, 2.0, 3.0, 0.5, 1.5, 2.5];
const PYTORCH_FWD_W: &[f32] = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
const PYTORCH_FWD_B: &[f32] = &[0.1, 0.2, 0.3, 0.4];
const PYTORCH_FWD_OUTPUT: &[f32] = &[3.9, 4.6, 5.3, 6.0, 3.15, 3.7, 4.25, 4.8];

// 测试: 反向传播梯度 (batch=2, in=3, out=2) + MSE Loss
const PYTORCH_BWD_X: &[f32] = &[1.0, 2.0, 3.0, 0.5, 1.0, 1.5];
const PYTORCH_BWD_W: &[f32] = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
const PYTORCH_BWD_B: &[f32] = &[0.1, 0.2];
const PYTORCH_BWD_TARGET: &[f32] = &[1.0, 0.5, 0.5, 1.0];
const PYTORCH_BWD_OUTPUT: &[f32] = &[2.3, 3.0, 1.2, 1.6];
const PYTORCH_BWD_LOSS: f32 = 2.1975;
const PYTORCH_BWD_GRAD_W: &[f32] = &[0.825, 1.4, 1.65, 2.8, 2.475, 4.2];
const PYTORCH_BWD_GRAD_B: &[f32] = &[1.0, 1.55];

// 测试: 两层 Linear + ReLU + SoftmaxCrossEntropy
const PYTORCH_CHAIN_X: &[f32] = &[1.0, 0.5, -0.5, 0.2, 0.3, -0.2, 0.8, -0.1];
const PYTORCH_CHAIN_W1: &[f32] = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.2, 0.3, 0.4];
const PYTORCH_CHAIN_B1: &[f32] = &[0.1, 0.1, 0.1];
const PYTORCH_CHAIN_W2: &[f32] = &[0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
const PYTORCH_CHAIN_B2: &[f32] = &[0.1, 0.2];
const PYTORCH_CHAIN_TARGET: &[f32] = &[1.0, 0.0, 0.0, 1.0];
const PYTORCH_CHAIN_H1_RELU: &[f32] = &[0.09, 0.21, 0.33, 0.59, 0.67, 0.75];
const PYTORCH_CHAIN_LOGITS: &[f32] = &[0.589, 0.752, 1.539, 1.84];
const PYTORCH_CHAIN_LOSS: f32 = 0.6659472;
const PYTORCH_CHAIN_GRAD_W1: &[f32] = &[
    0.02065329,
    0.02065331,
    0.02065329,
    0.01776964,
    0.01776965,
    0.01776964,
    -0.03052906,
    -0.03052907,
    -0.03052907,
    0.00753317,
    0.00753317,
    0.00753317,
];
const PYTORCH_CHAIN_GRAD_B1: &[f32] = &[0.00576731, 0.00576732, 0.00576729];
const PYTORCH_CHAIN_GRAD_W2: &[f32] = &[
    0.10113763,
    -0.10113766,
    0.08571057,
    -0.0857106,
    0.07028344,
    -0.07028348,
];
const PYTORCH_CHAIN_GRAD_B2: &[f32] = &[-0.0576735, 0.05767344];

// ==================== 基础功能测试 ====================

/// 测试 linear() 创建
#[test]
fn test_linear_creation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();
    let batch_size = 32;
    let input = graph.new_input_node(&[batch_size, 4], Some("input"))?;

    let fc = linear(&mut graph, input, 4, 3, batch_size, Some("fc"))?;

    // 验证返回的节点 ID 都有效
    assert!(graph.get_node_value(fc.output).is_ok());
    assert!(graph.get_node_value(fc.weights).is_ok());
    assert!(graph.get_node_value(fc.bias).is_ok());
    assert!(graph.get_node_value(fc.ones).is_ok());

    Ok(())
}

/// 测试 linear() 参数形状
#[test]
fn test_linear_shapes() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 64;
    let input = graph.new_input_node(&[batch_size, 784], Some("input"))?;

    let fc = linear(&mut graph, input, 784, 128, batch_size, Some("fc"))?;

    // 检查权重形状：[in_features, out_features]
    let w_shape = graph.get_node(fc.weights)?.value_expected_shape();
    assert_eq!(w_shape, &[784, 128]);

    // 检查偏置形状：[1, out_features]
    let b_shape = graph.get_node(fc.bias)?.value_expected_shape();
    assert_eq!(b_shape, &[1, 128]);

    // 检查 ones 形状：[batch_size, 1]
    let ones_shape = graph.get_node(fc.ones)?.value_expected_shape();
    assert_eq!(ones_shape, &[batch_size, 1]);

    Ok(())
}

/// 测试 linear() 前向传播
#[test]
fn test_linear_forward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 2;

    let input = graph.new_input_node(&[batch_size, 3], Some("input"))?;
    let fc = linear(&mut graph, input, 3, 2, batch_size, Some("fc"))?;

    // 设置输入: [2, 3]
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    // 设置权重: [3, 2] - 单位矩阵的前两列
    let w = Tensor::new(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[3, 2]);
    // 设置偏置: [1, 2]
    let b = Tensor::new(&[0.5, 0.5], &[1, 2]);
    graph.set_node_value(input, Some(&x))?;
    graph.set_node_value(fc.weights, Some(&w))?;
    graph.set_node_value(fc.bias, Some(&b))?;

    // 前向传播
    graph.forward(fc.output)?;

    // 验证输出
    // x @ W = [[1,2,3], [4,5,6]] @ [[1,0], [0,1], [0,0]] = [[1, 2], [4, 5]]
    // + b = [[1.5, 2.5], [4.5, 5.5]]
    let output = graph.get_node_value(fc.output)?.unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 1.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 2.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 4.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 5.5, epsilon = 1e-6);

    Ok(())
}

// ==================== 节点名称测试 ====================

/// 测试 linear() 带名称
#[test]
fn test_linear_with_name() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();
    let batch_size = 16;
    let input = graph.new_input_node(&[batch_size, 4], Some("input"))?;

    let fc = linear(&mut graph, input, 4, 2, batch_size, Some("hidden"))?;

    // 验证节点名称
    assert_eq!(graph.get_node(fc.weights)?.name(), "hidden_W");
    assert_eq!(graph.get_node(fc.bias)?.name(), "hidden_b");
    assert_eq!(graph.get_node(fc.ones)?.name(), "hidden_ones");
    assert_eq!(graph.get_node(fc.output)?.name(), "hidden_out");

    Ok(())
}

/// 测试 linear() 无名称（使用默认前缀）
#[test]
fn test_linear_without_name() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();
    let batch_size = 16;
    let input = graph.new_input_node(&[batch_size, 4], Some("input"))?;

    let fc = linear(&mut graph, input, 4, 2, batch_size, None)?;

    // 验证使用默认前缀 "linear"
    assert_eq!(graph.get_node(fc.weights)?.name(), "linear_W");
    assert_eq!(graph.get_node(fc.bias)?.name(), "linear_b");
    assert_eq!(graph.get_node(fc.ones)?.name(), "linear_ones");
    assert_eq!(graph.get_node(fc.output)?.name(), "linear_out");

    Ok(())
}

// ==================== 链式连接测试 ====================

/// 测试多层 linear() 链式连接
#[test]
fn test_linear_chain() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 8;

    let input = graph.new_input_node(&[batch_size, 4], Some("input"))?;
    let fc1 = linear(&mut graph, input, 4, 8, batch_size, Some("fc1"))?;
    let act = graph.new_sigmoid_node(fc1.output, Some("act"))?;
    let fc2 = linear(&mut graph, act, 8, 2, batch_size, Some("fc2"))?;

    // 设置输入
    let x = Tensor::normal(0.0, 1.0, &[batch_size, 4]);
    graph.set_node_value(input, Some(&x))?;

    // 前向传播
    graph.forward(fc2.output)?;

    // 验证输出存在且形状正确
    let output = graph.get_node_value(fc2.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[batch_size, 2]);

    Ok(())
}

// ==================== 反向传播测试 ====================

/// 测试 linear() 与 Batch 反向传播
#[test]
fn test_linear_batch_backward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 4;

    // 构建网络
    let input = graph.new_input_node(&[batch_size, 3], Some("input"))?;
    let fc = linear(&mut graph, input, 3, 2, batch_size, Some("fc"))?;
    let labels = graph.new_input_node(&[batch_size, 2], Some("labels"))?;
    let loss = graph.new_softmax_cross_entropy_node(fc.output, labels, Some("loss"))?;

    // 设置数据
    let x = Tensor::normal(0.0, 1.0, &[batch_size, 3]);
    let y = Tensor::new(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[batch_size, 2]); // one-hot

    graph.set_node_value(input, Some(&x))?;
    graph.set_node_value(labels, Some(&y))?;

    // Batch 训练
    graph.forward(loss)?;
    graph.backward(loss)?;

    // 验证权重和偏置都有梯度
    let w_grad = graph.get_node_grad_ref(fc.weights)?;
    let b_grad = graph.get_node_grad_ref(fc.bias)?;
    assert!(w_grad.is_some());
    assert!(b_grad.is_some());

    // 检查梯度形状
    assert_eq!(w_grad.unwrap().shape(), &[3, 2]);
    assert_eq!(b_grad.unwrap().shape(), &[1, 2]);

    Ok(())
}

/// 测试多层 linear() + Loss 的 Batch 训练
#[test]
fn test_linear_chain_batch_training() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 4;

    // 构建网络: input -> fc1 -> relu -> fc2 -> loss
    let input = graph.new_input_node(&[batch_size, 4], Some("input"))?;
    let fc1 = linear(&mut graph, input, 4, 8, batch_size, Some("fc1"))?;
    let relu = graph.new_leaky_relu_node(fc1.output, 0.0, Some("relu"))?;
    let fc2 = linear(&mut graph, relu, 8, 3, batch_size, Some("fc2"))?;

    let labels = graph.new_input_node(&[batch_size, 3], Some("labels"))?;
    let loss = graph.new_softmax_cross_entropy_node(fc2.output, labels, Some("loss"))?;

    // 设置数据
    let x = Tensor::normal(0.0, 1.0, &[batch_size, 4]);
    let y = Tensor::new(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        &[batch_size, 3],
    ); // one-hot

    graph.set_node_value(input, Some(&x))?;
    graph.set_node_value(labels, Some(&y))?;

    // Batch 训练
    graph.forward(loss)?;
    graph.backward(loss)?;

    // 验证所有参数都有梯度
    assert!(graph.get_node_grad_ref(fc1.weights)?.is_some());
    assert!(graph.get_node_grad_ref(fc1.bias)?.is_some());
    assert!(graph.get_node_grad_ref(fc2.weights)?.is_some());
    assert!(graph.get_node_grad_ref(fc2.bias)?.is_some());

    Ok(())
}

// ==================== 名称冲突测试 ====================

/// 测试多个 linear() 使用不同名称
#[test]
fn test_linear_multiple_layers_different_names() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();
    let batch_size = 16;
    let input = graph.new_input_node(&[batch_size, 4], Some("input"))?;

    let fc1 = linear(&mut graph, input, 4, 8, batch_size, Some("fc1"))?;
    let fc2 = linear(&mut graph, fc1.output, 8, 4, batch_size, Some("fc2"))?;
    let fc3 = linear(&mut graph, fc2.output, 4, 2, batch_size, Some("fc3"))?;

    // 验证各层节点独立存在
    assert!(graph.get_node_value(fc1.weights).is_ok());
    assert!(graph.get_node_value(fc2.weights).is_ok());
    assert!(graph.get_node_value(fc3.weights).is_ok());

    // 验证节点名称正确
    assert_eq!(graph.get_node(fc1.weights)?.name(), "fc1_W");
    assert_eq!(graph.get_node(fc2.weights)?.name(), "fc2_W");
    assert_eq!(graph.get_node(fc3.weights)?.name(), "fc3_W");

    Ok(())
}

/// 测试重复名称应该报错
#[test]
fn test_linear_duplicate_name_error() {
    let mut graph = GraphInner::new();
    let batch_size = 16;
    let input = graph
        .new_input_node(&[batch_size, 4], Some("input"))
        .unwrap();

    // 第一个 fc 成功
    let fc1 = linear(&mut graph, input, 4, 8, batch_size, Some("fc"));
    assert!(fc1.is_ok());

    // 第二个 fc 使用相同名称，应该失败
    let fc2 = linear(
        &mut graph,
        fc1.unwrap().output,
        8,
        2,
        batch_size,
        Some("fc"),
    );
    assert!(fc2.is_err());

    // 验证错误类型
    if let Err(e) = fc2 {
        let err_msg = format!("{:?}", e);
        assert!(
            err_msg.contains("Duplicate") || err_msg.contains("重复"),
            "错误信息应包含重复名称提示: {}",
            err_msg
        );
    }
}

/// 测试多个无名称层会冲突（预期行为）
#[test]
fn test_linear_multiple_unnamed_layers_conflict() {
    let mut graph = GraphInner::new();
    let batch_size = 16;
    let input = graph
        .new_input_node(&[batch_size, 4], Some("input"))
        .unwrap();

    // 第一个无名称层成功
    let fc1 = linear(&mut graph, input, 4, 8, batch_size, None);
    assert!(fc1.is_ok());

    // 第二个无名称层应该失败（名称冲突）
    let fc2 = linear(&mut graph, fc1.unwrap().output, 8, 4, batch_size, None);
    assert!(fc2.is_err(), "多个无名称 linear 层应该因名称冲突而失败");
}

// ==================== 边界维度测试 ====================

/// 测试单特征输入
#[test]
fn test_linear_single_input_feature() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 4;
    let input = graph.new_input_node(&[batch_size, 1], Some("input"))?;

    let fc = linear(&mut graph, input, 1, 4, batch_size, Some("fc"))?;

    // 设置输入
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[batch_size, 1]);
    graph.set_node_value(input, Some(&x))?;

    // 前向传播
    graph.forward(fc.output)?;

    let output = graph.get_node_value(fc.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[batch_size, 4]);

    Ok(())
}

/// 测试单特征输出
#[test]
fn test_linear_single_output_feature() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 4;
    let input = graph.new_input_node(&[batch_size, 4], Some("input"))?;

    let fc = linear(&mut graph, input, 4, 1, batch_size, Some("fc"))?;

    // 设置输入
    let x = Tensor::normal(0.0, 1.0, &[batch_size, 4]);
    graph.set_node_value(input, Some(&x))?;

    // 前向传播
    graph.forward(fc.output)?;

    let output = graph.get_node_value(fc.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[batch_size, 1]);

    Ok(())
}

/// 测试大维度 linear()（典型 MNIST 配置）
#[test]
fn test_linear_large_dimensions() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 64;
    let input = graph.new_input_node(&[batch_size, 784], Some("input"))?;

    // 典型 MNIST 第一层
    let fc = linear(&mut graph, input, 784, 256, batch_size, Some("fc"))?;

    // 验证参数形状
    let w_shape = graph.get_node(fc.weights)?.value_expected_shape();
    let b_shape = graph.get_node(fc.bias)?.value_expected_shape();

    assert_eq!(w_shape, &[784, 256]);
    assert_eq!(b_shape, &[1, 256]);

    Ok(())
}

/// 测试访问 linear() 内部参数
#[test]
fn test_linear_access_internal_params() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 4;
    let input = graph.new_input_node(&[batch_size, 4], Some("input"))?;

    let fc = linear(&mut graph, input, 4, 2, batch_size, Some("fc"))?;

    // 应该能访问并修改权重
    let custom_weights = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]);
    graph.set_node_value(fc.weights, Some(&custom_weights))?;

    let w = graph.get_node_value(fc.weights)?.unwrap();
    assert_abs_diff_eq!(w[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(w[[3, 1]], 8.0, epsilon = 1e-6);

    Ok(())
}

// ==================== PyTorch 数值对照测试 ====================

/// 测试前向传播数值（与 PyTorch 对照）
#[test]
fn test_linear_forward_pytorch_comparison() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 2;
    let in_features = 3;
    let out_features = 4;

    let input = graph.new_input_node(&[batch_size, in_features], Some("input"))?;
    let fc = linear(
        &mut graph,
        input,
        in_features,
        out_features,
        batch_size,
        Some("fc"),
    )?;

    // 设置与 PyTorch 相同的参数
    graph.set_node_value(
        input,
        Some(&Tensor::new(PYTORCH_FWD_X, &[batch_size, in_features])),
    )?;
    graph.set_node_value(
        fc.weights,
        Some(&Tensor::new(PYTORCH_FWD_W, &[in_features, out_features])),
    )?;
    graph.set_node_value(
        fc.bias,
        Some(&Tensor::new(PYTORCH_FWD_B, &[1, out_features])),
    )?;

    // 前向传播
    graph.forward(fc.output)?;

    // 验证输出
    let output = graph.get_node_value(fc.output)?.unwrap();
    assert_eq!(output.shape(), &[batch_size, out_features]);

    let output_data = output.data_as_slice();
    for (i, (&actual, &expected)) in output_data
        .iter()
        .zip(PYTORCH_FWD_OUTPUT.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
        println!(
            "output[{}]: actual={:.6}, expected={:.6}",
            i, actual, expected
        );
    }

    println!("✅ Linear 前向传播与 PyTorch 一致");
    Ok(())
}

/// 测试反向传播梯度数值（与 PyTorch 对照）
#[test]
fn test_linear_backward_pytorch_comparison() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 2;
    let in_features = 3;
    let out_features = 2;

    // 构建网络: input -> linear -> mse_loss
    let input = graph.new_input_node(&[batch_size, in_features], Some("input"))?;
    let fc = linear(
        &mut graph,
        input,
        in_features,
        out_features,
        batch_size,
        Some("fc"),
    )?;
    let target = graph.new_input_node(&[batch_size, out_features], Some("target"))?;
    let loss = graph.new_mse_loss_node(fc.output, target, Some("loss"))?;

    // 设置与 PyTorch 相同的参数
    graph.set_node_value(
        input,
        Some(&Tensor::new(PYTORCH_BWD_X, &[batch_size, in_features])),
    )?;
    graph.set_node_value(
        fc.weights,
        Some(&Tensor::new(PYTORCH_BWD_W, &[in_features, out_features])),
    )?;
    graph.set_node_value(
        fc.bias,
        Some(&Tensor::new(PYTORCH_BWD_B, &[1, out_features])),
    )?;
    graph.set_node_value(
        target,
        Some(&Tensor::new(
            PYTORCH_BWD_TARGET,
            &[batch_size, out_features],
        )),
    )?;

    // 前向传播
    graph.forward(loss)?;

    // 验证输出
    let output = graph.get_node_value(fc.output)?.unwrap();
    let output_data = output.data_as_slice();
    println!("输出:");
    for (i, (&actual, &expected)) in output_data
        .iter()
        .zip(PYTORCH_BWD_OUTPUT.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
        println!(
            "  output[{}]: actual={:.6}, expected={:.6}",
            i, actual, expected
        );
    }

    // 验证 loss
    let loss_val = graph.get_node_value(loss)?.unwrap();
    assert_abs_diff_eq!(loss_val[[0, 0]], PYTORCH_BWD_LOSS, epsilon = 1e-3);
    println!(
        "loss: actual={:.6}, expected={:.6}",
        loss_val[[0, 0]],
        PYTORCH_BWD_LOSS
    );

    // 反向传播
    graph.backward(loss)?;

    // 验证权重梯度
    let grad_w = graph.get_node_grad_ref(fc.weights)?.unwrap();
    let grad_w_data = grad_w.data_as_slice();
    println!("\n权重梯度:");
    for (i, (&actual, &expected)) in grad_w_data
        .iter()
        .zip(PYTORCH_BWD_GRAD_W.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-3);
        println!(
            "  grad_w[{}]: actual={:.6}, expected={:.6}",
            i, actual, expected
        );
    }

    // 验证偏置梯度
    let grad_b = graph.get_node_grad_ref(fc.bias)?.unwrap();
    let grad_b_data = grad_b.data_as_slice();
    println!("\n偏置梯度:");
    for (i, (&actual, &expected)) in grad_b_data
        .iter()
        .zip(PYTORCH_BWD_GRAD_B.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-3);
        println!(
            "  grad_b[{}]: actual={:.6}, expected={:.6}",
            i, actual, expected
        );
    }

    println!("\n✅ Linear 反向传播梯度与 PyTorch 一致");
    Ok(())
}

/// 测试两层网络反向传播（与 PyTorch 对照）
#[test]
fn test_linear_chain_backward_pytorch_comparison() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);
    let batch_size = 2;
    let in_features = 4;
    let hidden_features = 3;
    let out_features = 2;

    // 构建网络: input -> fc1 -> relu -> fc2 -> softmax_cross_entropy
    let input = graph.new_input_node(&[batch_size, in_features], Some("input"))?;
    let fc1 = linear(
        &mut graph,
        input,
        in_features,
        hidden_features,
        batch_size,
        Some("fc1"),
    )?;
    let relu = graph.new_leaky_relu_node(fc1.output, 0.0, Some("relu"))?;
    let fc2 = linear(
        &mut graph,
        relu,
        hidden_features,
        out_features,
        batch_size,
        Some("fc2"),
    )?;
    let target = graph.new_input_node(&[batch_size, out_features], Some("target"))?;
    let loss = graph.new_softmax_cross_entropy_node(fc2.output, target, Some("loss"))?;

    // 设置与 PyTorch 相同的参数
    graph.set_node_value(
        input,
        Some(&Tensor::new(PYTORCH_CHAIN_X, &[batch_size, in_features])),
    )?;
    graph.set_node_value(
        fc1.weights,
        Some(&Tensor::new(
            PYTORCH_CHAIN_W1,
            &[in_features, hidden_features],
        )),
    )?;
    graph.set_node_value(
        fc1.bias,
        Some(&Tensor::new(PYTORCH_CHAIN_B1, &[1, hidden_features])),
    )?;
    graph.set_node_value(
        fc2.weights,
        Some(&Tensor::new(
            PYTORCH_CHAIN_W2,
            &[hidden_features, out_features],
        )),
    )?;
    graph.set_node_value(
        fc2.bias,
        Some(&Tensor::new(PYTORCH_CHAIN_B2, &[1, out_features])),
    )?;
    graph.set_node_value(
        target,
        Some(&Tensor::new(
            PYTORCH_CHAIN_TARGET,
            &[batch_size, out_features],
        )),
    )?;

    // 前向传播
    graph.forward(loss)?;

    // 验证隐藏层 (ReLU 输出)
    let h1_relu = graph.get_node_value(relu)?.unwrap();
    let h1_data = h1_relu.data_as_slice();
    println!("隐藏层 ReLU 输出:");
    for (i, (&actual, &expected)) in h1_data.iter().zip(PYTORCH_CHAIN_H1_RELU.iter()).enumerate() {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
        println!(
            "  h1[{}]: actual={:.6}, expected={:.6}",
            i, actual, expected
        );
    }

    // 验证 logits
    let logits = graph.get_node_value(fc2.output)?.unwrap();
    let logits_data = logits.data_as_slice();
    println!("\nLogits:");
    for (i, (&actual, &expected)) in logits_data
        .iter()
        .zip(PYTORCH_CHAIN_LOGITS.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-3);
        println!(
            "  logits[{}]: actual={:.6}, expected={:.6}",
            i, actual, expected
        );
    }

    // 验证 loss
    let loss_val = graph.get_node_value(loss)?.unwrap();
    assert_abs_diff_eq!(loss_val[[0, 0]], PYTORCH_CHAIN_LOSS, epsilon = 1e-3);
    println!(
        "\nloss: actual={:.6}, expected={:.6}",
        loss_val[[0, 0]],
        PYTORCH_CHAIN_LOSS
    );

    // 反向传播
    graph.backward(loss)?;

    // 验证 fc1 权重梯度
    let grad_w1 = graph.get_node_grad_ref(fc1.weights)?.unwrap();
    let grad_w1_data = grad_w1.data_as_slice();
    println!("\nfc1 权重梯度:");
    for (i, (&actual, &expected)) in grad_w1_data
        .iter()
        .zip(PYTORCH_CHAIN_GRAD_W1.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-3);
        println!(
            "  grad_w1[{}]: actual={:.8}, expected={:.8}",
            i, actual, expected
        );
    }

    // 验证 fc1 偏置梯度
    let grad_b1 = graph.get_node_grad_ref(fc1.bias)?.unwrap();
    let grad_b1_data = grad_b1.data_as_slice();
    println!("\nfc1 偏置梯度:");
    for (i, (&actual, &expected)) in grad_b1_data
        .iter()
        .zip(PYTORCH_CHAIN_GRAD_B1.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-3);
        println!(
            "  grad_b1[{}]: actual={:.8}, expected={:.8}",
            i, actual, expected
        );
    }

    // 验证 fc2 权重梯度
    let grad_w2 = graph.get_node_grad_ref(fc2.weights)?.unwrap();
    let grad_w2_data = grad_w2.data_as_slice();
    println!("\nfc2 权重梯度:");
    for (i, (&actual, &expected)) in grad_w2_data
        .iter()
        .zip(PYTORCH_CHAIN_GRAD_W2.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-3);
        println!(
            "  grad_w2[{}]: actual={:.8}, expected={:.8}",
            i, actual, expected
        );
    }

    // 验证 fc2 偏置梯度
    let grad_b2 = graph.get_node_grad_ref(fc2.bias)?.unwrap();
    let grad_b2_data = grad_b2.data_as_slice();
    println!("\nfc2 偏置梯度:");
    for (i, (&actual, &expected)) in grad_b2_data
        .iter()
        .zip(PYTORCH_CHAIN_GRAD_B2.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-3);
        println!(
            "  grad_b2[{}]: actual={:.8}, expected={:.8}",
            i, actual, expected
        );
    }

    println!("\n✅ 两层 Linear 网络反向传播与 PyTorch 一致");
    Ok(())
}
