/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : Linear layer 单元测试（Batch-First 设计）
 */

use crate::nn::layer::linear;
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试 linear() 创建
#[test]
fn test_linear_creation() -> Result<(), GraphError> {
    let mut graph = Graph::new();
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
    let mut graph = Graph::new_with_seed(42);
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
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 2;

    let input = graph.new_input_node(&[batch_size, 3], Some("input"))?;
    let fc = linear(&mut graph, input, 3, 2, batch_size, Some("fc"))?;

    // 设置输入: [2, 3]
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    // 设置权重: [3, 2] - 单位矩阵的前两列
    let w = Tensor::new(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[3, 2]);
    // 设置偏置: [1, 2]
    let b = Tensor::new(&[0.5, 0.5], &[1, 2]);
    // 设置 ones: [2, 1]
    let ones = Tensor::ones(&[2, 1]);

    graph.set_node_value(input, Some(&x))?;
    graph.set_node_value(fc.weights, Some(&w))?;
    graph.set_node_value(fc.bias, Some(&b))?;
    graph.set_node_value(fc.ones, Some(&ones))?;

    // 前向传播
    graph.forward_node(fc.output)?;

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
    let mut graph = Graph::new();
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
    let mut graph = Graph::new();
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
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 8;

    let input = graph.new_input_node(&[batch_size, 4], Some("input"))?;
    let fc1 = linear(&mut graph, input, 4, 8, batch_size, Some("fc1"))?;
    let act = graph.new_sigmoid_node(fc1.output, Some("act"))?;
    let fc2 = linear(&mut graph, act, 8, 2, batch_size, Some("fc2"))?;

    // 设置输入和 ones
    let x = Tensor::normal(0.0, 1.0, &[batch_size, 4]);
    let ones = Tensor::ones(&[batch_size, 1]);

    graph.set_node_value(input, Some(&x))?;
    graph.set_node_value(fc1.ones, Some(&ones))?;
    graph.set_node_value(fc2.ones, Some(&ones))?;

    // 前向传播
    graph.forward_node(fc2.output)?;

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
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 4;

    // 构建网络
    let input = graph.new_input_node(&[batch_size, 3], Some("input"))?;
    let fc = linear(&mut graph, input, 3, 2, batch_size, Some("fc"))?;
    let labels = graph.new_input_node(&[batch_size, 2], Some("labels"))?;
    let loss = graph.new_softmax_cross_entropy_node(fc.output, labels, Some("loss"))?;

    // 设置数据
    let x = Tensor::normal(0.0, 1.0, &[batch_size, 3]);
    let y = Tensor::new(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[batch_size, 2]); // one-hot
    let ones = Tensor::ones(&[batch_size, 1]);

    graph.set_node_value(input, Some(&x))?;
    graph.set_node_value(labels, Some(&y))?;
    graph.set_node_value(fc.ones, Some(&ones))?;

    // Batch 训练
    graph.forward_batch(loss)?;
    graph.backward_batch(loss)?;

    // 验证权重和偏置都有梯度
    let w_grad = graph.get_node_grad_batch(fc.weights)?;
    let b_grad = graph.get_node_grad_batch(fc.bias)?;
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
    let mut graph = Graph::new_with_seed(42);
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
    let ones = Tensor::ones(&[batch_size, 1]);

    graph.set_node_value(input, Some(&x))?;
    graph.set_node_value(labels, Some(&y))?;
    graph.set_node_value(fc1.ones, Some(&ones))?;
    graph.set_node_value(fc2.ones, Some(&ones))?;

    // Batch 训练
    graph.forward_batch(loss)?;
    graph.backward_batch(loss)?;

    // 验证所有参数都有梯度
    assert!(graph.get_node_grad_batch(fc1.weights)?.is_some());
    assert!(graph.get_node_grad_batch(fc1.bias)?.is_some());
    assert!(graph.get_node_grad_batch(fc2.weights)?.is_some());
    assert!(graph.get_node_grad_batch(fc2.bias)?.is_some());

    Ok(())
}

// ==================== 名称冲突测试 ====================

/// 测试多个 linear() 使用不同名称
#[test]
fn test_linear_multiple_layers_different_names() -> Result<(), GraphError> {
    let mut graph = Graph::new();
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
    let mut graph = Graph::new();
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
    let mut graph = Graph::new();
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
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 4;
    let input = graph.new_input_node(&[batch_size, 1], Some("input"))?;

    let fc = linear(&mut graph, input, 1, 4, batch_size, Some("fc"))?;

    // 设置输入和 ones
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[batch_size, 1]);
    let ones = Tensor::ones(&[batch_size, 1]);
    graph.set_node_value(input, Some(&x))?;
    graph.set_node_value(fc.ones, Some(&ones))?;

    // 前向传播
    graph.forward_node(fc.output)?;

    let output = graph.get_node_value(fc.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[batch_size, 4]);

    Ok(())
}

/// 测试单特征输出
#[test]
fn test_linear_single_output_feature() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 4;
    let input = graph.new_input_node(&[batch_size, 4], Some("input"))?;

    let fc = linear(&mut graph, input, 4, 1, batch_size, Some("fc"))?;

    // 设置输入和 ones
    let x = Tensor::normal(0.0, 1.0, &[batch_size, 4]);
    let ones = Tensor::ones(&[batch_size, 1]);
    graph.set_node_value(input, Some(&x))?;
    graph.set_node_value(fc.ones, Some(&ones))?;

    // 前向传播
    graph.forward_node(fc.output)?;

    let output = graph.get_node_value(fc.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[batch_size, 1]);

    Ok(())
}

/// 测试大维度 linear()（典型 MNIST 配置）
#[test]
fn test_linear_large_dimensions() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
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
    let mut graph = Graph::new_with_seed(42);
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
