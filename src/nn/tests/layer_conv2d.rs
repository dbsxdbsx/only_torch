/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : Conv2d layer 单元测试（Batch-First 设计）
 */

use crate::nn::layer::conv2d;
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试 conv2d() 创建
#[test]
fn test_conv2d_creation() -> Result<(), GraphError> {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 1, 28, 28], Some("input"))?;

    let conv = conv2d(&mut graph, input, 1, 32, (3, 3), (1, 1), (1, 1), Some("conv1"))?;

    // 验证返回的节点 ID 都有效
    assert!(graph.get_node_value(conv.output).is_ok());
    assert!(graph.get_node_value(conv.kernel).is_ok());

    Ok(())
}

/// 测试 conv2d() 参数形状
#[test]
fn test_conv2d_shapes() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let input = graph.new_input_node(&[4, 3, 28, 28], Some("input"))?;

    // 3->16 通道，5x5 核
    let conv = conv2d(&mut graph, input, 3, 16, (5, 5), (1, 1), (2, 2), Some("conv1"))?;

    // 检查卷积核形状：[out_channels, in_channels, kH, kW]
    let k_shape = graph.get_node(conv.kernel)?.value_expected_shape();
    assert_eq!(k_shape, &[16, 3, 5, 5]);

    Ok(())
}

/// 测试 conv2d() 前向传播
#[test]
fn test_conv2d_forward() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);

    // 输入: [batch=1, C_in=1, H=4, W=4]
    let input = graph.new_input_node(&[1, 1, 4, 4], Some("input"))?;
    let conv = conv2d(&mut graph, input, 1, 2, (2, 2), (1, 1), (0, 0), Some("conv1"))?;

    // 设置输入全 1
    let x = Tensor::ones(&[1, 1, 4, 4]);
    // 设置卷积核全 1
    let k = Tensor::ones(&[2, 1, 2, 2]);

    graph.set_node_value(input, Some(&x))?;
    graph.set_node_value(conv.kernel, Some(&k))?;

    // 前向传播
    graph.forward_node(conv.output)?;

    // 验证输出形状: [1, 2, 3, 3]
    // H' = (4 + 0 - 2) / 1 + 1 = 3
    let output = graph.get_node_value(conv.output)?.unwrap();
    assert_eq!(output.shape(), &[1, 2, 3, 3]);

    // 验证输出值（全 1 输入，全 1 卷积核，2x2 窗口 → 每个输出为 4.0）
    for c in 0..2 {
        for h in 0..3 {
            for w in 0..3 {
                assert_abs_diff_eq!(output[[0, c, h, w]], 4.0, epsilon = 1e-6);
            }
        }
    }

    Ok(())
}

/// 测试 conv2d() 输出尺寸计算
#[test]
fn test_conv2d_output_size() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);

    // padding=1, kernel=3x3, stride=1 → same padding (保持尺寸)
    let input = graph.new_input_node(&[1, 1, 4, 4], Some("input"))?;
    let conv = conv2d(&mut graph, input, 1, 1, (3, 3), (1, 1), (1, 1), Some("conv1"))?;

    let x = Tensor::ones(&[1, 1, 4, 4]);
    let k = Tensor::ones(&[1, 1, 3, 3]);
    graph.set_node_value(input, Some(&x))?;
    graph.set_node_value(conv.kernel, Some(&k))?;

    graph.forward_node(conv.output)?;

    // 验证输出形状: [1, 1, 4, 4]（same padding）
    let output = graph.get_node_value(conv.output)?.unwrap();
    assert_eq!(output.shape(), &[1, 1, 4, 4]);

    Ok(())
}

// ==================== 节点名称测试 ====================

/// 测试 conv2d() 带名称
#[test]
fn test_conv2d_with_name() -> Result<(), GraphError> {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 1, 28, 28], Some("input"))?;

    let conv = conv2d(&mut graph, input, 1, 32, (3, 3), (1, 1), (1, 1), Some("encoder_conv1"))?;

    // 验证节点名称
    assert_eq!(graph.get_node(conv.kernel)?.name(), "encoder_conv1_K");
    assert_eq!(graph.get_node(conv.output)?.name(), "encoder_conv1_out");

    Ok(())
}

/// 测试 conv2d() 无名称（使用默认前缀）
#[test]
fn test_conv2d_without_name() -> Result<(), GraphError> {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 1, 28, 28], Some("input"))?;

    let conv = conv2d(&mut graph, input, 1, 32, (3, 3), (1, 1), (1, 1), None)?;

    // 验证使用默认前缀 "conv2d"
    assert_eq!(graph.get_node(conv.kernel)?.name(), "conv2d_K");
    assert_eq!(graph.get_node(conv.output)?.name(), "conv2d_out");

    Ok(())
}

// ==================== 链式连接测试 ====================

/// 测试多层 conv2d() 链式连接
#[test]
fn test_conv2d_chain() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);

    // 典型 CNN 结构: conv1 -> relu -> conv2 -> relu
    let input = graph.new_input_node(&[2, 1, 8, 8], Some("input"))?;
    let conv1 = conv2d(&mut graph, input, 1, 4, (3, 3), (1, 1), (1, 1), Some("conv1"))?;
    let relu1 = graph.new_leaky_relu_node(conv1.output, 0.0, Some("relu1"))?;
    let conv2 = conv2d(&mut graph, relu1, 4, 8, (3, 3), (1, 1), (1, 1), Some("conv2"))?;
    let relu2 = graph.new_leaky_relu_node(conv2.output, 0.0, Some("relu2"))?;

    // 设置输入
    let x = Tensor::normal(0.0, 1.0, &[2, 1, 8, 8]);
    graph.set_node_value(input, Some(&x))?;

    // 前向传播
    graph.forward_node(relu2)?;

    // 验证输出存在且形状正确 (same padding)
    let output = graph.get_node_value(relu2)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[2, 8, 8, 8]);

    Ok(())
}

/// 测试 conv2d + flatten 链式连接（典型 CNN 末端结构）
#[test]
fn test_conv2d_with_flatten() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);

    // conv -> flatten
    let input = graph.new_input_node(&[2, 1, 4, 4], Some("input"))?;
    let conv = conv2d(&mut graph, input, 1, 2, (2, 2), (1, 1), (0, 0), Some("conv"))?;
    let flat = graph.new_flatten_node(conv.output, true, Some("flat"))?;

    // 设置输入
    let x = Tensor::ones(&[2, 1, 4, 4]);
    graph.set_node_value(input, Some(&x))?;

    // 前向传播
    graph.forward_node(flat)?;

    // 验证展平输出: [2, 2*3*3] = [2, 18]
    let output = graph.get_node_value(flat)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[2, 18]);

    Ok(())
}

// ==================== 反向传播测试 ====================

/// 测试 conv2d() 与 Batch 反向传播
#[test]
fn test_conv2d_batch_backward() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 2;

    // 构建网络: conv -> flatten -> matmul (简化的 CNN 分类)
    let input = graph.new_input_node(&[batch_size, 1, 4, 4], Some("input"))?;
    let conv = conv2d(&mut graph, input, 1, 2, (2, 2), (1, 1), (0, 0), Some("conv"))?;
    let flat = graph.new_flatten_node(conv.output, true, Some("flat"))?;
    // flat 输出: [2, 18]

    // 简单分类器: flat -> matmul -> softmax_cross_entropy
    let fc_weight = graph.new_parameter_node(&[18, 3], Some("fc_w"))?;
    let logits = graph.new_mat_mul_node(flat, fc_weight, Some("logits"))?;

    // SoftmaxCrossEntropy Loss
    let labels = graph.new_input_node(&[batch_size, 3], Some("labels"))?;
    let loss = graph.new_softmax_cross_entropy_node(logits, labels, Some("loss"))?;

    // 设置数据
    let x = Tensor::normal(0.0, 1.0, &[batch_size, 1, 4, 4]);
    let y = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[batch_size, 3]); // one-hot

    graph.set_node_value(input, Some(&x))?;
    graph.set_node_value(labels, Some(&y))?;

    // Batch 训练
    graph.forward_batch(loss)?;
    graph.backward_batch(loss)?;

    // 验证卷积核有梯度
    let k_grad = graph.get_node_grad_batch(conv.kernel)?;
    assert!(k_grad.is_some());

    // 检查梯度形状
    assert_eq!(k_grad.unwrap().shape(), &[2, 1, 2, 2]);

    Ok(())
}

/// 测试多层 conv2d() 的 Batch 训练
#[test]
fn test_conv2d_chain_batch_training() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 2;

    // 构建网络: conv1 -> relu -> conv2 -> flatten -> loss
    let input = graph.new_input_node(&[batch_size, 1, 6, 6], Some("input"))?;
    let conv1 = conv2d(&mut graph, input, 1, 2, (3, 3), (1, 1), (0, 0), Some("conv1"))?;
    let relu1 = graph.new_leaky_relu_node(conv1.output, 0.0, Some("relu1"))?;
    // conv1 输出: [2, 2, 4, 4]

    let conv2 = conv2d(&mut graph, relu1, 2, 4, (2, 2), (1, 1), (0, 0), Some("conv2"))?;
    // conv2 输出: [2, 4, 3, 3]

    let flat = graph.new_flatten_node(conv2.output, true, Some("flat"))?;
    // flat 输出: [2, 36]

    // 分类器: flat -> matmul -> softmax_cross_entropy
    let fc_weight = graph.new_parameter_node(&[36, 4], Some("fc_w"))?;
    let logits = graph.new_mat_mul_node(flat, fc_weight, Some("logits"))?;

    // SoftmaxCrossEntropy Loss
    let labels = graph.new_input_node(&[batch_size, 4], Some("labels"))?;
    let loss = graph.new_softmax_cross_entropy_node(logits, labels, Some("loss"))?;

    // 设置数据
    let x = Tensor::normal(0.0, 1.0, &[batch_size, 1, 6, 6]);
    let y = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[batch_size, 4]); // one-hot

    graph.set_node_value(input, Some(&x))?;
    graph.set_node_value(labels, Some(&y))?;

    // Batch 训练
    graph.forward_batch(loss)?;
    graph.backward_batch(loss)?;

    // 验证所有卷积核参数都有梯度
    assert!(graph.get_node_grad_batch(conv1.kernel)?.is_some());
    assert!(graph.get_node_grad_batch(conv2.kernel)?.is_some());

    Ok(())
}

// ==================== 名称冲突测试 ====================

/// 测试多个 conv2d() 使用不同名称
#[test]
fn test_conv2d_multiple_layers_different_names() -> Result<(), GraphError> {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 1, 16, 16], Some("input"))?;

    let conv1 = conv2d(&mut graph, input, 1, 4, (3, 3), (1, 1), (1, 1), Some("conv1"))?;
    let conv2 = conv2d(&mut graph, conv1.output, 4, 8, (3, 3), (1, 1), (1, 1), Some("conv2"))?;
    let conv3 = conv2d(&mut graph, conv2.output, 8, 16, (3, 3), (1, 1), (1, 1), Some("conv3"))?;

    // 验证各层节点独立存在
    assert!(graph.get_node_value(conv1.kernel).is_ok());
    assert!(graph.get_node_value(conv2.kernel).is_ok());
    assert!(graph.get_node_value(conv3.kernel).is_ok());

    // 验证节点名称正确
    assert_eq!(graph.get_node(conv1.kernel)?.name(), "conv1_K");
    assert_eq!(graph.get_node(conv2.kernel)?.name(), "conv2_K");
    assert_eq!(graph.get_node(conv3.kernel)?.name(), "conv3_K");

    Ok(())
}

/// 测试重复名称应该报错
#[test]
fn test_conv2d_duplicate_name_error() {
    let mut graph = Graph::new();
    let input = graph
        .new_input_node(&[2, 1, 8, 8], Some("input"))
        .unwrap();

    // 第一个 conv 成功
    let conv1 = conv2d(&mut graph, input, 1, 4, (3, 3), (1, 1), (1, 1), Some("conv"));
    assert!(conv1.is_ok());

    // 第二个 conv 使用相同名称，应该失败
    let conv2 = conv2d(
        &mut graph,
        conv1.unwrap().output,
        4,
        8,
        (3, 3),
        (1, 1),
        (1, 1),
        Some("conv"),
    );
    assert!(conv2.is_err());

    // 验证错误类型
    if let Err(e) = conv2 {
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
fn test_conv2d_multiple_unnamed_layers_conflict() {
    let mut graph = Graph::new();
    let input = graph
        .new_input_node(&[2, 1, 8, 8], Some("input"))
        .unwrap();

    // 第一个无名称层成功
    let conv1 = conv2d(&mut graph, input, 1, 4, (3, 3), (1, 1), (1, 1), None);
    assert!(conv1.is_ok());

    // 第二个无名称层应该失败（名称冲突）
    let conv2 = conv2d(
        &mut graph,
        conv1.unwrap().output,
        4,
        8,
        (3, 3),
        (1, 1),
        (1, 1),
        None,
    );
    assert!(conv2.is_err(), "多个无名称 conv2d 层应该因名称冲突而失败");
}

// ==================== 边界维度测试 ====================

/// 测试单通道输入输出
#[test]
fn test_conv2d_single_channel() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let input = graph.new_input_node(&[2, 1, 4, 4], Some("input"))?;

    let conv = conv2d(&mut graph, input, 1, 1, (2, 2), (1, 1), (0, 0), Some("conv"))?;

    // 设置输入
    let x = Tensor::ones(&[2, 1, 4, 4]);
    graph.set_node_value(input, Some(&x))?;

    // 前向传播
    graph.forward_node(conv.output)?;

    let output = graph.get_node_value(conv.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[2, 1, 3, 3]);

    Ok(())
}

/// 测试大通道数
#[test]
fn test_conv2d_large_channels() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let input = graph.new_input_node(&[1, 64, 8, 8], Some("input"))?;

    let conv = conv2d(&mut graph, input, 64, 128, (3, 3), (1, 1), (1, 1), Some("conv"))?;

    // 验证参数形状
    let k_shape = graph.get_node(conv.kernel)?.value_expected_shape();
    assert_eq!(k_shape, &[128, 64, 3, 3]);

    Ok(())
}

/// 测试带 stride 的 conv2d
#[test]
fn test_conv2d_with_stride() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let input = graph.new_input_node(&[2, 1, 8, 8], Some("input"))?;

    // stride=2 会使输出尺寸减半
    let conv = conv2d(&mut graph, input, 1, 4, (3, 3), (2, 2), (1, 1), Some("conv"))?;

    let x = Tensor::ones(&[2, 1, 8, 8]);
    graph.set_node_value(input, Some(&x))?;

    graph.forward_node(conv.output)?;

    // 输出尺寸: (8 + 2 - 3) / 2 + 1 = 4
    let output = graph.get_node_value(conv.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[2, 4, 4, 4]);

    Ok(())
}

/// 测试非方形卷积核
#[test]
fn test_conv2d_nonsquare_kernel() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let input = graph.new_input_node(&[2, 1, 8, 8], Some("input"))?;

    // 使用 3x5 的非方形卷积核
    let conv = conv2d(&mut graph, input, 1, 4, (3, 5), (1, 1), (1, 2), Some("conv"))?;

    // 验证卷积核形状
    let k_shape = graph.get_node(conv.kernel)?.value_expected_shape();
    assert_eq!(k_shape, &[4, 1, 3, 5]);

    let x = Tensor::ones(&[2, 1, 8, 8]);
    graph.set_node_value(input, Some(&x))?;

    graph.forward_node(conv.output)?;

    // 输出尺寸:
    // H' = (8 + 2*1 - 3) / 1 + 1 = 8
    // W' = (8 + 2*2 - 5) / 1 + 1 = 8
    let output = graph.get_node_value(conv.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[2, 4, 8, 8]);

    Ok(())
}

/// 测试访问 conv2d() 内部参数
#[test]
fn test_conv2d_access_internal_params() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let input = graph.new_input_node(&[2, 1, 4, 4], Some("input"))?;

    let conv = conv2d(&mut graph, input, 1, 2, (2, 2), (1, 1), (0, 0), Some("conv"))?;

    // 应该能访问并修改卷积核
    let custom_kernel = Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 1, 2, 2],
    );
    graph.set_node_value(conv.kernel, Some(&custom_kernel))?;

    let k = graph.get_node_value(conv.kernel)?.unwrap();
    assert_abs_diff_eq!(k[[0, 0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(k[[1, 0, 1, 1]], 8.0, epsilon = 1e-6);

    Ok(())
}

/// 测试典型 MNIST CNN 配置
#[test]
fn test_conv2d_mnist_like() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 32;

    // 典型 MNIST CNN 第一层
    let input = graph.new_input_node(&[batch_size, 1, 28, 28], Some("input"))?;
    let conv = conv2d(&mut graph, input, 1, 32, (5, 5), (1, 1), (2, 2), Some("conv1"))?;

    // 验证参数形状
    let k_shape = graph.get_node(conv.kernel)?.value_expected_shape();
    assert_eq!(k_shape, &[32, 1, 5, 5]);

    Ok(())
}

// ==================== Bias 功能测试 ====================

/// 测试 conv2d() 默认包含 bias
#[test]
fn test_conv2d_has_bias() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let input = graph.new_input_node(&[2, 1, 4, 4], Some("input"))?;

    let conv = conv2d(&mut graph, input, 1, 2, (2, 2), (1, 1), (0, 0), Some("conv"))?;

    // 验证 bias 存在且形状正确
    let b_shape = graph.get_node(conv.bias)?.value_expected_shape();
    assert_eq!(b_shape, &[1, 2]);

    // 验证 bias 默认初始化为 0
    let bias = graph.get_node_value(conv.bias)?.unwrap();
    assert_abs_diff_eq!(bias[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(bias[[0, 1]], 0.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 conv2d() bias 正确应用
#[test]
fn test_conv2d_bias_applied() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let input = graph.new_input_node(&[1, 1, 3, 3], Some("input"))?;

    let conv = conv2d(&mut graph, input, 1, 2, (2, 2), (1, 1), (0, 0), Some("conv"))?;

    // 设置输入全 1，卷积核全 1
    let x = Tensor::ones(&[1, 1, 3, 3]);
    let k = Tensor::ones(&[2, 1, 2, 2]);

    graph.set_node_value(input, Some(&x))?;
    graph.set_node_value(conv.kernel, Some(&k))?;

    // 先测试 bias=0 的情况
    graph.forward_node(conv.output)?;
    let output_no_bias = graph.get_node_value(conv.output)?.unwrap().clone();

    // 全 1 输入，全 1 卷积核，2x2 窗口 → 每个输出为 4.0
    for c in 0..2 {
        for h in 0..2 {
            for w in 0..2 {
                assert_abs_diff_eq!(output_no_bias[[0, c, h, w]], 4.0, epsilon = 1e-6);
            }
        }
    }

    // 设置 bias：通道 0 加 1.0，通道 1 加 2.0
    let bias = Tensor::new(&[1.0, 2.0], &[1, 2]);
    graph.set_node_value(conv.bias, Some(&bias))?;

    // 重新前向传播
    graph.forward_node(conv.output)?;
    let output_with_bias = graph.get_node_value(conv.output)?.unwrap();

    // 验证 bias 被正确加上：通道 0 应该是 5.0，通道 1 应该是 6.0
    for h in 0..2 {
        for w in 0..2 {
            assert_abs_diff_eq!(output_with_bias[[0, 0, h, w]], 5.0, epsilon = 1e-6);
            assert_abs_diff_eq!(output_with_bias[[0, 1, h, w]], 6.0, epsilon = 1e-6);
        }
    }

    Ok(())
}

/// 测试 conv2d() bias 的梯度传播
#[test]
fn test_conv2d_bias_gradient() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 2;

    let input = graph.new_input_node(&[batch_size, 1, 4, 4], Some("input"))?;
    let conv = conv2d(&mut graph, input, 1, 2, (2, 2), (1, 1), (0, 0), Some("conv"))?;
    let flat = graph.new_flatten_node(conv.output, true, Some("flat"))?;

    // 简单分类器
    let fc_weight = graph.new_parameter_node(&[18, 3], Some("fc_w"))?;
    let logits = graph.new_mat_mul_node(flat, fc_weight, Some("logits"))?;
    let labels = graph.new_input_node(&[batch_size, 3], Some("labels"))?;
    let loss = graph.new_softmax_cross_entropy_node(logits, labels, Some("loss"))?;

    // 设置数据
    let x = Tensor::normal(0.0, 1.0, &[batch_size, 1, 4, 4]);
    let y = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[batch_size, 3]);

    graph.set_node_value(input, Some(&x))?;
    graph.set_node_value(labels, Some(&y))?;

    // Batch 训练
    graph.forward_batch(loss)?;
    graph.backward_batch(loss)?;

    // 验证 bias 有梯度
    let b_grad = graph.get_node_grad_batch(conv.bias)?;
    assert!(b_grad.is_some(), "bias 应该有梯度");
    assert_eq!(b_grad.unwrap().shape(), &[1, 2], "bias 梯度形状应该正确");

    Ok(())
}

/// 测试 conv2d() bias 节点命名
#[test]
fn test_conv2d_bias_naming() -> Result<(), GraphError> {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 1, 8, 8], Some("input"))?;

    let conv = conv2d(&mut graph, input, 1, 4, (3, 3), (1, 1), (1, 1), Some("encoder_conv1"))?;

    // 验证 bias 节点名称
    assert_eq!(graph.get_node(conv.bias)?.name(), "encoder_conv1_b");

    Ok(())
}

