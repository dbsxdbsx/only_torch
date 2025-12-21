/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : AvgPool2d layer 单元测试（Batch-First 设计）
 */

use crate::nn::layer::avg_pool2d;
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试 avg_pool2d() 创建
#[test]
fn test_avg_pool2d_creation() -> Result<(), GraphError> {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 32, 14, 14], Some("input"))?;

    let pool = avg_pool2d(&mut graph, input, (2, 2), Some((2, 2)), Some("pool1"))?;

    // 验证返回的节点 ID 有效
    assert!(graph.get_node_value(pool.output).is_ok());

    Ok(())
}

/// 测试 avg_pool2d() 输出形状
#[test]
fn test_avg_pool2d_shapes() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let input = graph.new_input_node(&[4, 16, 28, 28], Some("input"))?;

    // 2x2 池化, stride=2 → 尺寸减半
    let pool = avg_pool2d(&mut graph, input, (2, 2), Some((2, 2)), Some("pool1"))?;

    // 检查输出形状预期: [4, 16, 14, 14]
    let out_shape = graph.get_node(pool.output)?.value_expected_shape();
    assert_eq!(out_shape, &[4, 16, 14, 14]);

    Ok(())
}

/// 测试 avg_pool2d() 前向传播
#[test]
fn test_avg_pool2d_forward() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);

    // 输入: [batch=1, C=1, H=4, W=4]
    let input = graph.new_input_node(&[1, 1, 4, 4], Some("input"))?;

    // 创建特定的输入数据
    #[rustfmt::skip]
    let input_data = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ],
        &[1, 1, 4, 4],
    );
    graph.set_node_value(input, Some(&input_data))?;

    // 创建 avg_pool2d 层: 2x2 核, stride=2
    let pool = avg_pool2d(&mut graph, input, (2, 2), Some((2, 2)), Some("pool1"))?;

    // 前向传播
    graph.forward_node(pool.output)?;

    // 验证输出形状: [1, 1, 2, 2]
    let output = graph.get_node_value(pool.output)?.unwrap();
    assert_eq!(output.shape(), &[1, 1, 2, 2]);

    // 验证输出值（2x2 窗口的平均值）
    // 窗口1: (1+2+5+6)/4 = 3.5
    // 窗口2: (3+4+7+8)/4 = 5.5
    // 窗口3: (9+10+13+14)/4 = 11.5
    // 窗口4: (11+12+15+16)/4 = 13.5
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 3.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 0, 1]], 5.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 11.5, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 13.5, epsilon = 1e-6);

    Ok(())
}

/// 测试 avg_pool2d() 默认 stride（等于 kernel_size）
#[test]
fn test_avg_pool2d_default_stride() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);

    let input = graph.new_input_node(&[1, 1, 4, 4], Some("input"))?;
    graph.set_node_value(input, Some(&Tensor::ones(&[1, 1, 4, 4])))?;

    // stride=None → 默认等于 kernel_size
    let pool = avg_pool2d(&mut graph, input, (2, 2), None, Some("pool1"))?;

    graph.forward_node(pool.output)?;

    // 验证输出形状: [1, 1, 2, 2]
    let output = graph.get_node_value(pool.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[1, 1, 2, 2]);

    Ok(())
}

// ==================== 节点名称测试 ====================

/// 测试 avg_pool2d() 带名称
#[test]
fn test_avg_pool2d_with_name() -> Result<(), GraphError> {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 16, 8, 8], Some("input"))?;

    let pool = avg_pool2d(&mut graph, input, (2, 2), Some((2, 2)), Some("encoder_gap"))?;

    // 验证节点名称
    assert_eq!(graph.get_node(pool.output)?.name(), "encoder_gap_out");

    Ok(())
}

/// 测试 avg_pool2d() 无名称（使用默认前缀）
#[test]
fn test_avg_pool2d_without_name() -> Result<(), GraphError> {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 16, 8, 8], Some("input"))?;

    let pool = avg_pool2d(&mut graph, input, (2, 2), Some((2, 2)), None)?;

    // 验证使用默认前缀 "avg_pool2d"
    assert_eq!(graph.get_node(pool.output)?.name(), "avg_pool2d_out");

    Ok(())
}

// ==================== 链式连接测试 ====================

/// 测试 avg_pool2d() 与 conv2d 链式连接
#[test]
fn test_avg_pool2d_with_conv2d() -> Result<(), GraphError> {
    use crate::nn::layer::conv2d;

    let mut graph = Graph::new_with_seed(42);

    // 典型 CNN: conv -> relu -> avg_pool
    let input = graph.new_input_node(&[2, 1, 8, 8], Some("input"))?;
    let conv = conv2d(
        &mut graph,
        input,
        1,
        4,
        (3, 3),
        (1, 1),
        (1, 1),
        Some("conv1"),
    )?;
    let relu = graph.new_leaky_relu_node(conv.output, 0.0, Some("relu1"))?;
    let pool = avg_pool2d(&mut graph, relu, (2, 2), Some((2, 2)), Some("pool1"))?;

    // 设置输入
    let x = Tensor::normal(0.0, 1.0, &[2, 1, 8, 8]);
    graph.set_node_value(input, Some(&x))?;

    // 前向传播
    graph.forward_node(pool.output)?;

    // 验证输出形状: [2, 4, 4, 4]
    let output = graph.get_node_value(pool.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[2, 4, 4, 4]);

    Ok(())
}

/// 测试全局平均池化（Global Average Pooling）
#[test]
fn test_avg_pool2d_global_average_pooling() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);

    // GAP: 将特征图池化到 1x1
    let input = graph.new_input_node(&[2, 64, 7, 7], Some("input"))?;
    // 使用 7x7 的池化窗口实现全局平均池化
    let gap = avg_pool2d(&mut graph, input, (7, 7), None, Some("gap"))?;

    // 设置输入
    let x = Tensor::ones(&[2, 64, 7, 7]);
    graph.set_node_value(input, Some(&x))?;

    // 前向传播
    graph.forward_node(gap.output)?;

    // 验证输出形状: [2, 64, 1, 1]
    let output = graph.get_node_value(gap.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[2, 64, 1, 1]);

    // 全 1 输入的全局平均池化结果应该是 1.0
    assert_abs_diff_eq!(output.unwrap()[[0, 0, 0, 0]], 1.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 avg_pool2d + flatten 链式连接
#[test]
fn test_avg_pool2d_with_flatten() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);

    // pool -> flatten（CNN 末端典型结构）
    let input = graph.new_input_node(&[2, 4, 4, 4], Some("input"))?;
    let pool = avg_pool2d(&mut graph, input, (2, 2), Some((2, 2)), Some("pool"))?;
    let flat = graph.new_flatten_node(pool.output, true, Some("flat"))?;

    // 设置输入
    let x = Tensor::ones(&[2, 4, 4, 4]);
    graph.set_node_value(input, Some(&x))?;

    // 前向传播
    graph.forward_node(flat)?;

    // 验证展平输出: [2, 4*2*2] = [2, 16]
    let output = graph.get_node_value(flat)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[2, 16]);

    Ok(())
}

// ==================== 反向传播测试 ====================

/// 测试 avg_pool2d() 与 Batch 反向传播
#[test]
fn test_avg_pool2d_batch_backward() -> Result<(), GraphError> {
    use crate::nn::layer::conv2d;

    let mut graph = Graph::new_with_seed(42);
    let batch_size = 2;

    // 构建网络: conv -> avg_pool -> flatten -> matmul -> loss
    let input = graph.new_input_node(&[batch_size, 1, 8, 8], Some("input"))?;
    let conv = conv2d(
        &mut graph,
        input,
        1,
        2,
        (3, 3),
        (1, 1),
        (1, 1),
        Some("conv"),
    )?;
    // conv 输出: [2, 2, 8, 8]
    let pool = avg_pool2d(&mut graph, conv.output, (2, 2), Some((2, 2)), Some("pool"))?;
    // pool 输出: [2, 2, 4, 4]
    let flat = graph.new_flatten_node(pool.output, true, Some("flat"))?;
    // flat 输出: [2, 32]

    // 分类器
    let fc_weight = graph.new_parameter_node(&[32, 3], Some("fc_w"))?;
    let logits = graph.new_mat_mul_node(flat, fc_weight, Some("logits"))?;

    // SoftmaxCrossEntropy Loss
    let labels = graph.new_input_node(&[batch_size, 3], Some("labels"))?;
    let loss = graph.new_softmax_cross_entropy_node(logits, labels, Some("loss"))?;

    // 设置数据
    let x = Tensor::normal(0.0, 1.0, &[batch_size, 1, 8, 8]);
    let y = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[batch_size, 3]);

    graph.set_node_value(input, Some(&x))?;
    graph.set_node_value(labels, Some(&y))?;

    // Batch 训练
    graph.forward_batch(loss)?;
    graph.backward_batch(loss)?;

    // 验证卷积核有梯度（梯度通过池化层正确传播）
    let k_grad = graph.get_node_grad_batch(conv.kernel)?;
    assert!(k_grad.is_some());
    assert_eq!(k_grad.unwrap().shape(), &[2, 1, 3, 3]);

    Ok(())
}

// ==================== 名称冲突测试 ====================

/// 测试多个 avg_pool2d() 使用不同名称
#[test]
fn test_avg_pool2d_multiple_layers_different_names() -> Result<(), GraphError> {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 4, 16, 16], Some("input"))?;

    let pool1 = avg_pool2d(&mut graph, input, (2, 2), Some((2, 2)), Some("pool1"))?;
    let pool2 = avg_pool2d(
        &mut graph,
        pool1.output,
        (2, 2),
        Some((2, 2)),
        Some("pool2"),
    )?;
    let pool3 = avg_pool2d(
        &mut graph,
        pool2.output,
        (2, 2),
        Some((2, 2)),
        Some("pool3"),
    )?;

    // 验证各层节点独立存在
    assert!(graph.get_node_value(pool1.output).is_ok());
    assert!(graph.get_node_value(pool2.output).is_ok());
    assert!(graph.get_node_value(pool3.output).is_ok());

    // 验证节点名称正确
    assert_eq!(graph.get_node(pool1.output)?.name(), "pool1_out");
    assert_eq!(graph.get_node(pool2.output)?.name(), "pool2_out");
    assert_eq!(graph.get_node(pool3.output)?.name(), "pool3_out");

    Ok(())
}

/// 测试重复名称应该报错
#[test]
fn test_avg_pool2d_duplicate_name_error() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 4, 8, 8], Some("input")).unwrap();

    // 第一个 pool 成功
    let pool1 = avg_pool2d(&mut graph, input, (2, 2), Some((2, 2)), Some("pool"));
    assert!(pool1.is_ok());

    // 第二个 pool 使用相同名称，应该失败
    let pool2 = avg_pool2d(
        &mut graph,
        pool1.unwrap().output,
        (2, 2),
        Some((2, 2)),
        Some("pool"),
    );
    assert!(pool2.is_err());

    // 验证错误类型
    if let Err(e) = pool2 {
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
fn test_avg_pool2d_multiple_unnamed_layers_conflict() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 4, 8, 8], Some("input")).unwrap();

    // 第一个无名称层成功
    let pool1 = avg_pool2d(&mut graph, input, (2, 2), Some((2, 2)), None);
    assert!(pool1.is_ok());

    // 第二个无名称层应该失败（名称冲突）
    let pool2 = avg_pool2d(
        &mut graph,
        pool1.unwrap().output,
        (2, 2),
        Some((2, 2)),
        None,
    );
    assert!(
        pool2.is_err(),
        "多个无名称 avg_pool2d 层应该因名称冲突而失败"
    );
}

// ==================== 边界维度测试 ====================

/// 测试单通道输入
#[test]
fn test_avg_pool2d_single_channel() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let input = graph.new_input_node(&[2, 1, 4, 4], Some("input"))?;

    let pool = avg_pool2d(&mut graph, input, (2, 2), Some((2, 2)), Some("pool"))?;

    let x = Tensor::ones(&[2, 1, 4, 4]);
    graph.set_node_value(input, Some(&x))?;

    graph.forward_node(pool.output)?;

    let output = graph.get_node_value(pool.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[2, 1, 2, 2]);

    Ok(())
}

/// 测试大通道数
#[test]
fn test_avg_pool2d_large_channels() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let input = graph.new_input_node(&[1, 128, 8, 8], Some("input"))?;

    let pool = avg_pool2d(&mut graph, input, (2, 2), Some((2, 2)), Some("pool"))?;

    // 验证输出形状预期
    let out_shape = graph.get_node(pool.output)?.value_expected_shape();
    assert_eq!(out_shape, &[1, 128, 4, 4]);

    Ok(())
}

/// 测试非方形池化窗口
#[test]
fn test_avg_pool2d_nonsquare_kernel() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let input = graph.new_input_node(&[2, 4, 8, 8], Some("input"))?;

    // 使用 2x4 的非方形池化窗口
    let pool = avg_pool2d(&mut graph, input, (2, 4), Some((2, 4)), Some("pool"))?;

    let x = Tensor::ones(&[2, 4, 8, 8]);
    graph.set_node_value(input, Some(&x))?;

    graph.forward_node(pool.output)?;

    // 输出尺寸:
    // H' = (8 - 2) / 2 + 1 = 4
    // W' = (8 - 4) / 4 + 1 = 2
    let output = graph.get_node_value(pool.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[2, 4, 4, 2]);

    Ok(())
}

/// 测试不同 stride
#[test]
fn test_avg_pool2d_different_stride() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let input = graph.new_input_node(&[2, 4, 8, 8], Some("input"))?;

    // 2x2 kernel, stride=1 → 重叠池化
    let pool = avg_pool2d(&mut graph, input, (2, 2), Some((1, 1)), Some("pool"))?;

    let x = Tensor::ones(&[2, 4, 8, 8]);
    graph.set_node_value(input, Some(&x))?;

    graph.forward_node(pool.output)?;

    // 输出尺寸: (8 - 2) / 1 + 1 = 7
    let output = graph.get_node_value(pool.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[2, 4, 7, 7]);

    Ok(())
}

/// 测试 ResNet 风格全局平均池化
#[test]
fn test_avg_pool2d_resnet_gap() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let batch_size = 16;

    // ResNet 末端: 特征图 -> GAP -> FC
    // 假设最后卷积层输出 [batch, 512, 7, 7]
    let features = graph.new_input_node(&[batch_size, 512, 7, 7], Some("features"))?;

    // 全局平均池化: 7x7 → 1x1
    let gap = avg_pool2d(&mut graph, features, (7, 7), None, Some("gap"))?;
    // gap 输出: [16, 512, 1, 1]

    // 展平后接全连接
    let _flat = graph.new_flatten_node(gap.output, true, Some("flat"))?;
    // flat 输出: [16, 512]

    // 验证形状
    let gap_shape = graph.get_node(gap.output)?.value_expected_shape();
    assert_eq!(gap_shape, &[batch_size, 512, 1, 1]);

    Ok(())
}
