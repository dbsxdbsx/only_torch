/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : MaxPool2d layer 单元测试（Batch-First 设计）
 *
 * 参考值来源: tests/python/layer_reference/pool2d_layer_reference.py
 */

use crate::nn::layer::max_pool2d;
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== PyTorch 参考常量 ====================

// 测试: 简单前向传播 (batch=1, C=1, H=4, W=4, kernel=2, stride=2)
const TEST_PYTORCH_X: &[f32] = &[
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
];
const TEST_PYTORCH_OUTPUT: &[f32] = &[6.0, 8.0, 14.0, 16.0];

// 测试: 多通道多批次 (batch=2, C=3, H=4, W=4)
#[rustfmt::skip]
const TEST_MULTI_X: &[f32] = &[
    1.92691529, 1.48728406, 0.90071720, -2.10552096, 0.67841840, -1.23454487, -0.04306748, -1.60466695,
    -0.75213528, 1.64872301, -0.39247864, -1.40360713, -0.72788131, -0.55943018, -0.76883888, 0.76244539,
    1.64231694, -0.15959747, -0.49739754, 0.43958926, -0.75813115, 1.07831764, 0.80080056, 1.68062055,
    1.27912438, 1.29642284, 0.61046648, 1.33473778, -0.23162432, 0.04175949, -0.25157529, 0.85985851,
    -1.38467371, -0.87123615, -0.22336592, 1.71736145, 0.31888032, -0.42451897, 0.30572093, -0.77459252,
    -1.55757248, 0.99563611, -0.87978584, -0.60114205, -1.27415121, 2.12278509, -1.23465312, -0.48791388,
    -0.91382301, -0.65813726, 0.07802387, 0.52580875, -0.48799172, 1.19136906, -0.81400764, -0.73599279,
    -1.40324783, 0.03600367, -0.06347727, 0.67561489, -0.09780689, 1.84459400, -1.18453741, 1.38354933,
    1.44513381, 0.85641253, 2.21807575, 0.52316552, 0.34664667, -0.19733144, -1.05458891, 1.27799559,
    -0.17219013, 0.52378845, 0.05662182, 0.42629614, 0.57500505, -0.64172411, -2.20639849, -0.75080305,
    0.01086814, -0.33874235, -1.34067953, -0.58537054, 0.53618813, 0.52462262, 1.14120162, 0.05164360,
    0.74395198, -0.48158440, -1.04946601, 0.60389882, -1.72229505, -0.82776886, 1.33470297, 0.48353928,
];
#[rustfmt::skip]
const TEST_MULTI_OUTPUT: &[f32] = &[
    1.92691529, 0.90071720, 1.64872301, 0.76244539,
    1.64231694, 1.68062055, 1.29642284, 1.33473778,
    0.31888032, 1.71736145, 2.12278509, -0.48791388,
    1.19136906, 0.52580875, 1.84459400, 1.38354933,
    1.44513381, 2.21807575, 0.57500505, 0.42629614,
    0.53618813, 1.14120162, 0.74395198, 1.33470297,
];

// ==================== PyTorch 数值对照测试 ====================

/// 测试 MaxPool2d 前向传播（与 PyTorch 对照）
#[test]
fn test_max_pool2d_forward_pytorch_comparison() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);

    let input = graph.new_input_node(&[1, 1, 4, 4], Some("input"))?;
    let pool = max_pool2d(&mut graph, input, (2, 2), Some((2, 2)), Some("pool"))?;

    graph.set_node_value(input, Some(&Tensor::new(TEST_PYTORCH_X, &[1, 1, 4, 4])))?;

    graph.forward(pool.output)?;

    let output = graph.get_node_value(pool.output)?.unwrap();
    let output_data = output.data_as_slice();

    for (_, (&actual, &expected)) in output_data
        .iter()
        .zip(TEST_PYTORCH_OUTPUT.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
    }

    println!("✅ MaxPool2d 前向传播与 PyTorch 一致");
    Ok(())
}

/// 测试 MaxPool2d 多通道多批次（与 PyTorch 对照）
#[test]
fn test_max_pool2d_multi_channel_pytorch_comparison() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);

    let input = graph.new_input_node(&[2, 3, 4, 4], Some("input"))?;
    let pool = max_pool2d(&mut graph, input, (2, 2), Some((2, 2)), Some("pool"))?;

    graph.set_node_value(input, Some(&Tensor::new(TEST_MULTI_X, &[2, 3, 4, 4])))?;

    graph.forward(pool.output)?;

    let output = graph.get_node_value(pool.output)?.unwrap();
    let output_data = output.data_as_slice();

    for (_, (&actual, &expected)) in output_data.iter().zip(TEST_MULTI_OUTPUT.iter()).enumerate() {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
    }

    println!("✅ MaxPool2d 多通道多批次与 PyTorch 一致");
    Ok(())
}

/// 测试 MaxPool2d 反向传播（CNN 完整网络）
#[test]
fn test_max_pool2d_backward_pytorch_comparison() -> Result<(), GraphError> {
    use crate::nn::layer::{conv2d, linear};

    let mut graph = Graph::new_with_seed(42);
    let batch_size = 2;

    // 构建网络: input -> conv -> relu -> max_pool -> flatten -> fc -> softmax_ce
    let input = graph.new_input_node(&[batch_size, 1, 4, 4], Some("input"))?;
    let conv = conv2d(
        &mut graph,
        input,
        1,
        2,
        (2, 2),
        (1, 1),
        (0, 0),
        Some("conv"),
    )?;
    // conv 输出: [2, 2, 3, 3]
    let relu = graph.new_leaky_relu_node(conv.output, 0.0, Some("relu"))?;
    let pool = max_pool2d(&mut graph, relu, (2, 2), Some((2, 2)), Some("pool"))?;
    // pool 输出: [2, 2, 1, 1]
    let flat = graph.new_flatten_node(pool.output, true, Some("flat"))?;
    // flat 输出: [2, 2]
    let fc = linear(&mut graph, flat, 2, 3, batch_size, Some("fc"))?;

    // SoftmaxCrossEntropy Loss
    let labels = graph.new_input_node(&[batch_size, 3], Some("labels"))?;
    let loss = graph.new_softmax_cross_entropy_node(fc.output, labels, Some("loss"))?;

    // 设置固定的输入和权重（来自 PyTorch 参考脚本）
    #[rustfmt::skip]
    let x_data: &[f32] = &[
        1.92691529, 1.48728406, 0.90071720, -2.10552096, 0.67841840, -1.23454487, -0.04306748, -1.60466695,
        -0.75213528, 1.64872301, -0.39247864, -1.40360713, -0.72788131, -0.55943018, -0.76883888, 0.76244539,
        1.64231694, -0.15959747, -0.49739754, 0.43958926, -0.75813115, 1.07831764, 0.80080056, 1.68062055,
        1.27912438, 1.29642284, 0.61046648, 1.33473778, -0.23162432, 0.04175949, -0.25157529, 0.85985851,
    ];
    let conv_weight: &[f32] = &[
        -0.11146712,
        0.12036294,
        -0.36963451,
        -0.24041797,
        -1.19692433,
        0.20926936,
        -0.97235501,
        -0.75504547,
    ];
    let conv_bias: &[f32] = &[0.32390276, -0.10852263];
    let fc_weight: &[f32] = &[
        -2.43058109,
        1.60250008,
        -0.16449131,
        0.75442398,
        -1.14307523,
        0.69427645,
    ];
    let fc_bias: &[f32] = &[1.29803729, -0.74028862, 1.03223586];
    let target_data: &[f32] = &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    graph.set_node_value(input, Some(&Tensor::new(x_data, &[batch_size, 1, 4, 4])))?;
    graph.set_node_value(conv.kernel, Some(&Tensor::new(conv_weight, &[2, 1, 2, 2])))?;
    graph.set_node_value(conv.bias, Some(&Tensor::new(conv_bias, &[1, 2])))?;
    graph.set_node_value(fc.weights, Some(&Tensor::new(fc_weight, &[2, 3])))?;
    graph.set_node_value(fc.bias, Some(&Tensor::new(fc_bias, &[1, 3])))?;
    graph.set_node_value(labels, Some(&Tensor::new(target_data, &[batch_size, 3])))?;

    // 前向传播
    graph.forward(loss)?;

    // 验证 loss
    let loss_val = graph.get_node_value(loss)?.unwrap();
    let expected_loss: f32 = 2.13923550;
    assert_abs_diff_eq!(loss_val[[0, 0]], expected_loss, epsilon = 1e-4);

    // 反向传播
    graph.backward(loss)?;

    // 验证卷积核梯度
    let conv_grad = graph.get_node_grad_ref(conv.kernel)?.unwrap();
    let expected_conv_grad: &[f32] = &[
        -0.25443733,
        1.33351851,
        -0.56869137,
        -1.44372773,
        0.38442636,
        0.01341083,
        -0.51339775,
        0.12221438,
    ];
    let conv_grad_data = conv_grad.data_as_slice();
    for (&actual, &expected) in conv_grad_data.iter().zip(expected_conv_grad.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
    }

    println!("✅ MaxPool2d 反向传播梯度与 PyTorch 一致");
    Ok(())
}

// ==================== 基础功能测试 ====================

/// 测试 max_pool2d() 创建
#[test]
fn test_max_pool2d_creation() -> Result<(), GraphError> {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 32, 14, 14], Some("input"))?;

    let pool = max_pool2d(&mut graph, input, (2, 2), Some((2, 2)), Some("pool1"))?;

    // 验证返回的节点 ID 有效
    assert!(graph.get_node_value(pool.output).is_ok());

    Ok(())
}

/// 测试 max_pool2d() 输出形状
#[test]
fn test_max_pool2d_shapes() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let input = graph.new_input_node(&[4, 16, 28, 28], Some("input"))?;

    // 2x2 池化, stride=2 → 尺寸减半
    let pool = max_pool2d(&mut graph, input, (2, 2), Some((2, 2)), Some("pool1"))?;

    // 检查输出形状预期: [4, 16, 14, 14]
    let out_shape = graph.get_node(pool.output)?.value_expected_shape();
    assert_eq!(out_shape, &[4, 16, 14, 14]);

    Ok(())
}

/// 测试 max_pool2d() 前向传播
#[test]
fn test_max_pool2d_forward() -> Result<(), GraphError> {
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

    // 创建 max_pool2d 层: 2x2 核, stride=2
    let pool = max_pool2d(&mut graph, input, (2, 2), Some((2, 2)), Some("pool1"))?;

    // 前向传播
    graph.forward(pool.output)?;

    // 验证输出形状: [1, 1, 2, 2]
    let output = graph.get_node_value(pool.output)?.unwrap();
    assert_eq!(output.shape(), &[1, 1, 2, 2]);

    // 验证输出值（2x2 窗口的最大值）
    // 窗口1: [1,2,5,6] → max=6
    // 窗口2: [3,4,7,8] → max=8
    // 窗口3: [9,10,13,14] → max=14
    // 窗口4: [11,12,15,16] → max=16
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 0, 1]], 8.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 14.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 16.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 max_pool2d() 默认 stride（等于 kernel_size）
#[test]
fn test_max_pool2d_default_stride() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);

    let input = graph.new_input_node(&[1, 1, 4, 4], Some("input"))?;
    graph.set_node_value(input, Some(&Tensor::ones(&[1, 1, 4, 4])))?;

    // stride=None → 默认等于 kernel_size
    let pool = max_pool2d(&mut graph, input, (2, 2), None, Some("pool1"))?;

    graph.forward(pool.output)?;

    // 验证输出形状: [1, 1, 2, 2]
    let output = graph.get_node_value(pool.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[1, 1, 2, 2]);

    Ok(())
}

// ==================== 节点名称测试 ====================

/// 测试 max_pool2d() 带名称
#[test]
fn test_max_pool2d_with_name() -> Result<(), GraphError> {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 16, 8, 8], Some("input"))?;

    let pool = max_pool2d(
        &mut graph,
        input,
        (2, 2),
        Some((2, 2)),
        Some("encoder_pool1"),
    )?;

    // 验证节点名称
    assert_eq!(graph.get_node(pool.output)?.name(), "encoder_pool1_out");

    Ok(())
}

/// 测试 max_pool2d() 无名称（使用默认前缀）
#[test]
fn test_max_pool2d_without_name() -> Result<(), GraphError> {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 16, 8, 8], Some("input"))?;

    let pool = max_pool2d(&mut graph, input, (2, 2), Some((2, 2)), None)?;

    // 验证使用默认前缀 "max_pool2d"
    assert_eq!(graph.get_node(pool.output)?.name(), "max_pool2d_out");

    Ok(())
}

// ==================== 链式连接测试 ====================

/// 测试 max_pool2d() 与 conv2d 链式连接（典型 CNN 结构）
#[test]
fn test_max_pool2d_with_conv2d() -> Result<(), GraphError> {
    use crate::nn::layer::conv2d;

    let mut graph = Graph::new_with_seed(42);

    // 典型 CNN: conv -> relu -> pool
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
    let pool = max_pool2d(&mut graph, relu, (2, 2), Some((2, 2)), Some("pool1"))?;

    // 设置输入
    let x = Tensor::normal(0.0, 1.0, &[2, 1, 8, 8]);
    graph.set_node_value(input, Some(&x))?;

    // 前向传播
    graph.forward(pool.output)?;

    // 验证输出形状: [2, 4, 4, 4]
    // conv 输出 [2, 4, 8, 8] (same padding)
    // pool 输出 [2, 4, 4, 4] (2x2 池化, stride=2)
    let output = graph.get_node_value(pool.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[2, 4, 4, 4]);

    Ok(())
}

/// 测试多个 max_pool2d() 链式连接
#[test]
fn test_max_pool2d_chain() -> Result<(), GraphError> {
    use crate::nn::layer::conv2d;

    let mut graph = Graph::new_with_seed(42);

    // 多层 CNN: conv1 -> pool1 -> conv2 -> pool2
    let input = graph.new_input_node(&[2, 1, 16, 16], Some("input"))?;

    let conv1 = conv2d(
        &mut graph,
        input,
        1,
        4,
        (3, 3),
        (1, 1),
        (1, 1),
        Some("conv1"),
    )?;
    let pool1 = max_pool2d(
        &mut graph,
        conv1.output,
        (2, 2),
        Some((2, 2)),
        Some("pool1"),
    )?;
    // pool1 输出: [2, 4, 8, 8]

    let conv2 = conv2d(
        &mut graph,
        pool1.output,
        4,
        8,
        (3, 3),
        (1, 1),
        (1, 1),
        Some("conv2"),
    )?;
    let pool2 = max_pool2d(
        &mut graph,
        conv2.output,
        (2, 2),
        Some((2, 2)),
        Some("pool2"),
    )?;
    // pool2 输出: [2, 8, 4, 4]

    // 设置输入
    let x = Tensor::normal(0.0, 1.0, &[2, 1, 16, 16]);
    graph.set_node_value(input, Some(&x))?;

    // 前向传播
    graph.forward(pool2.output)?;

    // 验证输出
    let output = graph.get_node_value(pool2.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[2, 8, 4, 4]);

    Ok(())
}

/// 测试 max_pool2d + flatten 链式连接
#[test]
fn test_max_pool2d_with_flatten() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);

    // pool -> flatten（CNN 末端典型结构）
    let input = graph.new_input_node(&[2, 4, 4, 4], Some("input"))?;
    let pool = max_pool2d(&mut graph, input, (2, 2), Some((2, 2)), Some("pool"))?;
    let flat = graph.new_flatten_node(pool.output, true, Some("flat"))?;

    // 设置输入
    let x = Tensor::ones(&[2, 4, 4, 4]);
    graph.set_node_value(input, Some(&x))?;

    // 前向传播
    graph.forward(flat)?;

    // 验证展平输出: [2, 4*2*2] = [2, 16]
    let output = graph.get_node_value(flat)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[2, 16]);

    Ok(())
}

// ==================== 反向传播测试 ====================

/// 测试 max_pool2d() 与 Batch 反向传播
#[test]
fn test_max_pool2d_batch_backward() -> Result<(), GraphError> {
    use crate::nn::layer::conv2d;

    let mut graph = Graph::new_with_seed(42);
    let batch_size = 2;

    // 构建网络: conv -> pool -> flatten -> matmul -> loss
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
    let pool = max_pool2d(&mut graph, conv.output, (2, 2), Some((2, 2)), Some("pool"))?;
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
    graph.forward(loss)?;
    graph.backward(loss)?;

    // 验证卷积核有梯度（梯度通过池化层正确传播）
    let k_grad = graph.get_node_grad_ref(conv.kernel)?;
    assert!(k_grad.is_some());
    assert_eq!(k_grad.unwrap().shape(), &[2, 1, 3, 3]);

    Ok(())
}

// ==================== 名称冲突测试 ====================

/// 测试多个 max_pool2d() 使用不同名称
#[test]
fn test_max_pool2d_multiple_layers_different_names() -> Result<(), GraphError> {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 4, 16, 16], Some("input"))?;

    let pool1 = max_pool2d(&mut graph, input, (2, 2), Some((2, 2)), Some("pool1"))?;
    let pool2 = max_pool2d(
        &mut graph,
        pool1.output,
        (2, 2),
        Some((2, 2)),
        Some("pool2"),
    )?;
    let pool3 = max_pool2d(
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
fn test_max_pool2d_duplicate_name_error() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 4, 8, 8], Some("input")).unwrap();

    // 第一个 pool 成功
    let pool1 = max_pool2d(&mut graph, input, (2, 2), Some((2, 2)), Some("pool"));
    assert!(pool1.is_ok());

    // 第二个 pool 使用相同名称，应该失败
    let pool2 = max_pool2d(
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
fn test_max_pool2d_multiple_unnamed_layers_conflict() {
    let mut graph = Graph::new();
    let input = graph.new_input_node(&[2, 4, 8, 8], Some("input")).unwrap();

    // 第一个无名称层成功
    let pool1 = max_pool2d(&mut graph, input, (2, 2), Some((2, 2)), None);
    assert!(pool1.is_ok());

    // 第二个无名称层应该失败（名称冲突）
    let pool2 = max_pool2d(
        &mut graph,
        pool1.unwrap().output,
        (2, 2),
        Some((2, 2)),
        None,
    );
    assert!(
        pool2.is_err(),
        "多个无名称 max_pool2d 层应该因名称冲突而失败"
    );
}

// ==================== 边界维度测试 ====================

/// 测试单通道输入
#[test]
fn test_max_pool2d_single_channel() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let input = graph.new_input_node(&[2, 1, 4, 4], Some("input"))?;

    let pool = max_pool2d(&mut graph, input, (2, 2), Some((2, 2)), Some("pool"))?;

    let x = Tensor::ones(&[2, 1, 4, 4]);
    graph.set_node_value(input, Some(&x))?;

    graph.forward(pool.output)?;

    let output = graph.get_node_value(pool.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[2, 1, 2, 2]);

    Ok(())
}

/// 测试大通道数
#[test]
fn test_max_pool2d_large_channels() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let input = graph.new_input_node(&[1, 128, 8, 8], Some("input"))?;

    let pool = max_pool2d(&mut graph, input, (2, 2), Some((2, 2)), Some("pool"))?;

    // 验证输出形状预期
    let out_shape = graph.get_node(pool.output)?.value_expected_shape();
    assert_eq!(out_shape, &[1, 128, 4, 4]);

    Ok(())
}

/// 测试非方形池化窗口
#[test]
fn test_max_pool2d_nonsquare_kernel() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let input = graph.new_input_node(&[2, 4, 8, 8], Some("input"))?;

    // 使用 2x4 的非方形池化窗口
    let pool = max_pool2d(&mut graph, input, (2, 4), Some((2, 4)), Some("pool"))?;

    let x = Tensor::ones(&[2, 4, 8, 8]);
    graph.set_node_value(input, Some(&x))?;

    graph.forward(pool.output)?;

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
fn test_max_pool2d_different_stride() -> Result<(), GraphError> {
    let mut graph = Graph::new_with_seed(42);
    let input = graph.new_input_node(&[2, 4, 8, 8], Some("input"))?;

    // 2x2 kernel, stride=1 → 重叠池化
    let pool = max_pool2d(&mut graph, input, (2, 2), Some((1, 1)), Some("pool"))?;

    let x = Tensor::ones(&[2, 4, 8, 8]);
    graph.set_node_value(input, Some(&x))?;

    graph.forward(pool.output)?;

    // 输出尺寸: (8 - 2) / 1 + 1 = 7
    let output = graph.get_node_value(pool.output)?;
    assert!(output.is_some());
    assert_eq!(output.unwrap().shape(), &[2, 4, 7, 7]);

    Ok(())
}

/// 测试典型 MNIST/CIFAR CNN 配置
#[test]
fn test_max_pool2d_typical_cnn() -> Result<(), GraphError> {
    use crate::nn::layer::conv2d;

    let mut graph = Graph::new_with_seed(42);
    let batch_size = 32;

    // 典型 LeNet 风格网络
    // 输入: [32, 1, 28, 28] (MNIST)
    let input = graph.new_input_node(&[batch_size, 1, 28, 28], Some("input"))?;

    // conv1: 1->6, 5x5
    let conv1 = conv2d(
        &mut graph,
        input,
        1,
        6,
        (5, 5),
        (1, 1),
        (0, 0),
        Some("conv1"),
    )?;
    // conv1 输出: [32, 6, 24, 24]
    let pool1 = max_pool2d(
        &mut graph,
        conv1.output,
        (2, 2),
        Some((2, 2)),
        Some("pool1"),
    )?;
    // pool1 输出: [32, 6, 12, 12]

    // conv2: 6->16, 5x5
    let conv2 = conv2d(
        &mut graph,
        pool1.output,
        6,
        16,
        (5, 5),
        (1, 1),
        (0, 0),
        Some("conv2"),
    )?;
    // conv2 输出: [32, 16, 8, 8]
    let pool2 = max_pool2d(
        &mut graph,
        conv2.output,
        (2, 2),
        Some((2, 2)),
        Some("pool2"),
    )?;
    // pool2 输出: [32, 16, 4, 4]

    // 验证最终形状
    let out_shape = graph.get_node(pool2.output)?.value_expected_shape();
    assert_eq!(out_shape, &[batch_size, 16, 4, 4]);

    Ok(())
}
