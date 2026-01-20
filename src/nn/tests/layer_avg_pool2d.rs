/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : AvgPool2d layer 单元测试（PyTorch 风格 API）
 *
 * 参考值来源: tests/python/layer_reference/pool2d_layer_reference.py
 */

use crate::nn::layer::{AvgPool2d, Conv2d, Linear};
use crate::nn::{Graph, GraphError, VarActivationOps, VarLossOps, VarShapeOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== PyTorch 参考常量 ====================

// 测试: 简单前向传播 (batch=1, C=1, H=4, W=4, kernel=2, stride=2)
const TEST_PYTORCH_X: &[f32] = &[
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
];
const TEST_PYTORCH_OUTPUT: &[f32] = &[3.5, 5.5, 11.5, 13.5];

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
    0.71451831, -0.71313453, -0.09768094, -0.45061982,
    0.45072648, 0.60590321, 0.59642059, 0.63837188,
    -0.59038717, 0.25628099, 0.07167438, -0.80087370,
    -0.21714574, -0.23654196, 0.09488574, 0.20278738,
    0.61271536, 0.74116194, 0.07121982, -0.61857086,
    0.18323413, -0.18330121, -0.57192409, 0.34316877,
];

// ==================== PyTorch 数值对照测试 ====================

/// 测试 AvgPool2d 前向传播（与 PyTorch 对照）
#[test]
fn test_avg_pool2d_forward_pytorch_comparison() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    let x = graph.input(&Tensor::new(TEST_PYTORCH_X, &[1, 1, 4, 4]))?;
    let pool = AvgPool2d::new((2, 2), Some((2, 2)), "pool");
    let output = pool.forward(&x);

    output.forward()?;

    let out_val = output.value()?.unwrap();
    let output_data = out_val.data_as_slice();

    for (_, (&actual, &expected)) in output_data.iter().zip(TEST_PYTORCH_OUTPUT.iter()).enumerate() {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
    }

    println!("✅ AvgPool2d 前向传播与 PyTorch 一致");
    Ok(())
}

/// 测试 AvgPool2d 多通道多批次（与 PyTorch 对照）
#[test]
fn test_avg_pool2d_multi_channel_pytorch_comparison() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    let x = graph.input(&Tensor::new(TEST_MULTI_X, &[2, 3, 4, 4]))?;
    let pool = AvgPool2d::new((2, 2), Some((2, 2)), "pool");
    let output = pool.forward(&x);

    output.forward()?;

    let out_val = output.value()?.unwrap();
    let output_data = out_val.data_as_slice();

    for (_, (&actual, &expected)) in output_data.iter().zip(TEST_MULTI_OUTPUT.iter()).enumerate() {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
    }

    println!("✅ AvgPool2d 多通道多批次与 PyTorch 一致");
    Ok(())
}

/// 测试 AvgPool2d 反向传播（CNN 完整网络）
#[test]
fn test_avg_pool2d_backward_pytorch_comparison() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 2;

    // 构建网络: input -> conv -> relu -> avg_pool -> flatten -> fc -> softmax_ce
    let x = graph.input(&Tensor::new(&[0.0; 32], &[batch_size, 1, 4, 4]))?;
    let conv = Conv2d::new(&graph, 1, 2, (2, 2), (1, 1), (0, 0), true, "10")?;
    // conv 输出: [2, 2, 3, 3]
    let h = conv.forward(&x).relu();
    let pool = AvgPool2d::new((2, 2), Some((2, 2)), "pool");
    // pool 输出: [2, 2, 1, 1]
    let pooled = pool.forward(&h);
    let flat = pooled.flatten()?;
    // flat 输出: [2, 2]

    let fc = Linear::new(&graph, 2, 3, true, "fc")?;
    let fc_out = fc.forward(&flat);

    // SoftmaxCrossEntropy Loss
    let labels = graph.input(&Tensor::new(&[0.0; 6], &[batch_size, 3]))?;
    let loss = fc_out.cross_entropy(&labels)?;

    // 设置固定的输入和权重（来自 PyTorch 参考脚本）
    #[rustfmt::skip]
    let x_data: &[f32] = &[
        1.92691529, 1.48728406, 0.90071720, -2.10552096, 0.67841840, -1.23454487, -0.04306748, -1.60466695,
        -0.75213528, 1.64872301, -0.39247864, -1.40360713, -0.72788131, -0.55943018, -0.76883888, 0.76244539,
        1.64231694, -0.15959747, -0.49739754, 0.43958926, -0.75813115, 1.07831764, 0.80080056, 1.68062055,
        1.27912438, 1.29642284, 0.61046648, 1.33473778, -0.23162432, 0.04175949, -0.25157529, 0.85985851,
    ];
    let conv_weight: &[f32] = &[
        -0.11146712, 0.12036294, -0.36963451, -0.24041797,
        -1.19692433, 0.20926936, -0.97235501, -0.75504547,
    ];
    let conv_bias_data: &[f32] = &[0.32390276, -0.10852263];
    let fc_weight: &[f32] = &[
        -2.43058109, 1.60250008, -0.16449131,
        0.75442398, -1.14307523, 0.69427645,
    ];
    let fc_bias_data: &[f32] = &[1.29803729, -0.74028862, 1.03223586];
    let target_data: &[f32] = &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    x.set_value(&Tensor::new(x_data, &[batch_size, 1, 4, 4]))?;
    conv.kernel().set_value(&Tensor::new(conv_weight, &[2, 1, 2, 2]))?;
    conv.bias().unwrap().set_value(&Tensor::new(conv_bias_data, &[1, 2, 1, 1]))?;
    fc.weights().set_value(&Tensor::new(fc_weight, &[2, 3]))?;
    fc.bias().unwrap().set_value(&Tensor::new(fc_bias_data, &[1, 3]))?;
    labels.set_value(&Tensor::new(target_data, &[batch_size, 3]))?;

    // 先前向传播验证 loss
    loss.forward()?;
    let loss_val = loss.value()?.unwrap();
    let expected_loss: f32 = 1.79948235;
    assert_abs_diff_eq!(loss_val[[0, 0]], expected_loss, epsilon = 1e-4);

    // 反向传播
    loss.backward()?;

    // 验证卷积核梯度
    let conv_grad = conv.kernel().grad()?.unwrap();
    let expected_conv_grad: &[f32] = &[
        0.14983487, 0.55885887, 0.14587063, -0.64360768,
        0.04400400, 0.00153509, -0.05876692, 0.01398947,
    ];
    let conv_grad_data = conv_grad.data_as_slice();
    for (&actual, &expected) in conv_grad_data.iter().zip(expected_conv_grad.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
    }

    println!("✅ AvgPool2d 反向传播梯度与 PyTorch 一致");
    Ok(())
}

// ==================== 基础功能测试 ====================

/// 测试 AvgPool2d 创建
#[test]
fn test_avg_pool2d_creation() -> Result<(), GraphError> {
    let pool = AvgPool2d::new((2, 2), Some((2, 2)), "pool1");

    assert_eq!(pool.kernel_size(), (2, 2));
    assert_eq!(pool.stride(), Some((2, 2)));

    Ok(())
}

/// 测试 AvgPool2d 前向传播
#[test]
fn test_avg_pool2d_forward() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // 输入: [batch=1, C=1, H=4, W=4]
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
    let x = graph.input(&input_data)?;

    // 创建 AvgPool2d 层: 2x2 核, stride=2
    let pool = AvgPool2d::new((2, 2), Some((2, 2)), "pool1");
    let output = pool.forward(&x);

    output.forward()?;

    // 验证输出形状: [1, 1, 2, 2]
    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[1, 1, 2, 2]);

    // 验证输出值（2x2 窗口的平均值）
    assert_abs_diff_eq!(out_val[[0, 0, 0, 0]], 3.5, epsilon = 1e-6);
    assert_abs_diff_eq!(out_val[[0, 0, 0, 1]], 5.5, epsilon = 1e-6);
    assert_abs_diff_eq!(out_val[[0, 0, 1, 0]], 11.5, epsilon = 1e-6);
    assert_abs_diff_eq!(out_val[[0, 0, 1, 1]], 13.5, epsilon = 1e-6);

    Ok(())
}

/// 测试 AvgPool2d 默认 stride（等于 kernel_size）
#[test]
fn test_avg_pool2d_default_stride() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    let x = graph.input(&Tensor::ones(&[1, 1, 4, 4]))?;
    // stride=None → 默认等于 kernel_size
    let pool = AvgPool2d::new((2, 2), None, "pool1");
    let output = pool.forward(&x);

    output.forward()?;

    // 验证输出形状: [1, 1, 2, 2]
    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[1, 1, 2, 2]);

    Ok(())
}

// ==================== 链式连接测试 ====================

/// 测试 AvgPool2d 与 Conv2d 链式连接
#[test]
fn test_avg_pool2d_with_conv2d() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // 典型 CNN: conv -> relu -> avg_pool
    let x = graph.input(&Tensor::normal(0.0, 1.0, &[2, 1, 8, 8]))?;
    let conv = Conv2d::new(&graph, 1, 4, (3, 3), (1, 1), (1, 1), true, "10")?;
    let pool = AvgPool2d::new((2, 2), Some((2, 2)), "pool1");

    let h = conv.forward(&x).relu();
    let output = pool.forward(&h);

    output.forward()?;

    // 验证输出形状: [2, 4, 4, 4]
    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[2, 4, 4, 4]);

    Ok(())
}

/// 测试全局平均池化（Global Average Pooling）
#[test]
fn test_avg_pool2d_global_average_pooling() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // GAP: 将特征图池化到 1x1
    let x = graph.input(&Tensor::ones(&[2, 64, 7, 7]))?;
    // 使用 7x7 的池化窗口实现全局平均池化
    let gap = AvgPool2d::new((7, 7), None, "gap");
    let output = gap.forward(&x);

    output.forward()?;

    // 验证输出形状: [2, 64, 1, 1]
    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[2, 64, 1, 1]);

    // 全 1 输入的全局平均池化结果应该是 1.0
    assert_abs_diff_eq!(out_val[[0, 0, 0, 0]], 1.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 AvgPool2d + flatten 链式连接
#[test]
fn test_avg_pool2d_with_flatten() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // pool -> flatten（CNN 末端典型结构）
    let x = graph.input(&Tensor::ones(&[2, 4, 4, 4]))?;
    let pool = AvgPool2d::new((2, 2), Some((2, 2)), "pool");
    let output = pool.forward(&x).flatten()?;

    output.forward()?;

    // 验证展平输出: [2, 4*2*2] = [2, 16]
    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[2, 16]);

    Ok(())
}

// ==================== 反向传播测试 ====================

/// 测试 AvgPool2d 与 Batch 反向传播
#[test]
fn test_avg_pool2d_batch_backward() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 2;

    // 构建网络: conv -> avg_pool -> flatten -> fc -> loss
    let x = graph.input(&Tensor::normal(0.0, 1.0, &[batch_size, 1, 8, 8]))?;
    let conv = Conv2d::new(&graph, 1, 2, (3, 3), (1, 1), (1, 1), true, "10")?;
    // conv 输出: [2, 2, 8, 8]
    let pool = AvgPool2d::new((2, 2), Some((2, 2)), "pool");
    // pool 输出: [2, 2, 4, 4]
    let flat = pool.forward(&conv.forward(&x)).flatten()?;
    // flat 输出: [2, 32]

    let fc = Linear::new(&graph, 32, 3, true, "fc")?;
    let logits = fc.forward(&flat);

    // SoftmaxCrossEntropy Loss
    let labels = graph.input(&Tensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[batch_size, 3]))?;
    let loss = logits.cross_entropy(&labels)?;

    loss.backward()?;

    // 验证卷积核有梯度（梯度通过池化层正确传播）
    let k_grad = conv.kernel().grad()?.unwrap();
    assert_eq!(k_grad.shape(), &[2, 1, 3, 3]);

    Ok(())
}

// ==================== 边界维度测试 ====================

/// 测试单通道输入
#[test]
fn test_avg_pool2d_single_channel() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::ones(&[2, 1, 4, 4]))?;

    let pool = AvgPool2d::new((2, 2), Some((2, 2)), "pool");
    let output = pool.forward(&x);

    output.forward()?;

    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[2, 1, 2, 2]);

    Ok(())
}

/// 测试大通道数
#[test]
fn test_avg_pool2d_large_channels() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::ones(&[1, 128, 8, 8]))?;

    let pool = AvgPool2d::new((2, 2), Some((2, 2)), "pool");
    let output = pool.forward(&x);

    output.forward()?;

    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[1, 128, 4, 4]);

    Ok(())
}

/// 测试非方形池化窗口
#[test]
fn test_avg_pool2d_nonsquare_kernel() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::ones(&[2, 4, 8, 8]))?;

    // 使用 2x4 的非方形池化窗口
    let pool = AvgPool2d::new((2, 4), Some((2, 4)), "pool");
    let output = pool.forward(&x);

    output.forward()?;

    // 输出尺寸:
    // H' = (8 - 2) / 2 + 1 = 4
    // W' = (8 - 4) / 4 + 1 = 2
    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[2, 4, 4, 2]);

    Ok(())
}

/// 测试不同 stride
#[test]
fn test_avg_pool2d_different_stride() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::ones(&[2, 4, 8, 8]))?;

    // 2x2 kernel, stride=1 → 重叠池化
    let pool = AvgPool2d::new((2, 2), Some((1, 1)), "pool");
    let output = pool.forward(&x);

    output.forward()?;

    // 输出尺寸: (8 - 2) / 1 + 1 = 7
    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[2, 4, 7, 7]);

    Ok(())
}

/// 测试 ResNet 风格全局平均池化
#[test]
fn test_avg_pool2d_resnet_gap() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 16;

    // ResNet 末端: 特征图 -> GAP -> FC
    let features = graph.input(&Tensor::ones(&[batch_size, 512, 7, 7]))?;

    // 全局平均池化: 7x7 → 1x1
    let gap = AvgPool2d::new((7, 7), None, "gap");
    let pooled = gap.forward(&features);
    // gap 输出: [16, 512, 1, 1]

    // 展平后接全连接
    let flat = pooled.flatten()?;
    // flat 输出: [16, 512]

    flat.forward()?;

    // 验证形状
    let gap_val = pooled.value()?.unwrap();
    assert_eq!(gap_val.shape(), &[batch_size, 512, 1, 1]);

    Ok(())
}
