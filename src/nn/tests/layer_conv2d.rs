/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : Conv2d layer 单元测试（PyTorch 风格 API，含数值对照）
 *
 * 参考值来源: tests/python/layer_reference/conv2d_layer_reference.py
 */

use crate::nn::layer::{Conv2d, Linear};
use crate::nn::{Graph, GraphError, Module, VarActivationOps, VarLossOps, VarShapeOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== PyTorch 参考常量 ====================

// 测试 1: 简单前向传播 (batch=1, C_in=1, H=4, W=4, C_out=2, kernel=2x2)
#[rustfmt::skip]
const PYTORCH_FWD_X: &[f32] = &[
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0,
];
#[rustfmt::skip]
const PYTORCH_FWD_KERNEL: &[f32] = &[
    1.0, 0.0, 0.0, 1.0,  // filter 0: 对角线
    0.0, 1.0, 1.0, 0.0,  // filter 1: 反对角线
];
const PYTORCH_FWD_BIAS: &[f32] = &[0.5, -0.5];
#[rustfmt::skip]
const PYTORCH_FWD_OUTPUT: &[f32] = &[
    7.5, 9.5, 11.5,
    15.5, 17.5, 19.5,
    23.5, 25.5, 27.5,
    6.5, 8.5, 10.5,
    14.5, 16.5, 18.5,
    22.5, 24.5, 26.5,
];

// 测试 4: 反向传播梯度 (batch=1, C_in=1, H=3, W=3, C_out=1, kernel=2x2) + MSE
#[rustfmt::skip]
const PYTORCH_BWD_X: &[f32] = &[
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
];
const PYTORCH_BWD_KERNEL: &[f32] = &[0.1, 0.2, 0.3, 0.4];
const PYTORCH_BWD_BIAS: &[f32] = &[0.5];
#[rustfmt::skip]
const PYTORCH_BWD_TARGET: &[f32] = &[
    5.0, 6.0,
    8.0, 9.0,
];
#[rustfmt::skip]
const PYTORCH_BWD_OUTPUT: &[f32] = &[
    4.2, 5.2,
    7.2, 8.2,
];
const PYTORCH_BWD_LOSS: f32 = 0.64;
const PYTORCH_BWD_GRAD_KERNEL: &[f32] = &[-4.8, -6.4, -9.6, -11.2];
const PYTORCH_BWD_GRAD_BIAS: &[f32] = &[-1.6];

// 测试 5: Conv -> ReLU -> Flatten -> Linear -> SoftmaxCE
#[rustfmt::skip]
const PYTORCH_CHAIN_X: &[f32] = &[
    // batch 0
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0,
    // batch 1
    0.5, 1.0, 1.5, 2.0,
    2.5, 3.0, 3.5, 4.0,
    4.5, 5.0, 5.5, 6.0,
    6.5, 7.0, 7.5, 8.0,
];
const PYTORCH_CHAIN_CONV_KERNEL: &[f32] = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
const PYTORCH_CHAIN_CONV_BIAS: &[f32] = &[0.1, -0.1];
#[rustfmt::skip]
const PYTORCH_CHAIN_FC_WEIGHT: &[f32] = &[
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
    0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27,
    0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36,
    0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45,
    0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54,
];
const PYTORCH_CHAIN_FC_BIAS: &[f32] = &[0.1, 0.2, 0.3];
const PYTORCH_CHAIN_TARGET: &[f32] = &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
#[rustfmt::skip]

const PYTORCH_CHAIN_LOSS: f32 = 3.9335797;
const PYTORCH_CHAIN_GRAD_CONV_KERNEL: &[f32] = &[
    0.62899423, 0.73382658, 1.04832363, 1.15315592, 0.62899423, 0.73382658, 1.04832363, 1.15315592,
];
const PYTORCH_CHAIN_GRAD_CONV_BIAS: &[f32] = &[0.12196727, 0.1219673];

// ==================== 基础功能测试 ====================

/// 测试 Conv2d 创建
#[test]
fn test_conv2d_creation() -> Result<(), GraphError> {
    let graph = Graph::new();
    let conv = Conv2d::new(&graph, 1, 32, (3, 3), (1, 1), (1, 1), true, "conv1")?;

    // 验证属性
    assert_eq!(conv.in_channels(), 1);
    assert_eq!(conv.out_channels(), 32);
    assert_eq!(conv.kernel_size(), (3, 3));
    assert_eq!(conv.stride(), (1, 1));
    assert_eq!(conv.padding(), (1, 1));

    Ok(())
}

/// 测试 Conv2d 参数形状
#[test]
fn test_conv2d_shapes() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // 3->16 通道，5x5 核
    let conv = Conv2d::new(&graph, 3, 16, (5, 5), (1, 1), (2, 2), true, "conv1")?;

    // 检查卷积核形状：[out_channels, in_channels, kH, kW]
    let k = conv.kernel().value()?.unwrap();
    assert_eq!(k.shape(), &[16, 3, 5, 5]);

    Ok(())
}

/// 测试 Conv2d 前向传播
#[test]
fn test_conv2d_forward() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // 输入: [batch=1, C_in=1, H=4, W=4]
    let x = graph.input(&Tensor::ones(&[1, 1, 4, 4]))?;
    let conv = Conv2d::new(&graph, 1, 2, (2, 2), (1, 1), (0, 0), true, "10")?;

    // 设置卷积核全 1
    conv.kernel().set_value(&Tensor::ones(&[2, 1, 2, 2]))?;

    // 前向传播
    let output = conv.forward(&x);
    output.forward()?;

    // 验证输出形状: [1, 2, 3, 3]
    // H' = (4 + 0 - 2) / 1 + 1 = 3
    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[1, 2, 3, 3]);

    // 验证输出值（全 1 输入，全 1 卷积核，2x2 窗口 → 每个输出为 4.0）
    for c in 0..2 {
        for h in 0..3 {
            for w in 0..3 {
                assert_abs_diff_eq!(out_val[[0, c, h, w]], 4.0, epsilon = 1e-6);
            }
        }
    }

    Ok(())
}

/// 测试 Conv2d 输出尺寸计算
#[test]
fn test_conv2d_output_size() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // padding=1, kernel=3x3, stride=1 → same padding (保持尺寸)
    let x = graph.input(&Tensor::ones(&[1, 1, 4, 4]))?;
    let conv = Conv2d::new(&graph, 1, 1, (3, 3), (1, 1), (1, 1), true, "10")?;

    // 设置卷积核全 1
    conv.kernel().set_value(&Tensor::ones(&[1, 1, 3, 3]))?;

    // 前向传播
    let output = conv.forward(&x);
    output.forward()?;

    // 验证输出形状: [1, 1, 4, 4]（same padding）
    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[1, 1, 4, 4]);

    Ok(())
}

// ==================== 链式连接测试 ====================

/// 测试多层 Conv2d 链式连接
#[test]
fn test_conv2d_chain() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // 典型 CNN 结构: conv1 -> relu -> conv2 -> relu
    let x = graph.input(&Tensor::normal(0.0, 1.0, &[2, 1, 8, 8]))?;
    let conv1 = Conv2d::new(&graph, 1, 4, (3, 3), (1, 1), (1, 1), true, "10")?;
    let h1 = conv1.forward(&x).relu();
    let conv2 = Conv2d::new(&graph, 4, 8, (3, 3), (1, 1), (1, 1), true, "40")?;
    let output = conv2.forward(&h1).relu();

    // 前向传播
    output.forward()?;

    // 验证输出存在且形状正确 (same padding)
    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[2, 8, 8, 8]);

    Ok(())
}

/// 测试 Conv2d + flatten 链式连接（典型 CNN 末端结构）
#[test]
fn test_conv2d_with_flatten() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // conv -> flatten
    let x = graph.input(&Tensor::ones(&[2, 1, 4, 4]))?;
    let conv = Conv2d::new(&graph, 1, 2, (2, 2), (1, 1), (0, 0), true, "10")?;
    let flat = conv.forward(&x).flatten()?;

    // 前向传播
    flat.forward()?;

    // 验证展平输出: [2, 2*3*3] = [2, 18]
    let out_val = flat.value()?.unwrap();
    assert_eq!(out_val.shape(), &[2, 18]);

    Ok(())
}

// ==================== 反向传播测试 ====================

/// 测试 Conv2d 与 Batch 反向传播
#[test]
fn test_conv2d_batch_backward() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 2;

    // 构建网络: conv -> flatten -> fc -> softmax_ce
    let x = graph.input(&Tensor::normal(0.0, 1.0, &[batch_size, 1, 4, 4]))?;
    let conv = Conv2d::new(&graph, 1, 2, (2, 2), (1, 1), (0, 0), true, "10")?;
    let flat = conv.forward(&x).flatten()?;
    // flat 输出: [2, 18]

    let fc = Linear::new(&graph, 18, 3, true, "fc")?;
    let logits = fc.forward(&flat);

    // SoftmaxCrossEntropy Loss
    let labels = graph.input(&Tensor::new(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[batch_size, 3],
    ))?;
    let loss = logits.cross_entropy(&labels)?;

    // 反向传播
    loss.backward()?;

    // 验证卷积核有梯度
    let k_grad = conv.kernel().grad()?.unwrap();
    // 检查梯度形状
    assert_eq!(k_grad.shape(), &[2, 1, 2, 2]);

    Ok(())
}

/// 测试多层 Conv2d 的 Batch 训练
#[test]
fn test_conv2d_chain_batch_training() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 2;

    // 构建网络: conv1 -> relu -> conv2 -> flatten -> fc -> loss
    let x = graph.input(&Tensor::normal(0.0, 1.0, &[batch_size, 1, 6, 6]))?;
    let conv1 = Conv2d::new(&graph, 1, 2, (3, 3), (1, 1), (0, 0), true, "10")?;
    let h1 = conv1.forward(&x).relu();
    // conv1 输出: [2, 2, 4, 4]

    let conv2 = Conv2d::new(&graph, 2, 4, (2, 2), (1, 1), (0, 0), true, "20")?;
    let h2 = conv2.forward(&h1);
    // conv2 输出: [2, 4, 3, 3]

    let flat = h2.flatten()?;
    // flat 输出: [2, 36]

    let fc = Linear::new(&graph, 36, 4, true, "fc")?;
    let logits = fc.forward(&flat);

    // SoftmaxCrossEntropy Loss
    let labels = graph.input(&Tensor::new(
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        &[batch_size, 4],
    ))?;
    let loss = logits.cross_entropy(&labels)?;

    // 反向传播
    loss.backward()?;

    // 验证所有卷积核参数都有梯度
    assert!(conv1.kernel().grad()?.is_some());
    assert!(conv2.kernel().grad()?.is_some());

    Ok(())
}

// ==================== 边界维度测试 ====================

/// 测试单通道输入输出
#[test]
fn test_conv2d_single_channel() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::ones(&[2, 1, 4, 4]))?;

    let conv = Conv2d::new(&graph, 1, 1, (2, 2), (1, 1), (0, 0), true, "10")?;
    let output = conv.forward(&x);
    output.forward()?;

    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[2, 1, 3, 3]);

    Ok(())
}

/// 测试大通道数
#[test]
fn test_conv2d_large_channels() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    let conv = Conv2d::new(&graph, 64, 128, (3, 3), (1, 1), (1, 1), true, "conv")?;

    // 验证参数形状
    let k = conv.kernel().value()?.unwrap();
    assert_eq!(k.shape(), &[128, 64, 3, 3]);

    Ok(())
}

/// 测试带 stride 的 Conv2d
#[test]
fn test_conv2d_with_stride() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::ones(&[2, 1, 8, 8]))?;

    // stride=2 会使输出尺寸减半
    let conv = Conv2d::new(&graph, 1, 4, (3, 3), (2, 2), (1, 1), true, "10")?;
    let output = conv.forward(&x);
    output.forward()?;

    // 输出尺寸: (8 + 2 - 3) / 2 + 1 = 4
    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[2, 4, 4, 4]);

    Ok(())
}

/// 测试非方形卷积核
#[test]
fn test_conv2d_nonsquare_kernel() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::ones(&[2, 1, 8, 8]))?;

    // 使用 3x5 的非方形卷积核
    let conv = Conv2d::new(&graph, 1, 4, (3, 5), (1, 1), (1, 2), true, "10")?;

    // 验证卷积核形状
    let k = conv.kernel().value()?.unwrap();
    assert_eq!(k.shape(), &[4, 1, 3, 5]);

    let output = conv.forward(&x);
    output.forward()?;

    // 输出尺寸:
    // H' = (8 + 2*1 - 3) / 1 + 1 = 8
    // W' = (8 + 2*2 - 5) / 1 + 1 = 8
    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[2, 4, 8, 8]);

    Ok(())
}

/// 测试访问 Conv2d 内部参数
#[test]
fn test_conv2d_access_internal_params() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    let conv = Conv2d::new(&graph, 1, 2, (2, 2), (1, 1), (0, 0), true, "10")?;

    // 应该能访问并修改卷积核
    let custom_kernel = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 1, 2, 2]);
    conv.kernel().set_value(&custom_kernel)?;

    let k = conv.kernel().value()?.unwrap();
    assert_abs_diff_eq!(k[[0, 0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(k[[1, 0, 1, 1]], 8.0, epsilon = 1e-6);

    Ok(())
}

/// 测试典型 MNIST CNN 配置
#[test]
fn test_conv2d_mnist_like() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // 典型 MNIST CNN 第一层
    let conv = Conv2d::new(&graph, 1, 32, (5, 5), (1, 1), (2, 2), true, "conv1")?;

    // 验证参数形状
    let k = conv.kernel().value()?.unwrap();
    assert_eq!(k.shape(), &[32, 1, 5, 5]);

    Ok(())
}

// ==================== Bias 功能测试 ====================

/// 测试 Conv2d 默认包含 bias
#[test]
fn test_conv2d_has_bias() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    let conv = Conv2d::new(&graph, 1, 2, (2, 2), (1, 1), (0, 0), true, "conv")?;

    // 验证 bias 存在且形状正确（4D 形状用于广播）
    let bias = conv.bias().unwrap().value()?.unwrap();
    assert_eq!(bias.shape(), &[1, 2, 1, 1]);

    // 验证 bias 默认初始化为 0
    assert_abs_diff_eq!(bias[[0, 0, 0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(bias[[0, 1, 0, 0]], 0.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Conv2d bias 正确应用
#[test]
fn test_conv2d_bias_applied() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::ones(&[1, 1, 3, 3]))?;

    let conv = Conv2d::new(&graph, 1, 2, (2, 2), (1, 1), (0, 0), true, "10")?;

    // 设置卷积核全 1
    conv.kernel().set_value(&Tensor::ones(&[2, 1, 2, 2]))?;

    // 先测试 bias=0 的情况
    let output = conv.forward(&x);
    output.forward()?;
    let output_no_bias = output.value()?.unwrap().clone();

    // 全 1 输入，全 1 卷积核，2x2 窗口 → 每个输出为 4.0
    for c in 0..2 {
        for h in 0..2 {
            for w in 0..2 {
                assert_abs_diff_eq!(output_no_bias[[0, c, h, w]], 4.0, epsilon = 1e-6);
            }
        }
    }

    // 设置 bias：通道 0 加 1.0，通道 1 加 2.0
    conv.bias()
        .unwrap()
        .set_value(&Tensor::new(&[1.0, 2.0], &[1, 2, 1, 1]))?;

    // 重新前向传播
    output.forward()?;
    let output_with_bias = output.value()?.unwrap();

    // 验证 bias 被正确加上：通道 0 应该是 5.0，通道 1 应该是 6.0
    for h in 0..2 {
        for w in 0..2 {
            assert_abs_diff_eq!(output_with_bias[[0, 0, h, w]], 5.0, epsilon = 1e-6);
            assert_abs_diff_eq!(output_with_bias[[0, 1, h, w]], 6.0, epsilon = 1e-6);
        }
    }

    Ok(())
}

/// 测试 Conv2d bias 的梯度传播
#[test]
fn test_conv2d_bias_gradient() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let batch_size = 2;

    let x = graph.input(&Tensor::normal(0.0, 1.0, &[batch_size, 1, 4, 4]))?;
    let conv = Conv2d::new(&graph, 1, 2, (2, 2), (1, 1), (0, 0), true, "10")?;
    let flat = conv.forward(&x).flatten()?;

    // 简单分类器
    let fc = Linear::new(&graph, 18, 3, true, "fc")?;
    let logits = fc.forward(&flat);
    let labels = graph.input(&Tensor::new(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[batch_size, 3],
    ))?;
    let loss = logits.cross_entropy(&labels)?;

    // 反向传播
    loss.backward()?;

    // 验证 bias 有梯度
    let b_grad = conv.bias().unwrap().grad()?;
    assert!(b_grad.is_some(), "bias 应该有梯度");
    assert_eq!(
        b_grad.unwrap().shape(),
        &[1, 2, 1, 1],
        "bias 梯度形状应该正确"
    );

    Ok(())
}

/// 测试 Conv2d 无 bias
#[test]
fn test_conv2d_no_bias() -> Result<(), GraphError> {
    let graph = Graph::new();
    let conv = Conv2d::new(&graph, 1, 4, (3, 3), (1, 1), (0, 0), false, "conv")?;

    // 验证无 bias
    assert!(conv.bias().is_none());

    // 验证参数只有 kernel
    let params = conv.parameters();
    assert_eq!(params.len(), 1);

    Ok(())
}

// ==================== PyTorch 数值对照测试 ====================

/// 测试前向传播数值（与 PyTorch 对照）
#[test]
fn test_conv2d_forward_pytorch_comparison() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // batch=1, C_in=1, H=4, W=4, C_out=2, kernel=2x2
    let batch_size = 1;
    let in_channels = 1;
    let out_channels = 2;

    let x = graph.input(&Tensor::new(
        PYTORCH_FWD_X,
        &[batch_size, in_channels, 4, 4],
    ))?;
    let conv = Conv2d::new(
        &graph,
        in_channels,
        out_channels,
        (2, 2),
        (1, 1),
        (0, 0),
        true,
        "conv",
    )?;

    // 设置与 PyTorch 相同的参数
    conv.kernel().set_value(&Tensor::new(
        PYTORCH_FWD_KERNEL,
        &[out_channels, in_channels, 2, 2],
    ))?;
    conv.bias()
        .unwrap()
        .set_value(&Tensor::new(PYTORCH_FWD_BIAS, &[1, out_channels, 1, 1]))?;

    // 前向传播
    let output = conv.forward(&x);
    output.forward()?;

    // 验证输出
    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[batch_size, out_channels, 3, 3]);

    let output_data = out_val.data_as_slice();
    println!("Conv2d 前向传播输出:");
    for (i, (&actual, &expected)) in output_data
        .iter()
        .zip(PYTORCH_FWD_OUTPUT.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
        println!(
            "  output[{}]: actual={:.4}, expected={:.4}",
            i, actual, expected
        );
    }

    println!("✅ Conv2d 前向传播与 PyTorch 一致");
    Ok(())
}

/// 测试反向传播梯度数值（与 PyTorch 对照）
#[test]
fn test_conv2d_backward_pytorch_comparison() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // batch=1, C_in=1, H=3, W=3, C_out=1, kernel=2x2
    let batch_size = 1;
    let in_channels = 1;
    let out_channels = 1;

    let x = graph.input(&Tensor::new(
        PYTORCH_BWD_X,
        &[batch_size, in_channels, 3, 3],
    ))?;
    let conv = Conv2d::new(
        &graph,
        in_channels,
        out_channels,
        (2, 2),
        (1, 1),
        (0, 0),
        true,
        "conv",
    )?;

    // 设置与 PyTorch 相同的参数
    conv.kernel().set_value(&Tensor::new(
        PYTORCH_BWD_KERNEL,
        &[out_channels, in_channels, 2, 2],
    ))?;
    conv.bias()
        .unwrap()
        .set_value(&Tensor::new(PYTORCH_BWD_BIAS, &[1, out_channels, 1, 1]))?;

    let conv_out = conv.forward(&x);
    let target = graph.input(&Tensor::new(
        PYTORCH_BWD_TARGET,
        &[batch_size, out_channels, 2, 2],
    ))?;
    let loss = conv_out.mse_loss(&target)?;

    // 前向传播
    loss.forward()?;

    // 验证输出
    let out_val = conv_out.value()?.unwrap();
    let output_data = out_val.data_as_slice();
    println!("Conv2d 输出:");
    for (i, (&actual, &expected)) in output_data
        .iter()
        .zip(PYTORCH_BWD_OUTPUT.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
        println!(
            "  output[{}]: actual={:.4}, expected={:.4}",
            i, actual, expected
        );
    }

    // 验证 loss
    let loss_val = loss.value()?.unwrap();
    assert_abs_diff_eq!(loss_val[[0, 0]], PYTORCH_BWD_LOSS, epsilon = 1e-2);
    println!(
        "\nloss: actual={:.6}, expected={:.6}",
        loss_val[[0, 0]],
        PYTORCH_BWD_LOSS
    );

    // 反向传播
    loss.backward()?;

    // 验证卷积核梯度
    let grad_kernel = conv.kernel().grad()?.unwrap();
    let grad_kernel_data = grad_kernel.data_as_slice();
    println!("\n卷积核梯度:");
    for (i, (&actual, &expected)) in grad_kernel_data
        .iter()
        .zip(PYTORCH_BWD_GRAD_KERNEL.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-2);
        println!(
            "  grad_kernel[{}]: actual={:.4}, expected={:.4}",
            i, actual, expected
        );
    }

    // 验证偏置梯度
    let grad_bias = conv.bias().unwrap().grad()?.unwrap();
    let grad_bias_data = grad_bias.data_as_slice();
    println!("\n偏置梯度:");
    for (i, (&actual, &expected)) in grad_bias_data
        .iter()
        .zip(PYTORCH_BWD_GRAD_BIAS.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-2);
        println!(
            "  grad_bias[{}]: actual={:.4}, expected={:.4}",
            i, actual, expected
        );
    }

    println!("\n✅ Conv2d 反向传播梯度与 PyTorch 一致");
    Ok(())
}

/// 测试 CNN 链式网络反向传播（与 PyTorch 对照）
#[test]
fn test_conv2d_chain_backward_pytorch_comparison() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // 网络结构: conv -> relu -> flatten -> linear -> softmax_cross_entropy
    let batch_size = 2;
    let in_channels = 1;
    let out_channels = 2;
    let num_classes = 3;

    let x = graph.input(&Tensor::new(
        PYTORCH_CHAIN_X,
        &[batch_size, in_channels, 4, 4],
    ))?;
    let conv = Conv2d::new(
        &graph,
        in_channels,
        out_channels,
        (2, 2),
        (1, 1),
        (0, 0),
        true,
        "conv",
    )?;

    // 设置与 PyTorch 相同的参数
    conv.kernel().set_value(&Tensor::new(
        PYTORCH_CHAIN_CONV_KERNEL,
        &[out_channels, in_channels, 2, 2],
    ))?;
    conv.bias().unwrap().set_value(&Tensor::new(
        PYTORCH_CHAIN_CONV_BIAS,
        &[1, out_channels, 1, 1],
    ))?;

    // 保存 conv 输出以便后续验证
    let conv_out = conv.forward(&x);
    let h = conv_out.relu();
    // conv 输出: [2, 2, 3, 3] -> flatten: [2, 18]
    let flat = h.flatten()?;

    // 使用 Linear API
    let fc = Linear::new(&graph, 18, num_classes, true, "fc")?;
    fc.weights()
        .set_value(&Tensor::new(PYTORCH_CHAIN_FC_WEIGHT, &[18, num_classes]))?;
    fc.bias()
        .unwrap()
        .set_value(&Tensor::new(PYTORCH_CHAIN_FC_BIAS, &[1, num_classes]))?;

    let fc_out = fc.forward(&flat);

    // SoftmaxCrossEntropy Loss
    let target = graph.input(&Tensor::new(
        PYTORCH_CHAIN_TARGET,
        &[batch_size, num_classes],
    ))?;
    let loss = fc_out.cross_entropy(&target)?;

    // 先前向传播验证输出
    loss.forward()?;

    // 验证 loss
    let loss_val = loss.value()?.unwrap();
    assert_abs_diff_eq!(loss_val[[0, 0]], PYTORCH_CHAIN_LOSS, epsilon = 1e-1);
    println!(
        "loss: actual={:.6}, expected={:.6}",
        loss_val[[0, 0]],
        PYTORCH_CHAIN_LOSS
    );

    // 反向传播
    loss.backward()?;

    // 验证卷积核梯度
    let grad_kernel = conv.kernel().grad()?.unwrap();
    let grad_kernel_data = grad_kernel.data_as_slice();
    println!("\n卷积核梯度:");
    for (i, (&actual, &expected)) in grad_kernel_data
        .iter()
        .zip(PYTORCH_CHAIN_GRAD_CONV_KERNEL.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 5e-2);
        println!(
            "  grad_kernel[{}]: actual={:.6}, expected={:.6}",
            i, actual, expected
        );
    }

    // 验证偏置梯度
    let grad_bias = conv.bias().unwrap().grad()?.unwrap();
    let grad_bias_data = grad_bias.data_as_slice();
    println!("\n偏置梯度:");
    for (i, (&actual, &expected)) in grad_bias_data
        .iter()
        .zip(PYTORCH_CHAIN_GRAD_CONV_BIAS.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 5e-2);
        println!(
            "  grad_bias[{}]: actual={:.6}, expected={:.6}",
            i, actual, expected
        );
    }

    println!("\n✅ CNN 链式网络反向传播与 PyTorch 一致");
    Ok(())
}

// ==================== Module trait 测试 ====================

/// 测试 Conv2d struct Module trait
#[test]
fn test_conv2d_module_trait() -> Result<(), GraphError> {
    let graph = Graph::new();
    let conv = Conv2d::new(&graph, 1, 32, (3, 3), (1, 1), (1, 1), true, "conv1")?;

    // 验证 Module trait
    let params = conv.parameters();
    assert_eq!(params.len(), 2); // kernel + bias

    Ok(())
}

/// 测试 Conv2d struct 带种子的可重复性
#[test]
fn test_conv2d_seeded_reproducibility() -> Result<(), GraphError> {
    // 使用相同种子的 Graph 应产生相同权重
    let graph1 = Graph::new_with_seed(42);
    let conv1 = Conv2d::new(&graph1, 1, 4, (3, 3), (1, 1), (0, 0), true, "conv")?;
    let k1 = conv1.kernel().value()?.unwrap();

    let graph2 = Graph::new_with_seed(42);
    let conv2 = Conv2d::new(&graph2, 1, 4, (3, 3), (1, 1), (0, 0), true, "conv")?;
    let k2 = conv2.kernel().value()?.unwrap();

    // 相同种子应产生相同权重
    assert_eq!(k1.data_as_slice(), k2.data_as_slice());

    Ok(())
}

/// 测试 Conv2d struct 链式调用
#[test]
fn test_conv2d_chain_api() -> Result<(), GraphError> {
    let graph = Graph::new();
    let x = graph.input(&Tensor::normal(0.0, 1.0, &[2, 1, 8, 8]))?;

    // conv -> relu -> conv -> relu
    let conv1 = Conv2d::new(&graph, 1, 4, (3, 3), (1, 1), (1, 1), true, "10")?;
    let a1 = conv1.forward(&x).relu();
    let conv2 = Conv2d::new(&graph, 4, 8, (3, 3), (1, 1), (1, 1), true, "40")?;
    let output = conv2.forward(&a1).relu();

    // 前向传播
    output.forward()?;

    // 验证输出形状 [2, 8, 8, 8]
    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[2, 8, 8, 8]);

    // 收集所有参数
    let mut params = conv1.parameters();
    params.extend(conv2.parameters());
    assert_eq!(params.len(), 4); // 2 * (kernel + bias)

    Ok(())
}
