/*
 * @Author       : 老董
 * @Date         : 2026-04-19
 * @Description  : ConvTranspose2d layer 单元测试（PyTorch 风格 API，含数值对照）
 *
 * 参考值来源: PyTorch torch.nn.ConvTranspose2d 手动计算
 *
 * ConvTranspose2d 核心公式：
 * H_out = (H_in - 1) * stride - 2 * padding + kernel + output_padding
 * 卷积核布局: [C_in, C_out, kH, kW]
 */

use crate::nn::layer::{ConvTranspose2d, Linear};
use crate::nn::{Graph, GraphError, Module, VarActivationOps, VarLossOps, VarShapeOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试 ConvTranspose2d 创建
#[test]
fn test_conv_transpose2d_creation() -> Result<(), GraphError> {
    let graph = Graph::new();
    let deconv = ConvTranspose2d::new(
        &graph,
        1,
        32,
        (3, 3),
        (1, 1),
        (0, 0),
        (0, 0),
        true,
        "deconv1",
    )?;

    assert_eq!(deconv.in_channels(), 1);
    assert_eq!(deconv.out_channels(), 32);
    assert_eq!(deconv.kernel_size(), (3, 3));
    assert_eq!(deconv.stride(), (1, 1));
    assert_eq!(deconv.padding(), (0, 0));
    assert_eq!(deconv.output_padding(), (0, 0));

    Ok(())
}

/// 测试 ConvTranspose2d 参数形状
#[test]
fn test_conv_transpose2d_shapes() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // 3→16 通道，5×5 核
    let deconv = ConvTranspose2d::new(
        &graph,
        3,
        16,
        (5, 5),
        (1, 1),
        (0, 0),
        (0, 0),
        true,
        "deconv1",
    )?;

    // 卷积核形状: [in_channels, out_channels, kH, kW]
    let k = deconv.kernel().value()?.unwrap();
    assert_eq!(k.shape(), &[3, 16, 5, 5]);

    Ok(())
}

/// 测试 ConvTranspose2d 前向传播：全 1 输入 × 全 1 核（与 raw op 测试对齐）
#[test]
fn test_conv_transpose2d_forward() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // 输入: [1, 1, 2, 2]，转置卷积核: [1, 1, 2, 2]
    let x = graph.input(&Tensor::ones(&[1, 1, 2, 2]))?;
    let deconv = ConvTranspose2d::new(
        &graph,
        1,
        1,
        (2, 2),
        (1, 1),
        (0, 0),
        (0, 0),
        false,
        "deconv",
    )?;

    // 设置卷积核全 1
    deconv.kernel().set_value(&Tensor::ones(&[1, 1, 2, 2]))?;

    let output = deconv.forward(&x);
    output.forward()?;

    // H_out = (2-1)*1 - 0 + 2 + 0 = 3
    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[1, 1, 3, 3]);

    // 对照 PyTorch: ConvTranspose2d(ones[1,1,2,2], ones[1,1,2,2]) = [[1,2,1],[2,4,2],[1,2,1]]
    #[rustfmt::skip]
    let expected = [
        1.0, 2.0, 1.0,
        2.0, 4.0, 2.0,
        1.0, 2.0, 1.0,
    ];
    let flat = out_val.data_as_slice();
    for (i, (&actual, &expected)) in flat.iter().zip(expected.iter()).enumerate() {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-5,);
        let _ = i;
    }

    Ok(())
}

/// 测试 ConvTranspose2d 输出尺寸计算（stride=2 上采样）
#[test]
fn test_conv_transpose2d_stride2_upsample() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // stride=2: 上采样 2 倍
    let x = graph.input(&Tensor::ones(&[1, 1, 4, 4]))?;
    let deconv = ConvTranspose2d::new(
        &graph,
        1,
        1,
        (3, 3),
        (2, 2),
        (1, 1),
        (0, 0),
        false,
        "deconv",
    )?;

    deconv.kernel().set_value(&Tensor::ones(&[1, 1, 3, 3]))?;

    let output = deconv.forward(&x);
    output.forward()?;

    // H_out = (4-1)*2 - 2*1 + 3 + 0 = 7
    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[1, 1, 7, 7]);

    Ok(())
}

/// 测试 ConvTranspose2d 多通道
#[test]
fn test_conv_transpose2d_multi_channel() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // in_ch=4, out_ch=2
    let x = graph.input(&Tensor::ones(&[2, 4, 3, 3]))?;
    let deconv =
        ConvTranspose2d::new(&graph, 4, 2, (3, 3), (1, 1), (0, 0), (0, 0), true, "deconv")?;

    let output = deconv.forward(&x);
    output.forward()?;

    // H_out = (3-1)*1 + 3 = 5
    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[2, 2, 5, 5]);

    Ok(())
}

// ==================== Bias 测试 ====================

/// 测试 ConvTranspose2d 含 bias
#[test]
fn test_conv_transpose2d_has_bias() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    let deconv =
        ConvTranspose2d::new(&graph, 1, 2, (2, 2), (1, 1), (0, 0), (0, 0), true, "deconv")?;

    let bias = deconv.bias().unwrap().value()?.unwrap();
    assert_eq!(bias.shape(), &[1, 2, 1, 1]);
    assert_abs_diff_eq!(bias[[0, 0, 0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(bias[[0, 1, 0, 0]], 0.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 ConvTranspose2d bias 正确应用
#[test]
fn test_conv_transpose2d_bias_applied() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::ones(&[1, 1, 2, 2]))?;

    let deconv =
        ConvTranspose2d::new(&graph, 1, 1, (2, 2), (1, 1), (0, 0), (0, 0), true, "deconv")?;

    deconv.kernel().set_value(&Tensor::ones(&[1, 1, 2, 2]))?;
    deconv
        .bias()
        .unwrap()
        .set_value(&Tensor::new(&[0.5], &[1, 1, 1, 1]))?;

    let output = deconv.forward(&x);
    output.forward()?;

    let out_val = output.value()?.unwrap();
    // 中心位置原值 4.0 + bias 0.5 = 4.5
    assert_abs_diff_eq!(out_val[[0, 0, 1, 1]], 4.5, epsilon = 1e-5);
    // 角落原值 1.0 + bias 0.5 = 1.5
    assert_abs_diff_eq!(out_val[[0, 0, 0, 0]], 1.5, epsilon = 1e-5);

    Ok(())
}

/// 测试 ConvTranspose2d 无 bias
#[test]
fn test_conv_transpose2d_no_bias() -> Result<(), GraphError> {
    let graph = Graph::new();
    let deconv = ConvTranspose2d::new(
        &graph,
        1,
        4,
        (3, 3),
        (1, 1),
        (0, 0),
        (0, 0),
        false,
        "deconv",
    )?;

    assert!(deconv.bias().is_none());
    assert_eq!(deconv.parameters().len(), 1);

    Ok(())
}

// ==================== 反向传播测试 ====================

/// 测试 ConvTranspose2d 反向传播（kernel 梯度存在）
#[test]
fn test_conv_transpose2d_backward_has_grad() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    let x = graph.input(&Tensor::normal(0.0, 1.0, &[2, 1, 4, 4]))?;
    let deconv =
        ConvTranspose2d::new(&graph, 1, 2, (2, 2), (1, 1), (0, 0), (0, 0), true, "deconv")?;
    // 输出: [2, 2, 5, 5] → flatten: [2, 50]
    let flat = deconv.forward(&x).flatten()?;

    let fc = Linear::new(&graph, 50, 3, true, "fc")?;
    let logits = fc.forward(&flat);

    let labels = graph.input(&Tensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]))?;
    let loss = logits.cross_entropy(&labels)?;

    loss.backward()?;

    let k_grad = deconv.kernel().grad()?.unwrap();
    assert_eq!(k_grad.shape(), &[1, 2, 2, 2]);

    let b_grad = deconv.bias().unwrap().grad()?.unwrap();
    assert_eq!(b_grad.shape(), &[1, 2, 1, 1]);

    Ok(())
}

/// 测试 ConvTranspose2d 反向传播数值（全 1 输入/核，sum 上游）
#[test]
fn test_conv_transpose2d_backward_numerical() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    let x = graph.input(&Tensor::ones(&[1, 1, 2, 2]))?;
    let deconv = ConvTranspose2d::new(
        &graph,
        1,
        1,
        (2, 2),
        (1, 1),
        (0, 0),
        (0, 0),
        false,
        "deconv",
    )?;
    deconv.kernel().set_value(&Tensor::ones(&[1, 1, 2, 2]))?;

    // 对输出取 MSE（target=0 等价于 sum(output^2) / n）
    let target = graph.input(&Tensor::zeros(&[1, 1, 3, 3]))?;
    let loss = deconv.forward(&x).mse_loss(&target)?;

    loss.backward()?;

    // 验证 kernel 梯度存在且有意义
    let k_grad = deconv.kernel().grad()?.unwrap();
    assert_eq!(k_grad.shape(), &[1, 1, 2, 2]);
    // 梯度不应全零
    let sum: f32 = k_grad.data_as_slice().iter().map(|v| v.abs()).sum();
    assert!(sum > 0.0, "kernel 梯度不应全零");

    Ok(())
}

// ==================== 链式连接测试 ====================

/// 测试 Conv2d → ConvTranspose2d（编解码器结构）
#[test]
fn test_conv_then_deconv_chain() -> Result<(), GraphError> {
    use crate::nn::layer::Conv2d;

    let graph = Graph::new_with_seed(42);

    // 编码器: Conv2d(1→4, k=3, stride=2) [8,8] → [3,3]
    let x = graph.input(&Tensor::normal(0.0, 0.1, &[2, 1, 8, 8]))?;
    let encoder = Conv2d::new(&graph, 1, 4, (3, 3), (2, 2), (0, 0), (1, 1), true, "enc")?;
    let enc_out = encoder.forward(&x).relu();

    // 解码器: ConvTranspose2d(4→1, k=3, stride=2) [3,3] → [7,7]
    let decoder = ConvTranspose2d::new(&graph, 4, 1, (3, 3), (2, 2), (0, 0), (0, 0), true, "dec")?;
    let dec_out = decoder.forward(&enc_out);

    dec_out.forward()?;

    let out_val = dec_out.value()?.unwrap();
    // enc: (8-3)/2+1 = 3 → dec: (3-1)*2+3 = 7
    assert_eq!(out_val.shape(), &[2, 1, 7, 7]);

    Ok(())
}

// ==================== Module trait 测试 ====================

/// 测试 ConvTranspose2d Module trait
#[test]
fn test_conv_transpose2d_module_trait() -> Result<(), GraphError> {
    let graph = Graph::new();
    let deconv = ConvTranspose2d::new(
        &graph,
        1,
        32,
        (3, 3),
        (1, 1),
        (0, 0),
        (0, 0),
        true,
        "deconv",
    )?;

    let params = deconv.parameters();
    assert_eq!(params.len(), 2); // kernel + bias

    Ok(())
}

/// 测试 ConvTranspose2d 种子可复现性
#[test]
fn test_conv_transpose2d_seeded_reproducibility() -> Result<(), GraphError> {
    let graph1 = Graph::new_with_seed(42);
    let d1 = ConvTranspose2d::new(
        &graph1,
        1,
        4,
        (3, 3),
        (1, 1),
        (0, 0),
        (0, 0),
        true,
        "deconv",
    )?;
    let k1 = d1.kernel().value()?.unwrap();

    let graph2 = Graph::new_with_seed(42);
    let d2 = ConvTranspose2d::new(
        &graph2,
        1,
        4,
        (3, 3),
        (1, 1),
        (0, 0),
        (0, 0),
        true,
        "deconv",
    )?;
    let k2 = d2.kernel().value()?.unwrap();

    assert_eq!(k1.data_as_slice(), k2.data_as_slice());

    Ok(())
}

/// 测试 output_padding 参数
#[test]
fn test_conv_transpose2d_output_padding() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // stride=2, output_padding=1: 补偿 stride 歧义
    let x = graph.input(&Tensor::ones(&[1, 1, 4, 4]))?;
    let deconv = ConvTranspose2d::new(
        &graph,
        1,
        1,
        (3, 3),
        (2, 2),
        (1, 1),
        (1, 1),
        false,
        "deconv",
    )?;
    deconv.kernel().set_value(&Tensor::ones(&[1, 1, 3, 3]))?;

    let output = deconv.forward(&x);
    output.forward()?;

    // H_out = (4-1)*2 - 2*1 + 3 + 1 = 8
    let out_val = output.value()?.unwrap();
    assert_eq!(out_val.shape(), &[1, 1, 8, 8]);

    Ok(())
}
