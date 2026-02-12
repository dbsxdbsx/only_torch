/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : Conv2d 节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（底层 inner_rc API）→ simple; padding; 4D batch; 多输出通道; stride; 错误处理
 * 2. VJP 单元测试（底层 calc_grad_to_parent_index）→ 对 input; 对 kernel; batch 对 kernel; batch 对 input
 * 3. 端到端反向传播测试（底层构建完整图）→ kernel 梯度验证; batch 网络
 * 4. 梯度累积测试
 * 5. 动态形状 + 动态 batch 测试
 * 6. Create API 测试（保持原样）
 *
 * 注意：Conv2d 没有高层 Var API（无 .conv2d() 方法），所有测试均使用底层 inner_rc() API。
 *
 * Key: Conv2d 是二元运算 (input, kernel)。
 * - 输入 [batch, C_in, H, W]，卷积核 [C_out, C_in, kH, kW]
 * - 输出 [batch, C_out, oH, oW]，oH = (H + 2*pH - kH) / sH + 1
 * - VJP 向 input (idx=0): 转置卷积（full convolution with flipped kernel）
 * - VJP 向 kernel (idx=1): 互相关（input 的窗口与 upstream 的相关）
 */

use crate::assert_err;
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::rc::Rc;

// ==================== 1. 前向传播测试（底层 inner_rc API）====================

/// 测试 Conv2d 前向传播（简单情况）：全1输入×全1核→每个位置=4
#[test]
fn test_conv2d_forward_simple() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 输入: [1, 1, 3, 3]，卷积核: [1, 1, 2, 2]
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 3, 3], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("kernel"))?;

    let conv = inner.borrow_mut().create_conv2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (0, 0),
        Some("conv"),
    )?;

    input.set_value(Some(&Tensor::ones(&[1, 1, 3, 3])))?;
    kernel.set_value(Some(&Tensor::ones(&[1, 1, 2, 2])))?;

    conv.forward_recursive(1, false)?;

    // 输出 [1, 1, 2, 2]，2×2 窗口求和 = 4.0
    let output = conv.value().unwrap();
    assert_eq!(output.shape(), &[1, 1, 2, 2]);
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 0, 1]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 4.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Conv2d 前向传播（带 padding）：padding=1 保持尺寸
#[test]
fn test_conv2d_forward_with_padding() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 输入: [1, 1, 3, 3]，卷积核: [1, 1, 3, 3]，padding=1
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 3, 3], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 3, 3], Some("kernel"))?;

    let conv = inner.borrow_mut().create_conv2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (1, 1),
        Some("conv"),
    )?;

    input.set_value(Some(&Tensor::ones(&[1, 1, 3, 3])))?;
    kernel.set_value(Some(&Tensor::ones(&[1, 1, 3, 3])))?;

    conv.forward_recursive(1, false)?;

    // 输出 [1, 1, 3, 3]（same padding）
    let output = conv.value().unwrap();
    assert_eq!(output.shape(), &[1, 1, 3, 3]);

    // 中心位置：3×3 窗口全有值，sum = 9
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 9.0, epsilon = 1e-6);
    // 角落位置：只有 2×2 区域有值（其余被 padding 的 0 填充），sum = 4
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 4.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Conv2d 前向传播（4D 批量输入）：batch=2
#[test]
fn test_conv2d_forward_batch() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 输入: [2, 1, 4, 4]，卷积核: [1, 1, 2, 2]
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 1, 4, 4], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("kernel"))?;

    let conv = inner.borrow_mut().create_conv2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (0, 0),
        Some("conv"),
    )?;

    // 第一个 batch 全 1，第二个 batch 全 2
    let mut input_data = vec![1.0f32; 16];
    input_data.extend(vec![2.0f32; 16]);
    let input_val = Tensor::new(&input_data, &[2, 1, 4, 4]);
    let kernel_val = Tensor::ones(&[1, 1, 2, 2]);

    input.set_value(Some(&input_val))?;
    kernel.set_value(Some(&kernel_val))?;

    conv.forward_recursive(1, false)?;

    let output = conv.value().unwrap();
    assert_eq!(output.shape(), &[2, 1, 3, 3]);

    // 第一个 batch: 2×2 窗口全 1, sum = 4
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 4.0, epsilon = 1e-6);
    // 第二个 batch: 2×2 窗口全 2, sum = 8
    assert_abs_diff_eq!(output[[1, 0, 0, 0]], 8.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Conv2d 多输出通道
#[test]
fn test_conv2d_multi_output_channels() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 输入: [1, 1, 4, 4]，卷积核: [2, 1, 2, 2]（2 个输出通道）
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 4, 4], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 1, 2, 2], Some("kernel"))?;

    let conv = inner.borrow_mut().create_conv2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (0, 0),
        Some("conv"),
    )?;

    // 输入全 1；卷积核：第一个全 1，第二个全 2
    let input_val = Tensor::ones(&[1, 1, 4, 4]);
    let mut kernel_data = vec![1.0f32; 4];
    kernel_data.extend(vec![2.0f32; 4]);
    let kernel_val = Tensor::new(&kernel_data, &[2, 1, 2, 2]);

    input.set_value(Some(&input_val))?;
    kernel.set_value(Some(&kernel_val))?;

    conv.forward_recursive(1, false)?;

    let output = conv.value().unwrap();
    assert_eq!(output.shape(), &[1, 2, 3, 3]);

    // 第一个输出通道: 2×2 窗口 × 全1核 = 4
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 4.0, epsilon = 1e-6);
    // 第二个输出通道: 2×2 窗口 × 全2核 = 8
    assert_abs_diff_eq!(output[[0, 1, 0, 0]], 8.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Conv2d 带 stride
#[test]
fn test_conv2d_forward_with_stride() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 输入: [1, 1, 8, 8]，卷积核: [1, 1, 3, 3]，stride=2
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 8, 8], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 3, 3], Some("kernel"))?;

    let conv = inner.borrow_mut().create_conv2d_node(
        vec![input.clone(), kernel.clone()],
        (2, 2),
        (0, 0),
        Some("conv"),
    )?;

    input.set_value(Some(&Tensor::ones(&[1, 1, 8, 8])))?;
    kernel.set_value(Some(&Tensor::ones(&[1, 1, 3, 3])))?;

    conv.forward_recursive(1, false)?;

    // 输出 H' = (8 - 3) / 2 + 1 = 3
    let output = conv.value().unwrap();
    assert_eq!(output.shape(), &[1, 1, 3, 3]);

    // 全 1 输入 × 全 1 核(3×3) → 每个位置 = 9
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 9.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 2, 2]], 9.0, epsilon = 1e-6);

    Ok(())
}

/// 测试通道数不匹配 → 报错
#[test]
fn test_conv2d_channel_mismatch() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 输入 C_in=3，卷积核 C_in=1 → 不匹配
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3, 5, 5], Some("input"))
        .unwrap();
    let kernel = inner
        .borrow_mut()
        .create_parameter_node(&[2, 1, 3, 3], Some("kernel"))
        .unwrap();

    let result = inner.borrow_mut().create_conv2d_node(
        vec![input, kernel],
        (1, 1),
        (0, 0),
        Some("conv"),
    );
    assert_err!(result, GraphError::ShapeMismatch { message, .. } if message.contains("通道数"));
}

/// 测试无效的输入维度（2D 对 Conv2d 无效）
#[test]
fn test_conv2d_invalid_input_dims() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 输入: [5, 5]（2D，缺少 batch 和通道维度）
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[5, 5], Some("input"))
        .unwrap();
    let kernel = inner
        .borrow_mut()
        .create_parameter_node(&[2, 1, 3, 3], Some("kernel"))
        .unwrap();

    let result = inner.borrow_mut().create_conv2d_node(
        vec![input, kernel],
        (1, 1),
        (0, 0),
        Some("conv"),
    );
    assert_err!(result, GraphError::ShapeMismatch { message, .. }
        if message.contains("4D"));
}

/// 测试无效的卷积核维度（2D 卷积核无效）
#[test]
fn test_conv2d_invalid_kernel_dims() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 5, 5], Some("input"))
        .unwrap();
    // 卷积核: [3, 3]（2D，缺少通道维度）
    let kernel = inner
        .borrow_mut()
        .create_parameter_node(&[3, 3], Some("kernel"))
        .unwrap();

    let result = inner.borrow_mut().create_conv2d_node(
        vec![input, kernel],
        (1, 1),
        (0, 0),
        Some("conv"),
    );
    assert_err!(result, GraphError::ShapeMismatch { message, .. }
        if message.contains("4D") || message.contains("C_out"));
}

// ==================== 2. VJP 单元测试（底层 calc_grad_to_parent_index）====================

/// 测试 Conv2d 对输入的 VJP（parent_index=0）
///
/// 输入 [1, 1, 2, 2], 卷积核 [1, 1, 2, 2] → 输出 [1, 1, 1, 1]
/// upstream 全 1，grad 形状 = input 形状，值 = kernel 值
#[test]
fn test_conv2d_vjp_to_input() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("kernel"))?;

    let conv = inner.borrow_mut().create_conv2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (0, 0),
        Some("conv"),
    )?;

    let input_val = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let kernel_val = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    input.set_value(Some(&input_val))?;
    kernel.set_value(Some(&kernel_val))?;
    conv.forward_recursive(1, false)?;

    // upstream 形状 = conv 输出: [1, 1, 1, 1]
    let upstream = Tensor::ones(&[1, 1, 1, 1]);
    let grad = conv.calc_grad_to_parent_index(0, &upstream)?;

    // grad 形状 = input 形状
    assert_eq!(grad.shape(), &[1, 1, 2, 2]);

    // ∂out/∂input[i,j] = kernel[i,j]（因为输出只有一个位置）
    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 0, 1]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 1, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 1, 1]], 4.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Conv2d 对卷积核的 VJP（parent_index=1）
///
/// 输入 [1, 1, 2, 2], 卷积核 [1, 1, 2, 2] → 输出 [1, 1, 1, 1]
/// upstream 全 1，grad[kh,kw] = input[kh,kw]
#[test]
fn test_conv2d_vjp_to_kernel() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("kernel"))?;

    let conv = inner.borrow_mut().create_conv2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (0, 0),
        Some("conv"),
    )?;

    let input_val = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let kernel_val = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
    input.set_value(Some(&input_val))?;
    kernel.set_value(Some(&kernel_val))?;
    conv.forward_recursive(1, false)?;

    let upstream = Tensor::ones(&[1, 1, 1, 1]);
    let grad = conv.calc_grad_to_parent_index(1, &upstream)?;

    assert_eq!(grad.shape(), &[1, 1, 2, 2]);

    // ∂out/∂kernel[kh,kw] = input[kh,kw]（因为输出只有一个位置）
    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 0, 1]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 1, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 1, 1]], 4.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Conv2d 对卷积核的 VJP（batch=2）
///
/// 输入 [2, 1, 3, 3], 卷积核 [1, 1, 2, 2] → 输出 [2, 1, 2, 2]
/// grad 需要在 batch 维度求和
#[test]
fn test_conv2d_vjp_to_kernel_batch() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 1, 3, 3], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_parameter_node(&[1, 1, 2, 2], Some("kernel"))?;

    let conv = inner.borrow_mut().create_conv2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (0, 0),
        Some("conv"),
    )?;

    // batch 0 全 1, batch 1 全 2
    let mut input_data = vec![1.0f32; 9];
    input_data.extend(vec![2.0f32; 9]);
    let input_val = Tensor::new(&input_data, &[2, 1, 3, 3]);
    let kernel_val = Tensor::ones(&[1, 1, 2, 2]);

    input.set_value(Some(&input_val))?;
    kernel.set_value(Some(&kernel_val))?;
    conv.forward_recursive(1, false)?;

    // upstream 全 1，形状 [2, 1, 2, 2]
    let upstream = Tensor::ones(&[2, 1, 2, 2]);
    let grad = conv.calc_grad_to_parent_index(1, &upstream)?;

    assert_eq!(grad.shape(), &[1, 1, 2, 2]);

    // grad[kh,kw] = sum over (b, oh, ow) of upstream[b,oc,oh,ow] * input[b,ic,oh+kh,ow+kw]
    // upstream 全 1:
    //   batch 0: sum_{oh,ow} input[0,0,oh+kh,ow+kw]（每个 kh,kw 有 2×2=4 个位置）
    //            = input[0,0,0,0]+input[0,0,0,1]+input[0,0,1,0]+input[0,0,1,1] = 4
    //   batch 1: 同理 = 2+2+2+2 = 8
    //   total = 12
    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 12.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Conv2d 对输入的 VJP（多输出位置）
///
/// 输入 [1, 1, 3, 3]，卷积核 [1, 1, 2, 2] → 输出 [1, 1, 2, 2]
/// 验证反卷积效果：grad 是 upstream 与翻转 kernel 的 full convolution
#[test]
fn test_conv2d_vjp_to_input_multi_output() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 3, 3], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("kernel"))?;

    let conv = inner.borrow_mut().create_conv2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (0, 0),
        Some("conv"),
    )?;

    input.set_value(Some(&Tensor::ones(&[1, 1, 3, 3])))?;
    kernel.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])))?;
    conv.forward_recursive(1, false)?;

    let upstream = Tensor::ones(&[1, 1, 2, 2]);
    let grad = conv.calc_grad_to_parent_index(0, &upstream)?;

    assert_eq!(grad.shape(), &[1, 1, 3, 3]);

    // grad[0,0,0,0]: 只有 (oh=0,ow=0,kh=0,kw=0) 贡献 → upstream[0,0,0,0]*kernel[0,0,0,0] = 1*1 = 1
    // grad[0,0,0,1]: (oh=0,ow=0,kh=0,kw=1) + (oh=0,ow=1,kh=0,kw=0) → 1*2 + 1*1 = 3
    // grad[0,0,1,1]: 所有 4 个输出位置都贡献 → 1*1 + 1*2 + 1*3 + 1*4 = 10
    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 0, 1]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 1, 1]], 10.0, epsilon = 1e-6);

    Ok(())
}

// ==================== 3. 端到端反向传播测试（底层构建完整图）====================

/// 测试 Conv2d E2E 反向传播（kernel 梯度验证）
///
/// 构建完整图：input → conv → reshape → mse_loss
/// 输入 [1, 1, 2, 2], 卷积核 [1, 1, 2, 2] → conv 输出 [1, 1, 1, 1] → reshape [1, 1] → loss
///
/// conv = 1*1 + 0*2 + 0*3 + 1*4 = 5, loss = (5-0)^2 = 25
/// d_loss/d_conv = 2*(5-0) = 10
/// d_loss/d_kernel[kh,kw] = 10 * input[kh,kw]
#[test]
fn test_conv2d_backward_kernel_grad() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_parameter_node(&[1, 1, 2, 2], Some("kernel"))?;

    let conv = inner.borrow_mut().create_conv2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (0, 0),
        Some("conv"),
    )?;

    // conv 输出 [1, 1, 1, 1] → reshape 为 [1, 1]
    let conv_flat = inner
        .borrow_mut()
        .create_reshape_node(conv.clone(), &[1, 1], Some("conv_flat"))?;

    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1], Some("target"))?;
    let loss = inner
        .borrow_mut()
        .create_mse_mean_node(conv_flat, target.clone(), Some("loss"))?;

    // 设值
    input.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])))?;
    kernel.set_value(Some(&Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])))?;
    target.set_value(Some(&Tensor::new(&[0.0], &[1, 1])))?;

    // 前向 + 反向
    inner.borrow_mut().forward_via_node_inner(&loss)?;

    // conv = 1*1 + 0*2 + 0*3 + 1*4 = 5, loss = 25
    let loss_val = loss.value().unwrap().get_data_number().unwrap();
    assert_abs_diff_eq!(loss_val, 25.0, epsilon = 1e-6);

    inner.borrow_mut().zero_grad()?;
    inner.borrow_mut().backward_via_node_inner(&loss)?;

    // 验证 kernel 梯度
    let grad = kernel.grad().expect("kernel 应有 grad");
    assert_eq!(grad.shape(), &[1, 1, 2, 2]);

    // d_loss/d_kernel = 2 * (conv - target) * input = 10 * input
    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 10.0, epsilon = 1e-5); // 10 * 1
    assert_abs_diff_eq!(grad[[0, 0, 0, 1]], 20.0, epsilon = 1e-5); // 10 * 2
    assert_abs_diff_eq!(grad[[0, 0, 1, 0]], 30.0, epsilon = 1e-5); // 10 * 3
    assert_abs_diff_eq!(grad[[0, 0, 1, 1]], 40.0, epsilon = 1e-5); // 10 * 4

    Ok(())
}

/// 测试 Conv2d batch 网络 E2E：conv → flatten → loss
///
/// 构建 Conv2d -> Flatten -> MSE 完整网络，验证 batch 前向和反向均可执行
#[test]
fn test_conv2d_backward_batch_network() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 输入: [2, 1, 4, 4], 卷积核: [1, 1, 2, 2]
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 1, 4, 4], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_parameter_node(&[1, 1, 2, 2], Some("kernel"))?;

    // conv 输出: [2, 1, 3, 3]
    let conv = inner.borrow_mut().create_conv2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (0, 0),
        Some("conv"),
    )?;

    // flatten: [2, 9]
    let flat = inner
        .borrow_mut()
        .create_flatten_node(conv.clone(), true, Some("flat"))?;

    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 9], Some("target"))?;
    let loss = inner
        .borrow_mut()
        .create_mse_mean_node(flat.clone(), target.clone(), Some("loss"))?;

    // batch 0 全 1, batch 1 全 2
    let mut input_data = vec![1.0f32; 16];
    input_data.extend(vec![2.0f32; 16]);
    input.set_value(Some(&Tensor::new(&input_data, &[2, 1, 4, 4])))?;
    kernel.set_value(Some(&Tensor::ones(&[1, 1, 2, 2])))?;
    target.set_value(Some(&Tensor::zeros(&[2, 9])))?;

    // 前向传播
    inner.borrow_mut().forward_via_node_inner(&loss)?;

    // 验证卷积输出
    let conv_output = conv.value().unwrap();
    assert_eq!(conv_output.shape(), &[2, 1, 3, 3]);
    assert_abs_diff_eq!(conv_output[[0, 0, 0, 0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(conv_output[[1, 0, 0, 0]], 8.0, epsilon = 1e-6);

    // 验证 flatten 输出
    let flat_output = flat.value().unwrap();
    assert_eq!(flat_output.shape(), &[2, 9]);
    assert_abs_diff_eq!(flat_output[[0, 0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(flat_output[[1, 0]], 8.0, epsilon = 1e-6);

    // loss >= 0
    let loss_val = loss.value().unwrap().get_data_number().unwrap();
    assert!(loss_val >= 0.0);

    // 反向传播
    inner.borrow_mut().zero_grad()?;
    inner.borrow_mut().backward_via_node_inner(&loss)?;

    // kernel 应有梯度，形状正确
    let grad = kernel.grad().expect("kernel 应有 grad");
    assert_eq!(grad.shape(), &[1, 1, 2, 2]);

    Ok(())
}

// ==================== 4. 梯度累积测试 ====================

/// 测试 Conv2d 梯度累积：clear_grad 后梯度一致 + 不清零则累积翻倍
///
/// 注意：inner_rc API 的 parameter 不会注册到 GraphInner 的参数表，
/// 因此 zero_grad() 不生效。需手动调用 kernel.clear_grad() 清零。
///
/// 使用与 test_conv2d_backward_kernel_grad 相同配置：
/// kernel=[1,0,0,1], input=[1,2,3,4] → grad = [10, 20, 30, 40]
#[test]
fn test_conv2d_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 2, 2], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_parameter_node(&[1, 1, 2, 2], Some("kernel"))?;

    let conv = inner.borrow_mut().create_conv2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (0, 0),
        Some("conv"),
    )?;

    let conv_flat = inner
        .borrow_mut()
        .create_reshape_node(conv, &[1, 1], Some("conv_flat"))?;
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1], Some("target"))?;
    let loss = inner
        .borrow_mut()
        .create_mse_mean_node(conv_flat, target.clone(), Some("loss"))?;

    input.set_value(Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])))?;
    kernel.set_value(Some(&Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])))?;
    target.set_value(Some(&Tensor::new(&[0.0], &[1, 1])))?;

    // 第一次 backward
    kernel.clear_grad()?;
    inner.borrow_mut().forward_via_node_inner(&loss)?;
    inner.borrow_mut().backward_via_node_inner(&loss)?;
    let grad_first = kernel.grad().unwrap().clone();

    // 验证已知值（同 test_conv2d_backward_kernel_grad）
    assert_abs_diff_eq!(grad_first[[0, 0, 0, 0]], 10.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad_first[[0, 0, 0, 1]], 20.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad_first[[0, 0, 1, 0]], 30.0, epsilon = 1e-5);
    assert_abs_diff_eq!(grad_first[[0, 0, 1, 1]], 40.0, epsilon = 1e-5);

    // clear_grad 后再 backward → 恢复单次值（验证可重复性）
    kernel.clear_grad()?;
    assert!(kernel.grad().is_none(), "clear_grad 后 kernel 梯度应为 None");
    inner.borrow_mut().forward_via_node_inner(&loss)?;
    inner.borrow_mut().backward_via_node_inner(&loss)?;
    let grad_second = kernel.grad().unwrap();
    assert_eq!(&grad_second, &grad_first, "clear_grad 后重新 backward 应得到相同梯度");

    // 第三次 backward（不清零）→ 梯度翻倍
    inner.borrow_mut().forward_via_node_inner(&loss)?;
    inner.borrow_mut().backward_via_node_inner(&loss)?;
    let grad_accumulated = kernel.grad().unwrap();
    assert_eq!(&grad_accumulated, &(&grad_first * 2.0));

    // clear_grad 后恢复单次值
    kernel.clear_grad()?;
    inner.borrow_mut().forward_via_node_inner(&loss)?;
    inner.borrow_mut().backward_via_node_inner(&loss)?;
    let grad_after_clear = kernel.grad().unwrap();
    assert_eq!(&grad_after_clear, &grad_first);

    Ok(())
}

// ==================== 5. 动态形状 + 动态 batch 测试 ====================

/// 测试 Conv2d 动态形状传播
#[test]
fn test_conv2d_dynamic_shape_propagation() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 输入 [2, 1, 5, 5], 卷积核 [2, 1, 3, 3]
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 1, 5, 5], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_parameter_node(&[2, 1, 3, 3], Some("kernel"))?;

    let conv = inner.borrow_mut().create_conv2d_node(
        vec![input, kernel],
        (1, 1),
        (0, 0),
        Some("conv"),
    )?;

    let dyn_shape = conv.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "out_channels 维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(2), "out_channels 应该是 2");

    Ok(())
}

/// 测试 Conv2d 动态 batch 前向
#[test]
fn test_conv2d_dynamic_batch_forward() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 1, 5, 5], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_parameter_node(&[2, 1, 3, 3], Some("kernel"))?;

    let conv = inner.borrow_mut().create_conv2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (0, 0),
        Some("conv"),
    )?;

    // 第一次 forward：batch=2
    input.set_value(Some(&Tensor::zeros(&[2, 1, 5, 5])))?;
    kernel.set_value(Some(&Tensor::ones(&[2, 1, 3, 3])))?;
    conv.forward_recursive(1, false)?;
    let value1 = conv.value().unwrap();
    assert_eq!(value1.shape(), &[2, 2, 3, 3], "第一次 forward: batch=2");

    // 第二次 forward：batch=5（动态改变 batch）
    input.set_value(Some(&Tensor::zeros(&[5, 1, 5, 5])))?;
    conv.forward_recursive(2, false)?;
    let value2 = conv.value().unwrap();
    assert_eq!(value2.shape(), &[5, 2, 3, 3], "第二次 forward: batch=5");

    Ok(())
}

/// 测试 Conv2d 动态 batch 反向传播
///
/// 构建 Conv2d -> Flatten -> MSE，先 batch=2 训练，再 batch=4 训练
#[test]
fn test_conv2d_dynamic_batch_backward() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 1, 5, 5], Some("input"))?;
    let kernel = inner
        .borrow_mut()
        .create_parameter_node(&[2, 1, 3, 3], Some("kernel"))?;

    // Conv2d 输出: [batch, 2, 3, 3]
    let conv = inner.borrow_mut().create_conv2d_node(
        vec![input.clone(), kernel.clone()],
        (1, 1),
        (0, 0),
        Some("conv"),
    )?;

    // Flatten: [batch, 18]
    let flat = inner
        .borrow_mut()
        .create_flatten_node(conv, true, Some("flat"))?;

    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 18], Some("target"))?;
    let loss = inner
        .borrow_mut()
        .create_mse_mean_node(flat, target.clone(), Some("loss"))?;

    // 第一次训练：batch=2
    input.set_value(Some(&Tensor::normal_seeded(0.0, 1.0, &[2, 1, 5, 5], 42)))?;
    kernel.set_value(Some(&Tensor::ones(&[2, 1, 3, 3])))?;
    target.set_value(Some(&Tensor::zeros(&[2, 18])))?;

    inner.borrow_mut().forward_via_node_inner(&loss)?;
    let loss_val1 = loss.value().unwrap().get_data_number().unwrap();
    assert!(loss_val1 >= 0.0);
    inner.borrow_mut().zero_grad()?;
    inner.borrow_mut().backward_via_node_inner(&loss)?;

    // 更新输入为不同 batch_size
    input.set_value(Some(&Tensor::normal_seeded(0.0, 1.0, &[4, 1, 5, 5], 100)))?;
    target.set_value(Some(&Tensor::zeros(&[4, 18])))?;

    // 第二次训练：batch=4
    inner.borrow_mut().forward_via_node_inner(&loss)?;
    let loss_val2 = loss.value().unwrap().get_data_number().unwrap();
    assert!(loss_val2 >= 0.0);
    inner.borrow_mut().zero_grad()?;
    inner.borrow_mut().backward_via_node_inner(&loss)?;

    // kernel 梯度形状不随 batch 变化
    let grad = kernel.grad().expect("kernel 应有 grad");
    assert_eq!(grad.shape(), &[2, 1, 3, 3]);

    Ok(())
}

// ==================== 6. Create API 测试（保持原样）====================

#[test]
fn test_create_conv2d_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 输入: [2, 3, 8, 8], 卷积核: [16, 3, 3, 3]
    // 输出: [2, 16, 6, 6]（stride=1, padding=0）
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3, 8, 8], Some("input"))
        .unwrap();
    let kernel = inner
        .borrow_mut()
        .create_parameter_node(&[16, 3, 3, 3], Some("kernel"))
        .unwrap();

    let conv = inner
        .borrow_mut()
        .create_conv2d_node(vec![input.clone(), kernel.clone()], (1, 1), (0, 0), Some("conv"))
        .unwrap();

    assert_eq!(conv.shape(), vec![2, 16, 6, 6]);
    assert_eq!(conv.name(), Some("conv"));
    assert!(!conv.is_leaf());
    assert_eq!(conv.parents().len(), 2);
}

#[test]
fn test_create_conv2d_with_padding() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 输入: [1, 1, 5, 5], 卷积核: [2, 1, 3, 3]
    // padding=1 -> 输出: [1, 2, 5, 5]
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 5, 5], None)
        .unwrap();
    let kernel = inner
        .borrow_mut()
        .create_parameter_node(&[2, 1, 3, 3], None)
        .unwrap();

    let conv = inner
        .borrow_mut()
        .create_conv2d_node(vec![input, kernel], (1, 1), (1, 1), None)
        .unwrap();

    assert_eq!(conv.shape(), vec![1, 2, 5, 5]);
}

#[test]
fn test_create_conv2d_with_stride() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 输入: [1, 1, 8, 8], 卷积核: [4, 1, 3, 3]
    // stride=2 -> 输出: [1, 4, 3, 3]
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 1, 8, 8], None)
        .unwrap();
    let kernel = inner
        .borrow_mut()
        .create_parameter_node(&[4, 1, 3, 3], None)
        .unwrap();

    let conv = inner
        .borrow_mut()
        .create_conv2d_node(vec![input, kernel], (2, 2), (0, 0), None)
        .unwrap();

    assert_eq!(conv.shape(), vec![1, 4, 3, 3]);
}

#[test]
fn test_create_conv2d_channel_mismatch() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // 输入通道 3，卷积核输入通道 4 -> 应该失败
    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3, 8, 8], None)
        .unwrap();
    let kernel = inner
        .borrow_mut()
        .create_parameter_node(&[16, 4, 3, 3], None) // 4 != 3
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_conv2d_node(vec![input, kernel], (1, 1), (0, 0), None);
    assert!(result.is_err());
}

#[test]
fn test_create_conv2d_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_conv;
    let weak_input;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[1, 3, 8, 8], None)
            .unwrap();
        let kernel = inner
            .borrow_mut()
            .create_parameter_node(&[16, 3, 3, 3], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let conv = inner
            .borrow_mut()
            .create_conv2d_node(vec![input, kernel], (1, 1), (0, 0), None)
            .unwrap();
        weak_conv = Rc::downgrade(&conv);

        assert!(weak_conv.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
    }
    assert!(weak_conv.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
}
