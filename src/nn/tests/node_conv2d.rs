/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : Conv2d 节点单元测试
 *
 * 测试策略：
 * 1. 基础功能测试（形状、前向传播）
 * 2. Jacobi 模式反向传播（单样本）
 * 3. Batch 模式反向传播
 * 4. 各种参数组合（stride, padding）
 */

use crate::assert_err;
use crate::nn::{GraphError, GraphInner};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试 Conv2d 节点创建（单样本）
#[test]
fn test_conv2d_creation_single() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 输入: [C_in=1, H=5, W=5]
    let input = graph.new_basic_input_node(&[1, 5, 5], Some("input"))?;
    // 卷积核: [C_out=2, C_in=1, kH=3, kW=3]
    let kernel = graph.new_parameter_node(&[2, 1, 3, 3], Some("kernel"))?;

    // 创建 Conv2d: stride=1, padding=0
    let conv = graph.new_conv2d_node(input, kernel, (1, 1), (0, 0), Some("conv"))?;

    // 验证输出形状: [C_out=2, H'=3, W'=3]
    // H' = (5 + 0 - 3) / 1 + 1 = 3
    let output_shape = graph.get_node(conv)?.value_expected_shape();
    assert_eq!(output_shape, &[2, 3, 3]);

    Ok(())
}

/// 测试 Conv2d 节点创建（Batch）
#[test]
fn test_conv2d_creation_batch() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 输入: [batch=4, C_in=3, H=28, W=28]
    let input = graph.new_basic_input_node(&[4, 3, 28, 28], Some("input"))?;
    // 卷积核: [C_out=16, C_in=3, kH=5, kW=5]
    let kernel = graph.new_parameter_node(&[16, 3, 5, 5], Some("kernel"))?;

    // 创建 Conv2d: stride=1, padding=2
    let conv = graph.new_conv2d_node(input, kernel, (1, 1), (2, 2), Some("conv"))?;

    // 验证输出形状: [batch=4, C_out=16, H'=28, W'=28]
    // H' = (28 + 4 - 5) / 1 + 1 = 28（same padding）
    let output_shape = graph.get_node(conv)?.value_expected_shape();
    assert_eq!(output_shape, &[4, 16, 28, 28]);

    Ok(())
}

/// 测试 Conv2d 带 stride
#[test]
fn test_conv2d_with_stride() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 输入: [batch=2, C_in=1, H=8, W=8]
    let input = graph.new_basic_input_node(&[2, 1, 8, 8], Some("input"))?;
    // 卷积核: [C_out=4, C_in=1, kH=3, kW=3]
    let kernel = graph.new_parameter_node(&[4, 1, 3, 3], Some("kernel"))?;

    // 创建 Conv2d: stride=2, padding=0
    let conv = graph.new_conv2d_node(input, kernel, (2, 2), (0, 0), Some("conv"))?;

    // 验证输出形状: [batch=2, C_out=4, H'=3, W'=3]
    // H' = (8 + 0 - 3) / 2 + 1 = 3
    let output_shape = graph.get_node(conv)?.value_expected_shape();
    assert_eq!(output_shape, &[2, 4, 3, 3]);

    Ok(())
}

// ==================== 前向传播测试 ====================

/// 测试 Conv2d 前向传播（简单情况）
#[test]
fn test_conv2d_forward_simple() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 输入: [C_in=1, H=3, W=3]，全 1
    let input = graph.new_basic_input_node(&[1, 3, 3], Some("input"))?;
    // 卷积核: [C_out=1, C_in=1, kH=2, kW=2]，全 1
    let kernel = graph.new_basic_input_node(&[1, 1, 2, 2], Some("kernel"))?;

    let conv = graph.new_conv2d_node(input, kernel, (1, 1), (0, 0), Some("conv"))?;

    // 设置值
    let input_val = Tensor::ones(&[1, 3, 3]);
    let kernel_val = Tensor::ones(&[1, 1, 2, 2]);

    graph.set_node_value(input, Some(&input_val))?;
    graph.set_node_value(kernel, Some(&kernel_val))?;

    // 前向传播
    graph.forward(conv)?;

    // 验证输出: 2x2 窗口求和 = 4.0（每个位置）
    let output = graph.get_node_value(conv)?.unwrap();
    assert_eq!(output.shape(), &[1, 2, 2]);
    assert_abs_diff_eq!(output[[0, 0, 0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1, 0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1, 1]], 4.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Conv2d 前向传播（带 padding）
#[test]
fn test_conv2d_forward_with_padding() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 输入: [C_in=1, H=3, W=3]
    let input = graph.new_basic_input_node(&[1, 3, 3], Some("input"))?;
    // 卷积核: [C_out=1, C_in=1, kH=3, kW=3]
    let kernel = graph.new_basic_input_node(&[1, 1, 3, 3], Some("kernel"))?;

    // padding=1 保持尺寸
    let conv = graph.new_conv2d_node(input, kernel, (1, 1), (1, 1), Some("conv"))?;

    // 输入全 1
    let input_val = Tensor::ones(&[1, 3, 3]);
    // 卷积核全 1
    let kernel_val = Tensor::ones(&[1, 1, 3, 3]);

    graph.set_node_value(input, Some(&input_val))?;
    graph.set_node_value(kernel, Some(&kernel_val))?;

    graph.forward(conv)?;

    // 验证输出形状: [1, 3, 3]（same padding）
    let output = graph.get_node_value(conv)?.unwrap();
    assert_eq!(output.shape(), &[1, 3, 3]);

    // 中心位置：3x3 窗口全部有值，sum = 9
    assert_abs_diff_eq!(output[[0, 1, 1]], 9.0, epsilon = 1e-6);
    // 角落位置：只有 2x2 区域有值（其余被 padding 的 0 填充），sum = 4
    assert_abs_diff_eq!(output[[0, 0, 0]], 4.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Conv2d 前向传播（Batch 模式）
#[test]
fn test_conv2d_forward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 输入: [batch=2, C_in=1, H=4, W=4]
    let input = graph.new_basic_input_node(&[2, 1, 4, 4], Some("input"))?;
    // 卷积核: [C_out=1, C_in=1, kH=2, kW=2]
    let kernel = graph.new_basic_input_node(&[1, 1, 2, 2], Some("kernel"))?;

    let conv = graph.new_conv2d_node(input, kernel, (1, 1), (0, 0), Some("conv"))?;

    // 设置值：第一个 batch 全 1，第二个 batch 全 2
    let mut input_data = vec![1.0f32; 16];
    input_data.extend(vec![2.0f32; 16]);
    let input_val = Tensor::new(&input_data, &[2, 1, 4, 4]);
    let kernel_val = Tensor::ones(&[1, 1, 2, 2]);

    graph.set_node_value(input, Some(&input_val))?;
    graph.set_node_value(kernel, Some(&kernel_val))?;

    graph.forward(conv)?;

    let output = graph.get_node_value(conv)?.unwrap();
    assert_eq!(output.shape(), &[2, 1, 3, 3]);

    // 第一个 batch: sum = 4
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 4.0, epsilon = 1e-6);
    // 第二个 batch: sum = 8
    assert_abs_diff_eq!(output[[1, 0, 0, 0]], 8.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Conv2d 多输出通道
#[test]
fn test_conv2d_multi_output_channels() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 输入: [C_in=1, H=4, W=4]
    let input = graph.new_basic_input_node(&[1, 4, 4], Some("input"))?;
    // 卷积核: [C_out=2, C_in=1, kH=2, kW=2]
    let kernel = graph.new_basic_input_node(&[2, 1, 2, 2], Some("kernel"))?;

    let conv = graph.new_conv2d_node(input, kernel, (1, 1), (0, 0), Some("conv"))?;

    // 输入全 1
    let input_val = Tensor::ones(&[1, 4, 4]);
    // 卷积核：第一个全 1，第二个全 2
    let mut kernel_data = vec![1.0f32; 4];
    kernel_data.extend(vec![2.0f32; 4]);
    let kernel_val = Tensor::new(&kernel_data, &[2, 1, 2, 2]);

    graph.set_node_value(input, Some(&input_val))?;
    graph.set_node_value(kernel, Some(&kernel_val))?;

    graph.forward(conv)?;

    let output = graph.get_node_value(conv)?.unwrap();
    assert_eq!(output.shape(), &[2, 3, 3]);

    // 第一个输出通道: sum = 4
    assert_abs_diff_eq!(output[[0, 0, 0]], 4.0, epsilon = 1e-6);
    // 第二个输出通道: sum = 8
    assert_abs_diff_eq!(output[[1, 0, 0]], 8.0, epsilon = 1e-6);

    Ok(())
}

// ==================== VJP 模式测试（单样本）====================

/// 测试 Conv2d 对卷积核的梯度（VJP 模式）
///
/// 构建完整计算图：input -> conv -> flatten -> mse_loss
/// 验证 kernel 的梯度正确性
#[test]
fn test_conv2d_jacobi_to_kernel() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 简单情况：输入 [1, 2, 2]，卷积核 [1, 1, 2, 2]
    // 输出形状：[1, 1, 1]（out_channels=1, H=1, W=1）
    let input = graph.new_basic_input_node(&[1, 2, 2], Some("input"))?;
    let kernel = graph.new_parameter_node(&[1, 1, 2, 2], Some("kernel"))?;

    let conv = graph.new_conv2d_node(input, kernel, (1, 1), (0, 0), Some("conv"))?;

    // 将 conv 输出 [1, 1, 1] reshape 成 [1, 1] 以便与 target 匹配
    let conv_flat = graph.new_reshape_node(conv, &[1, 1], Some("conv_flat"))?;

    // 添加 MSE loss 使输出为标量 [1, 1]
    let target = graph.new_basic_input_node(&[1, 1], Some("target"))?;
    let loss = graph.new_mse_loss_node(conv_flat, target, Some("loss"))?;

    // 设置值
    let input_val = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let kernel_val = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
    let target_val = Tensor::new(&[0.0], &[1, 1]);

    graph.set_node_value(input, Some(&input_val))?;
    graph.set_node_value(kernel, Some(&kernel_val))?;
    graph.set_node_value(target, Some(&target_val))?;

    // 前向传播
    graph.forward(loss)?;

    // conv 输出 = 1*1 + 0*2 + 0*3 + 1*4 = 5
    // loss = (5 - 0)^2 = 25
    let loss_val = graph
        .get_node_value(loss)?
        .unwrap()
        .get_data_number()
        .unwrap();
    assert_abs_diff_eq!(loss_val, 25.0, epsilon = 1e-6);

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // 验证 kernel 的梯度
    let grad = graph.get_node(kernel)?.grad().expect("kernel 应有 grad");
    assert_eq!(grad.shape(), &[1, 1, 2, 2]);

    // d_loss/d_conv = 2 * (conv - target) = 2 * 5 = 10
    // d_conv/d_kernel[i,j] = input[i,j]
    // d_loss/d_kernel = 10 * input
    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 10.0, epsilon = 1e-5); // 10 * input[0,0]
    assert_abs_diff_eq!(grad[[0, 0, 0, 1]], 20.0, epsilon = 1e-5); // 10 * input[0,1]
    assert_abs_diff_eq!(grad[[0, 0, 1, 0]], 30.0, epsilon = 1e-5); // 10 * input[1,0]
    assert_abs_diff_eq!(grad[[0, 0, 1, 1]], 40.0, epsilon = 1e-5); // 10 * input[1,1]

    Ok(())
}

/// 测试 Conv2d calc_grad_to_parent 直接调用（对输入）
///
/// 直接调用节点的 calc_grad_to_parent 方法测试对输入的 grad
#[test]
fn test_conv2d_grad_to_input() -> Result<(), GraphError> {
    use crate::tensor::Tensor;
    let mut graph = GraphInner::new();

    // 输入 [1, 2, 2]，卷积核 [1, 1, 2, 2]
    let input_id = graph.new_basic_input_node(&[1, 2, 2], Some("input"))?;
    let kernel_id = graph.new_basic_input_node(&[1, 1, 2, 2], Some("kernel"))?;

    let conv_id = graph.new_conv2d_node(input_id, kernel_id, (1, 1), (0, 0), Some("conv"))?;

    // 设置值
    let input_val = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let kernel_val = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);

    graph.set_node_value(input_id, Some(&input_val))?;
    graph.set_node_value(kernel_id, Some(&kernel_val))?;

    graph.forward(conv_id)?;

    // 直接调用 calc_grad_to_parent（VJP 模式）
    let conv_node = graph.get_node(conv_id)?;
    let input_node = graph.get_node(input_id)?;
    let kernel_node = graph.get_node(kernel_id)?;

    // upstream_grad 形状与 conv 输出一致：[1, 1, 1]
    let upstream_grad = Tensor::ones(&[1, 1, 1]);
    let grad = conv_node.calc_grad_to_parent(input_node, &upstream_grad, Some(kernel_node))?;

    // VJP 模式下验证 grad 形状与 input 值一致：[1, 2, 2]
    assert_eq!(grad.shape(), &[1, 2, 2]);

    // grad 值应该等于卷积核值（因为 ∂out/∂input[i,j] = kernel[i,j]）
    assert_abs_diff_eq!(grad[[0, 0, 0]], 1.0, epsilon = 1e-6); // kernel[0,0]
    assert_abs_diff_eq!(grad[[0, 0, 1]], 2.0, epsilon = 1e-6); // kernel[0,1]
    assert_abs_diff_eq!(grad[[0, 1, 0]], 3.0, epsilon = 1e-6); // kernel[1,0]
    assert_abs_diff_eq!(grad[[0, 1, 1]], 4.0, epsilon = 1e-6); // kernel[1,1]

    Ok(())
}

// ==================== Batch 模式测试（通过完整网络）====================

/// 测试 Conv2d 在完整网络中的前向传播
///
/// 通过构建 Conv2d -> Flatten 的网络来测试
#[test]
fn test_conv2d_batch_in_network() -> Result<(), GraphError> {
    let mut graph = GraphInner::new_with_seed(42);

    // 输入: [batch=2, C_in=1, H=4, W=4]
    let input = graph.new_basic_input_node(&[2, 1, 4, 4], Some("input"))?;
    // 卷积核: [C_out=1, C_in=1, kH=2, kW=2]
    let kernel = graph.new_parameter_node(&[1, 1, 2, 2], Some("kernel"))?;

    let conv = graph.new_conv2d_node(input, kernel, (1, 1), (0, 0), Some("conv"))?;
    // 输出形状: [2, 1, 3, 3]

    // 展平
    let flat = graph.new_flatten_node(conv, true, Some("flat"))?; // [2, 9]

    // 设置值
    let mut input_data = vec![1.0f32; 16];
    input_data.extend(vec![2.0f32; 16]);
    let input_val = Tensor::new(&input_data, &[2, 1, 4, 4]);
    let kernel_val = Tensor::ones(&[1, 1, 2, 2]);

    graph.set_node_value(input, Some(&input_val))?;
    graph.set_node_value(kernel, Some(&kernel_val))?;

    // 前向传播
    graph.forward(flat)?;

    // 验证卷积输出
    let conv_output = graph.get_node_value(conv)?.unwrap();
    assert_eq!(conv_output.shape(), &[2, 1, 3, 3]);

    // 第一个 batch: 2x2 窗口全 1，sum = 4
    assert_abs_diff_eq!(conv_output[[0, 0, 0, 0]], 4.0, epsilon = 1e-6);
    // 第二个 batch: 2x2 窗口全 2，sum = 8
    assert_abs_diff_eq!(conv_output[[1, 0, 0, 0]], 8.0, epsilon = 1e-6);

    // 验证展平输出
    let flat_output = graph.get_node_value(flat)?.unwrap();
    assert_eq!(flat_output.shape(), &[2, 9]);
    // 第一个 batch 所有值都是 4.0
    assert_abs_diff_eq!(flat_output[[0, 0]], 4.0, epsilon = 1e-6);
    // 第二个 batch 所有值都是 8.0
    assert_abs_diff_eq!(flat_output[[1, 0]], 8.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Conv2d calc_grad_to_parent 直接调用（对卷积核）
#[test]
fn test_conv2d_calc_grad_to_kernel_direct() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 输入: [batch=2, C_in=1, H=3, W=3]
    let input_id = graph.new_basic_input_node(&[2, 1, 3, 3], Some("input"))?;
    // 卷积核: [C_out=1, C_in=1, kH=2, kW=2]
    let kernel_id = graph.new_parameter_node(&[1, 1, 2, 2], Some("kernel"))?;

    let conv_id = graph.new_conv2d_node(input_id, kernel_id, (1, 1), (0, 0), Some("conv"))?;

    // 设置输入：第一个 batch 全 1，第二个 batch 全 2
    let mut input_data = vec![1.0f32; 9];
    input_data.extend(vec![2.0f32; 9]);
    let input_val = Tensor::new(&input_data, &[2, 1, 3, 3]);
    let kernel_val = Tensor::ones(&[1, 1, 2, 2]);

    graph.set_node_value(input_id, Some(&input_val))?;
    graph.set_node_value(kernel_id, Some(&kernel_val))?;

    // 前向传播
    graph.forward(conv_id)?;

    // 直接调用 calc_grad_to_parent
    // upstream_grad 全 1，形状 [2, 1, 2, 2]
    let upstream_grad = Tensor::ones(&[2, 1, 2, 2]);

    let conv_node = graph.get_node(conv_id)?;
    let kernel_node = graph.get_node(kernel_id)?;
    let input_node = graph.get_node(input_id)?;

    // 直接调用 NodeHandle 上的 calc_grad_to_parent
    let grad = conv_node.calc_grad_to_parent(kernel_node, &upstream_grad, Some(input_node))?;

    // 验证梯度形状
    assert_eq!(grad.shape(), &[1, 1, 2, 2]);

    // 梯度值：sum over (batch, oh, ow) of upstream[b,oc,oh,ow] * input[b,ic,oh+kh,ow+kw]
    // upstream 全 1，所以 grad[kh,kw] = sum over (b,oh,ow) of input[b,ic,oh+kh,ow+kw]
    //
    // 对于 kernel[0,0,0,0] (kh=0, kw=0):
    //   batch 0: input[0,0,0,0] + input[0,0,0,1] + input[0,0,1,0] + input[0,0,1,1] = 1+1+1+1 = 4
    //   batch 1: input[1,0,0,0] + input[1,0,0,1] + input[1,0,1,0] + input[1,0,1,1] = 2+2+2+2 = 8
    //   total = 12
    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 12.0, epsilon = 1e-6);

    Ok(())
}

/// 测试 Conv2d calc_grad_to_parent 直接调用（对输入）
#[test]
fn test_conv2d_calc_grad_to_input_direct() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 输入: [batch=1, C_in=1, H=3, W=3]
    let input_id = graph.new_basic_input_node(&[1, 1, 3, 3], Some("input"))?;
    // 卷积核: [C_out=1, C_in=1, kH=2, kW=2]
    let kernel_id = graph.new_basic_input_node(&[1, 1, 2, 2], Some("kernel"))?;

    let conv_id = graph.new_conv2d_node(input_id, kernel_id, (1, 1), (0, 0), Some("conv"))?;

    let input_val = Tensor::ones(&[1, 1, 3, 3]);
    let kernel_val = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);

    graph.set_node_value(input_id, Some(&input_val))?;
    graph.set_node_value(kernel_id, Some(&kernel_val))?;

    graph.forward(conv_id)?;

    // upstream_grad 全 1
    let upstream_grad = Tensor::ones(&[1, 1, 2, 2]);

    let conv_node = graph.get_node(conv_id)?;
    let input_node = graph.get_node(input_id)?;
    let kernel_node = graph.get_node(kernel_id)?;

    // 直接调用 NodeHandle 上的 calc_grad_to_parent
    let grad = conv_node.calc_grad_to_parent(input_node, &upstream_grad, Some(kernel_node))?;

    assert_eq!(grad.shape(), &[1, 1, 3, 3]);

    // 验证梯度：反卷积效果
    // grad[b,c,h,w] = sum over (oc,kh,kw) of upstream[b,oc,h-kh,w-kw] * kernel[oc,c,kh,kw]
    //                 (仅当 h-kh 和 w-kw 在有效范围内)
    //
    // grad[0,0,0,0]: 只有 (oh=0,ow=0,kh=0,kw=0) 贡献，= upstream[0,0,0,0] * kernel[0,0,0,0] = 1*1 = 1
    // grad[0,0,0,1]: (oh=0,ow=0,kh=0,kw=1) + (oh=0,ow=1,kh=0,kw=0) = 1*2 + 1*1 = 3
    // grad[0,0,1,1]: 所有 4 个输出位置都贡献，= 1*1 + 1*2 + 1*3 + 1*4 = 10
    assert_abs_diff_eq!(grad[[0, 0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 0, 1]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(grad[[0, 0, 1, 1]], 10.0, epsilon = 1e-6);

    Ok(())
}

// ==================== 错误处理测试 ====================

/// 测试通道数不匹配
#[test]
fn test_conv2d_channel_mismatch() {
    let mut graph = GraphInner::new();

    // 输入: [C_in=3, H=5, W=5]
    let input = graph
        .new_basic_input_node(&[3, 5, 5], Some("input"))
        .unwrap();
    // 卷积核: [C_out=2, C_in=1, kH=3, kW=3]（C_in 不匹配）
    let kernel = graph
        .new_parameter_node(&[2, 1, 3, 3], Some("kernel"))
        .unwrap();

    let result = graph.new_conv2d_node(input, kernel, (1, 1), (0, 0), Some("conv"));
    assert_err!(result, GraphError::ShapeMismatch { message, .. } if message.contains("通道数"));
}

/// 测试无效的输入维度（2D 输入对于 Conv2d 无效）
#[test]
fn test_conv2d_invalid_input_dims() {
    let mut graph = GraphInner::new();

    // 输入: [H=5, W=5]（2D，缺少通道维度，对 Conv2d 无效）
    let input = graph.new_basic_input_node(&[5, 5], Some("input")).unwrap();
    let kernel = graph
        .new_parameter_node(&[2, 1, 3, 3], Some("kernel"))
        .unwrap();

    let result = graph.new_conv2d_node(input, kernel, (1, 1), (0, 0), Some("conv"));
    assert_err!(result, GraphError::ShapeMismatch { message, .. }
        if message.contains("3D") || message.contains("4D"));
}

/// 测试无效的卷积核维度（2D 卷积核无效）
#[test]
fn test_conv2d_invalid_kernel_dims() {
    let mut graph = GraphInner::new();

    let input = graph
        .new_basic_input_node(&[1, 5, 5], Some("input"))
        .unwrap();
    // 卷积核: [kH=3, kW=3]（2D，缺少通道维度）
    let kernel = graph.new_parameter_node(&[3, 3], Some("kernel")).unwrap();

    let result = graph.new_conv2d_node(input, kernel, (1, 1), (0, 0), Some("conv"));
    assert_err!(result, GraphError::ShapeMismatch { message, .. }
        if message.contains("4D") || message.contains("C_out"));
}

// ==================== 动态形状测试 ====================

/// 测试 Conv2d 节点的动态形状传播
#[test]
fn test_conv2d_dynamic_shape_propagation() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建 4D 输入：[batch, channels, height, width]
    // Input 节点默认支持动态 batch
    let input = graph.new_basic_input_node(&[2, 1, 5, 5], Some("input"))?;
    let kernel = graph.new_parameter_node(&[2, 1, 3, 3], Some("kernel"))?;

    // Conv2d: [batch, 1, 5, 5] -> [batch, 2, 3, 3]
    let conv = graph.new_conv2d_node(input, kernel, (1, 1), (0, 0), Some("conv"))?;

    // 验证动态形状传播
    let conv_node = graph.get_node(conv)?;
    let dyn_shape = conv_node.dynamic_expected_shape();
    assert!(dyn_shape.is_dynamic(0), "batch 维度应该是动态的");
    assert!(!dyn_shape.is_dynamic(1), "out_channels 维度应该是固定的");
    assert_eq!(dyn_shape.dim(1), Some(2), "out_channels 应该是 2");

    Ok(())
}

/// 测试 Conv2d 在不同 batch_size 下的前向计算
#[test]
fn test_conv2d_dynamic_batch_forward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建 4D 输入：[batch, channels, height, width]
    let input = graph.new_basic_input_node(&[2, 1, 5, 5], Some("input"))?;
    let kernel = graph.new_parameter_node(&[2, 1, 3, 3], Some("kernel"))?;

    // Conv2d
    let conv = graph.new_conv2d_node(input, kernel, (1, 1), (0, 0), Some("conv"))?;

    // 设置初始值
    graph.set_node_value(input, Some(&Tensor::zeros(&[2, 1, 5, 5])))?;
    graph.set_node_value(kernel, Some(&Tensor::ones(&[2, 1, 3, 3])))?;

    // 第一次 forward：batch=2
    graph.forward(conv)?;
    let value1 = graph.get_node_value(conv)?.unwrap();
    assert_eq!(value1.shape(), &[2, 2, 3, 3], "第一次 forward: batch=2");

    // 更新输入为不同的 batch_size
    graph.set_node_value(input, Some(&Tensor::zeros(&[5, 1, 5, 5])))?;

    // 第二次 forward：batch=5
    graph.forward(conv)?;
    let value2 = graph.get_node_value(conv)?.unwrap();
    assert_eq!(value2.shape(), &[5, 2, 3, 3], "第二次 forward: batch=5");

    Ok(())
}

/// 测试 Conv2d 在不同 batch_size 下的反向传播
#[test]
fn test_conv2d_dynamic_batch_backward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建 4D 输入
    let input = graph.new_basic_input_node(&[2, 1, 5, 5], Some("input"))?;
    let kernel = graph.new_parameter_node(&[2, 1, 3, 3], Some("kernel"))?;

    // Conv2d -> Flatten -> MSE
    let conv = graph.new_conv2d_node(input, kernel, (1, 1), (0, 0), Some("conv"))?;
    // 输出形状: [batch, 2, 3, 3]
    let flat = graph.new_flatten_node(conv, true, Some("flat"))?;
    // 输出形状: [batch, 18]
    let target = graph.new_basic_input_node(&[2, 18], Some("target"))?;
    let loss = graph.new_mse_loss_node(flat, target, Some("loss"))?;

    // 设置初始值
    graph.set_node_value(
        input,
        Some(&Tensor::normal_seeded(0.0, 1.0, &[2, 1, 5, 5], 42)),
    )?;
    graph.set_node_value(kernel, Some(&Tensor::ones(&[2, 1, 3, 3])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 18])))?;

    // 第一次训练：batch=2
    graph.forward(loss)?;
    let loss_val1 = graph.get_node_value(loss)?.unwrap()[[0, 0]];
    assert!(loss_val1 >= 0.0);
    graph.zero_grad()?;
    graph.backward(loss)?;

    // 更新输入为不同的 batch_size
    graph.set_node_value(
        input,
        Some(&Tensor::normal_seeded(0.0, 1.0, &[4, 1, 5, 5], 100)),
    )?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[4, 18])))?;

    // 第二次训练：batch=4
    graph.forward(loss)?;
    let loss_val2 = graph.get_node_value(loss)?.unwrap()[[0, 0]];
    assert!(loss_val2 >= 0.0);
    graph.zero_grad()?;
    graph.backward(loss)?;

    Ok(())
}
