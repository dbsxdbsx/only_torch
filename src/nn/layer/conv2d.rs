/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : Conv2d (2D 卷积) 层
 *
 * 设计决策：Layer 层只提供 Batch 版本（Batch-First 设计）
 * - 输入格式：[batch_size, in_channels, H, W]
 * - 输出格式：[batch_size, out_channels, H', W']
 * - 单样本需求可直接用 Node API
 *
 * 等价于 PyTorch 的 nn.Conv2d（天然支持 batch）
 *
 * Bias 设计：
 * - 默认包含 bias，初始化为 0（数学上等价于无 bias 的初始状态）
 * - bias 形状 [1, out_channels]，通过 ChannelBiasAdd 节点广播到 [batch, C, H, W]
 * - 用户可通过 Conv2dOutput.bias 访问并修改 bias 参数
 */

use crate::nn::{Graph, GraphError, NodeId};
use crate::tensor::Tensor;

/// Conv2d 层的输出结构
///
/// 暴露所有内部节点，便于：
/// - 访问卷积核和 bias 进行初始化或检查
/// - NEAT 进化时操作内部参数
/// - 调试和可视化
#[derive(Debug, Clone, Copy)]
pub struct Conv2dOutput {
    /// 输出节点 ID (最终计算结果，含 bias) [batch, out_channels, H', W']
    pub output: NodeId,
    /// 卷积核参数节点 ID [out_channels, in_channels, kernel_h, kernel_w]
    pub kernel: NodeId,
    /// 偏置参数节点 ID [1, out_channels]，初始化为 0
    pub bias: NodeId,
}

/// 创建 Conv2d (2D 卷积) 层
///
/// # 设计
/// - **Batch-First**：输入 `[batch, C_in, H, W]`，输出 `[batch, C_out, H', W']`
/// - 符合 PyTorch `nn.Conv2d` 语义（默认包含 bias，初始化为 0）
///
/// # 参数
/// - `graph`: 计算图
/// - `input`: 输入节点 ID，形状 `[batch_size, in_channels, H, W]`
/// - `in_channels`: 输入通道数
/// - `out_channels`: 输出通道数
/// - `kernel_size`: 卷积核大小 (kH, kW)
/// - `stride`: 步长 (sH, sW)
/// - `padding`: 填充 (pH, pW)
/// - `name`: 层名称前缀（可选）
///
/// # 返回
/// - `Conv2dOutput` 包含 output、kernel、bias 节点 ID
///
/// # 内部结构
/// ```text
/// input [batch, C_in, H, W]
///       │
///       ▼
/// ┌─────────────────────┐
/// │ Conv2d(input, K)    │◄── K [C_out, C_in, kH, kW]
/// └─────────────────────┘
///       │
///       ▼
/// ┌─────────────────────────────┐
/// │ ChannelBiasAdd(conv, bias)  │◄── bias [1, C_out]
/// └─────────────────────────────┘
///       │
///       ▼
/// output [batch, C_out, H', W']
/// ```
///
/// # 输出尺寸计算
/// ```text
/// H' = (H + 2*padding_h - kernel_h) / stride_h + 1
/// W' = (W + 2*padding_w - kernel_w) / stride_w + 1
/// ```
///
/// # 示例
/// ```ignore
/// // 构建简单 CNN
/// let conv1 = conv2d(&mut graph, x, 1, 32, (3, 3), (1, 1), (1, 1), Some("conv1"))?;
/// let act1 = graph.new_relu_node(conv1.output, Some("act1"))?;
/// let pool1 = max_pool2d(&mut graph, act1, (2, 2), (2, 2), Some("pool1"))?;
///
/// // 访问参数
/// let kernel = graph.get_node_value(conv1.kernel)?;
/// let bias = graph.get_node_value(conv1.bias)?;
///
/// // 自定义 bias 初始化
/// graph.set_node_value(conv1.bias, Some(&Tensor::new(&[0.1, 0.2, ...], &[1, 32])))?;
/// ```
pub fn conv2d(
    graph: &mut Graph,
    input: NodeId,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    name: Option<&str>,
) -> Result<Conv2dOutput, GraphError> {
    let prefix = name.unwrap_or("conv2d");

    // 创建卷积核参数 K: [out_channels, in_channels, kH, kW]
    let (k_h, k_w) = kernel_size;
    let kernel = graph.new_parameter_node(
        &[out_channels, in_channels, k_h, k_w],
        Some(&format!("{}_K", prefix)),
    )?;

    // 创建 Conv2d 节点（无 bias）
    let conv_out = graph.new_conv2d_node(
        input,
        kernel,
        stride,
        padding,
        Some(&format!("{}_conv", prefix)),
    )?;

    // 创建 bias 参数 [1, out_channels]，初始化为 0
    let bias = graph.new_parameter_node(&[1, out_channels], Some(&format!("{}_b", prefix)))?;
    graph.set_node_value(bias, Some(&Tensor::zeros(&[1, out_channels])))?;

    // 通过 ChannelBiasAdd 节点添加 bias
    let output =
        graph.new_channel_bias_add_node(conv_out, bias, Some(&format!("{}_out", prefix)))?;

    // 注册层分组（用于可视化时将这些节点框在一起）
    graph.register_layer_group(
        prefix,
        "Conv2d",
        &format!("{}→{}, {}×{}", in_channels, out_channels, k_h, k_w),
        vec![kernel, bias, conv_out, output],
    );

    Ok(Conv2dOutput {
        output,
        kernel,
        bias,
    })
}
