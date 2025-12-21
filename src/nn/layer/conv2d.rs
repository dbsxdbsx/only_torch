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
 */

use crate::nn::{Graph, GraphError, NodeId};

/// Conv2d 层的输出结构
///
/// 暴露所有内部节点，便于：
/// - 访问卷积核进行初始化或检查
/// - NEAT 进化时操作内部参数
/// - 调试和可视化
#[derive(Debug, Clone, Copy)]
pub struct Conv2dOutput {
    /// 输出节点 ID (最终计算结果) [batch, out_channels, H', W']
    pub output: NodeId,
    /// 卷积核参数节点 ID [out_channels, in_channels, kernel_h, kernel_w]
    pub kernel: NodeId,
}

/// 创建 Conv2d (2D 卷积) 层
///
/// # 设计
/// - **Batch-First**：输入 `[batch, C_in, H, W]`，输出 `[batch, C_out, H', W']`
/// - 符合 PyTorch `nn.Conv2d` 语义（不含 bias，需要 bias 可手动添加 Add 节点）
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
/// - `Conv2dOutput` 包含 output、kernel 节点 ID
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
/// // 访问卷积核参数
/// let kernel = graph.get_node_value(conv1.kernel)?;
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

    // 创建 Conv2d 节点
    let output = graph.new_conv2d_node(
        input,
        kernel,
        stride,
        padding,
        Some(&format!("{}_out", prefix)),
    )?;

    Ok(Conv2dOutput { output, kernel })
}
