/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : MaxPool2d (2D 最大池化) 层
 *
 * 设计决策：Layer 层只提供 Batch 版本（Batch-First 设计）
 * - 输入格式：[batch_size, channels, H, W]
 * - 输出格式：[batch_size, channels, H', W']
 * - 单样本需求可直接用 Node API
 *
 * 等价于 PyTorch 的 nn.MaxPool2d
 */

use crate::nn::{GraphError, GraphInner, NodeId};

/// `MaxPool2d` 层的输出结构
///
/// 池化层没有可学习参数，只返回输出节点
#[derive(Debug, Clone, Copy)]
pub struct MaxPool2dOutput {
    /// 输出节点 ID [batch, channels, H', W']
    pub output: NodeId,
}

/// 创建 `MaxPool2d` (2D 最大池化) 层
///
/// # 设计
/// - **Batch-First**：输入 `[batch, C, H, W]`，输出 `[batch, C, H', W']`
/// - 无可学习参数
/// - 符合 `PyTorch` `nn.MaxPool2d` 语义
///
/// # 参数
/// - `graph`: 计算图
/// - `input`: 输入节点 ID，形状 `[batch_size, channels, H, W]`
/// - `kernel_size`: 池化窗口大小 (kH, kW)
/// - `stride`: 步长 (sH, sW)，若为 None 则默认等于 `kernel_size`
/// - `name`: 层名称前缀（可选）
///
/// # 返回
/// - `MaxPool2dOutput` 包含 output 节点 ID
///
/// # 输出尺寸计算
/// ```text
/// H' = (H - kernel_h) / stride_h + 1
/// W' = (W - kernel_w) / stride_w + 1
/// ```
///
/// # 示例
/// ```ignore
/// // 典型用法：卷积后接池化
/// let conv1 = conv2d(&mut graph, x, 1, 32, (3, 3), (1, 1), (1, 1), Some("conv1"))?;
/// let act1 = graph.new_relu_node(conv1.output, Some("act1"))?;
/// let pool1 = max_pool2d(&mut graph, act1, (2, 2), Some((2, 2)), Some("pool1"))?;
/// ```
pub fn max_pool2d(
    graph: &mut GraphInner,
    input: NodeId,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    name: Option<&str>,
) -> Result<MaxPool2dOutput, GraphError> {
    let prefix = name.unwrap_or("max_pool2d");

    // 创建 MaxPool2d 节点
    let output =
        graph.new_max_pool2d_node(input, kernel_size, stride, Some(&format!("{prefix}_out")))?;

    Ok(MaxPool2dOutput { output })
}
