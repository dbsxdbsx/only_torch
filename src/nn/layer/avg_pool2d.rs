/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : AvgPool2d (2D 平均池化) 层
 *
 * 设计决策：Layer 层只提供 Batch 版本（Batch-First 设计）
 * - 输入格式：[batch_size, channels, H, W]
 * - 输出格式：[batch_size, channels, H', W']
 * - 单样本需求可直接用 Node API
 *
 * 等价于 PyTorch 的 nn.AvgPool2d
 */

use crate::nn::{Graph, GraphError, NodeId};

/// AvgPool2d 层的输出结构
///
/// 池化层没有可学习参数，只返回输出节点
#[derive(Debug, Clone, Copy)]
pub struct AvgPool2dOutput {
    /// 输出节点 ID [batch, channels, H', W']
    pub output: NodeId,
}

/// 创建 AvgPool2d (2D 平均池化) 层
///
/// # 设计
/// - **Batch-First**：输入 `[batch, C, H, W]`，输出 `[batch, C, H', W']`
/// - 无可学习参数
/// - 符合 PyTorch `nn.AvgPool2d` 语义
///
/// # 参数
/// - `graph`: 计算图
/// - `input`: 输入节点 ID，形状 `[batch_size, channels, H, W]`
/// - `kernel_size`: 池化窗口大小 (kH, kW)
/// - `stride`: 步长 (sH, sW)，若为 None 则默认等于 kernel_size
/// - `name`: 层名称前缀（可选）
///
/// # 返回
/// - `AvgPool2dOutput` 包含 output 节点 ID
///
/// # 输出尺寸计算
/// ```text
/// H' = (H - kernel_h) / stride_h + 1
/// W' = (W - kernel_w) / stride_w + 1
/// ```
///
/// # 示例
/// ```ignore
/// // 典型用法：全局平均池化
/// let gap = avg_pool2d(&mut graph, features, (7, 7), None, Some("gap"))?;
/// ```
pub fn avg_pool2d(
    graph: &mut Graph,
    input: NodeId,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    name: Option<&str>,
) -> Result<AvgPool2dOutput, GraphError> {
    let prefix = name.unwrap_or("avg_pool2d");

    // 创建 AvgPool2d 节点
    let output = graph.new_avg_pool2d_node(input, kernel_size, stride, Some(&format!("{}_out", prefix)))?;

    Ok(AvgPool2dOutput { output })
}
