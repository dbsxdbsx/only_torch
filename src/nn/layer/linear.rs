/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : Linear (全连接) 层
 *
 * 设计决策：Layer 层只提供 Batch 版本（Batch-First 设计）
 * - 输入格式：[batch_size, in_features]
 * - 输出格式：[batch_size, out_features]
 * - 单样本需求可直接用 Node API
 *
 * 等价于 PyTorch 的 nn.Linear（天然支持 batch）
 */

use crate::nn::{Graph, GraphError, NodeId};

/// Linear 层的输出结构
///
/// 暴露所有内部节点，便于：
/// - 访问权重/偏置进行初始化或检查
/// - NEAT 进化时操作内部参数
/// - 调试和可视化
#[derive(Debug, Clone, Copy)]
pub struct LinearOutput {
    /// 输出节点 ID (最终计算结果) [batch_size, out_features]
    pub output: NodeId,
    /// 权重参数节点 ID [in_features, out_features]
    pub weights: NodeId,
    /// 偏置参数节点 ID [1, out_features]
    pub bias: NodeId,
    /// ones 矩阵节点 ID [batch_size, 1]，用于 bias 广播
    pub ones: NodeId,
}

/// 创建 Linear (全连接) 层
///
/// # 设计
/// - **Batch-First**：输入 `[batch, in]`，输出 `[batch, out]`
/// - 计算：`output = x @ W + b`
/// - 符合 PyTorch `nn.Linear` 语义
///
/// # 参数
/// - `graph`: 计算图
/// - `input`: 输入节点 ID，形状 `[batch_size, in_features]`
/// - `in_features`: 输入特征维度
/// - `out_features`: 输出特征维度
/// - `batch_size`: 批大小（用于创建 ones 矩阵）
/// - `name`: 层名称前缀（可选）
///
/// # 返回
/// - `LinearOutput` 包含 output、weights、bias、ones 节点 ID
///
/// # 内部结构
/// ```text
/// input [batch, in]
///       │
///       ▼
/// ┌─────────────┐
/// │ MatMul(x,W) │◄── W [in, out]
/// └─────────────┘
///       │
///       ▼
/// ┌─────────────────────┐
/// │ Add(+ones @ b)      │◄── b [1, out], ones [batch, 1]
/// └─────────────────────┘
///       │
///       ▼
/// output [batch, out]
/// ```
///
/// # 示例
/// ```ignore
/// // 构建 2 层 MLP
/// let fc1 = linear(&mut graph, x, 784, 128, batch_size, Some("fc1"))?;
/// let act = graph.new_sigmoid_node(fc1.output, Some("act"))?;
/// let fc2 = linear(&mut graph, act, 128, 10, batch_size, Some("fc2"))?;
///
/// // 设置 ones 矩阵（训练前必须设置）
/// graph.set_node_value(fc1.ones, Some(&Tensor::ones(&[batch_size, 1])))?;
/// graph.set_node_value(fc2.ones, Some(&Tensor::ones(&[batch_size, 1])))?;
///
/// // 访问内部参数
/// let weights = graph.get_node_value(fc1.weights)?;
/// ```
pub fn linear(
    graph: &mut Graph,
    input: NodeId,
    in_features: usize,
    out_features: usize,
    batch_size: usize,
    name: Option<&str>,
) -> Result<LinearOutput, GraphError> {
    let prefix = name.unwrap_or("linear");

    // 创建权重参数 W: [in_features, out_features]
    let weights =
        graph.new_parameter_node(&[in_features, out_features], Some(&format!("{}_W", prefix)))?;

    // 创建偏置参数 b: [1, out_features]
    let bias = graph.new_parameter_node(&[1, out_features], Some(&format!("{}_b", prefix)))?;

    // 创建 ones 矩阵: [batch_size, 1]，用于 bias 广播
    let ones = graph.new_input_node(&[batch_size, 1], Some(&format!("{}_ones", prefix)))?;

    // 计算 x @ W: [batch, in] @ [in, out] = [batch, out]
    let xw = graph.new_mat_mul_node(input, weights, Some(&format!("{}_xW", prefix)))?;

    // 计算 ones @ b: [batch, 1] @ [1, out] = [batch, out]（广播 bias）
    let b_broadcast =
        graph.new_mat_mul_node(ones, bias, Some(&format!("{}_b_broadcast", prefix)))?;

    // 计算 x @ W + b
    let output = graph.new_add_node(&[xw, b_broadcast], Some(&format!("{}_out", prefix)))?;

    Ok(LinearOutput {
        output,
        weights,
        bias,
        ones,
    })
}
