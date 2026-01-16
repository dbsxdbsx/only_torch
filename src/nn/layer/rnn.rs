/*
 * @Author       : 老董
 * @Date         : 2025-12-30
 * @Description  : Vanilla RNN 层 - 便捷函数，组合 Node 构建 RNN 结构
 *
 * 公式: h_t = tanh(x_t @ W_ih + h_{t-1} @ W_hh + b_h)
 *
 * 与 PyTorch nn.RNNCell 对齐:
 * - input: [batch, input_size]
 * - hidden: [batch, hidden_size]
 * - W_ih: [input_size, hidden_size]  (PyTorch 是转置的 [hidden, input])
 * - W_hh: [hidden_size, hidden_size] (PyTorch 是转置的 [hidden, hidden])
 * - b_ih + b_hh: 合并为 b_h [1, hidden_size]
 */

use crate::nn::GraphError;
use crate::nn::graph::GraphInner;
use crate::nn::nodes::NodeId;
use crate::tensor::Tensor;

/// RNN 层输出结构
///
/// 包含所有创建的节点 ID，便于后续访问和调试
#[derive(Debug, Clone)]
pub struct RnnOutput {
    /// 隐藏状态输出节点（经过 tanh 激活）
    pub hidden: NodeId,
    /// 隐藏状态输入节点（State 节点，接收上一时间步的 hidden）
    pub h_prev: NodeId,
    /// 输入到隐藏权重 `W_ih`: [`input_size`, `hidden_size`]
    pub w_ih: NodeId,
    /// 隐藏到隐藏权重 `W_hh`: [`hidden_size`, `hidden_size`]
    pub w_hh: NodeId,
    /// 隐藏层偏置 `b_h`: [1, `hidden_size`]
    pub b_h: NodeId,
}

/// 创建 Vanilla RNN 层
///
/// # 参数
/// - `graph`: 计算图
/// - `input`: 输入节点，形状 [`batch_size`, `input_size`]
/// - `input_size`: 输入特征维度
/// - `hidden_size`: 隐藏状态维度
/// - `batch_size`: 批大小
/// - `name`: 可选的层名称前缀
///
/// # 返回
/// - `RnnOutput`: 包含所有相关节点 ID
///
/// # 公式
/// ```text
/// h_t = tanh(x_t @ W_ih + h_{t-1} @ W_hh + b_h)
/// ```
///
/// # 示例
/// ```ignore
/// let rnn = rnn(&mut graph, input, 10, 20, 32, Some("rnn1"))?;
/// // 使用 rnn.hidden 作为输出
/// // 使用 rnn.h_prev 设置初始隐藏状态（可选）
/// ```
pub fn rnn(
    graph: &mut GraphInner,
    input: NodeId,
    input_size: usize,
    hidden_size: usize,
    batch_size: usize,
    name: Option<&str>,
) -> Result<RnnOutput, GraphError> {
    let prefix = name.unwrap_or("rnn");

    // === 创建参数节点 ===

    // W_ih: [input_size, hidden_size] - 输入到隐藏权重
    let w_ih =
        graph.new_parameter_node(&[input_size, hidden_size], Some(&format!("{prefix}_W_ih")))?;

    // W_hh: [hidden_size, hidden_size] - 隐藏到隐藏权重
    let w_hh =
        graph.new_parameter_node(&[hidden_size, hidden_size], Some(&format!("{prefix}_W_hh")))?;

    // b_h: [1, hidden_size] - 隐藏层偏置
    let b_h = graph.new_parameter_node(&[1, hidden_size], Some(&format!("{prefix}_b_h")))?;
    // 初始化偏置为 0
    graph.set_node_value(b_h, Some(&Tensor::zeros(&[1, hidden_size])))?;

    // === 创建状态节点 ===

    // h_prev: [batch_size, hidden_size] - 上一时间步的隐藏状态
    let h_prev = graph.new_state_node(
        &[batch_size, hidden_size],
        Some(&format!("{prefix}_h_prev")),
    )?;
    // 初始化为 0
    graph.set_node_value(h_prev, Some(&Tensor::zeros(&[batch_size, hidden_size])))?;

    // === 创建 ones 用于偏置广播 ===

    // ones: [batch_size, 1] - 用于将 [1, hidden_size] 广播到 [batch_size, hidden_size]
    let ones = graph.new_input_node(&[batch_size, 1], Some(&format!("{prefix}_ones")))?;
    graph.set_node_value(ones, Some(&Tensor::ones(&[batch_size, 1])))?;

    // === 计算隐藏状态 ===

    // input_contrib = x_t @ W_ih: [batch, input] @ [input, hidden] = [batch, hidden]
    let input_contrib =
        graph.new_mat_mul_node(input, w_ih, Some(&format!("{prefix}_input_contrib")))?;

    // hidden_contrib = h_{t-1} @ W_hh: [batch, hidden] @ [hidden, hidden] = [batch, hidden]
    let hidden_contrib =
        graph.new_mat_mul_node(h_prev, w_hh, Some(&format!("{prefix}_hidden_contrib")))?;

    // bias_broadcast = ones @ b_h: [batch, 1] @ [1, hidden] = [batch, hidden]
    let bias_broadcast =
        graph.new_mat_mul_node(ones, b_h, Some(&format!("{prefix}_bias_broadcast")))?;

    // pre_hidden = input_contrib + hidden_contrib
    let sum1 = graph.new_add_node(
        &[input_contrib, hidden_contrib],
        Some(&format!("{prefix}_sum1")),
    )?;

    // pre_hidden = sum1 + bias_broadcast
    let pre_hidden =
        graph.new_add_node(&[sum1, bias_broadcast], Some(&format!("{prefix}_pre_h")))?;

    // hidden = tanh(pre_hidden)
    let hidden = graph.new_tanh_node(pre_hidden, Some(&format!("{prefix}_h")))?;

    // === 建立循环连接 ===
    // hidden 的值在下一个 step() 时传递给 h_prev
    graph.connect_recurrent(hidden, h_prev)?;

    // === 注册层分组（用于可视化） ===
    graph.register_layer_group(
        prefix,
        "RNN",
        &format!("{input_size}→{hidden_size}"),
        vec![
            w_ih,
            w_hh,
            b_h,
            ones,
            input_contrib,
            hidden_contrib,
            bias_broadcast,
            sum1,
            pre_hidden,
            hidden,
        ],
    );

    Ok(RnnOutput {
        hidden,
        h_prev,
        w_ih,
        w_hh,
        b_h,
    })
}
// 单元测试位于 src/nn/tests/layer_rnn.rs
