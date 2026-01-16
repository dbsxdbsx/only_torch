/*
 * @Author       : 老董
 * @Date         : 2025-12-30
 * @Description  : LSTM 层 - 便捷函数，组合 Node 构建 LSTM 结构
 *
 * 公式:
 *   i_t = σ(x_t @ W_ii + h_{t-1} @ W_hi + b_i)   # 输入门
 *   f_t = σ(x_t @ W_if + h_{t-1} @ W_hf + b_f)   # 遗忘门
 *   g_t = tanh(x_t @ W_ig + h_{t-1} @ W_hg + b_g) # 候选细胞
 *   o_t = σ(x_t @ W_io + h_{t-1} @ W_ho + b_o)   # 输出门
 *   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t              # 细胞状态
 *   h_t = o_t ⊙ tanh(c_t)                        # 隐藏状态
 *
 * 与 PyTorch nn.LSTMCell 对齐:
 * - input: [batch, input_size]
 * - hidden: [batch, hidden_size]
 * - cell: [batch, hidden_size]
 *
 * 权重布局（与 PyTorch 不同，我们使用更清晰的分离结构）:
 * - W_ii, W_if, W_ig, W_io: 各 [input_size, hidden_size]
 * - W_hi, W_hf, W_hg, W_ho: 各 [hidden_size, hidden_size]
 * - b_i, b_f, b_g, b_o: 各 [1, hidden_size]
 */

use crate::nn::GraphError;
use crate::nn::graph::GraphInner;
use crate::nn::nodes::NodeId;
use crate::tensor::Tensor;

/// LSTM 层输出结构
#[derive(Debug, Clone)]
pub struct LstmOutput {
    /// 隐藏状态输出节点 `h_t`: [batch, `hidden_size`]
    pub hidden: NodeId,
    /// 细胞状态输出节点 `c_t`: [batch, `hidden_size`]
    pub cell: NodeId,
    /// 上一时间步隐藏状态 (State 节点)
    pub h_prev: NodeId,
    /// 上一时间步细胞状态 (State 节点)
    pub c_prev: NodeId,
    // === 输入门参数 ===
    pub w_ii: NodeId,
    pub w_hi: NodeId,
    pub b_i: NodeId,
    // === 遗忘门参数 ===
    pub w_if: NodeId,
    pub w_hf: NodeId,
    pub b_f: NodeId,
    // === 候选细胞参数 ===
    pub w_ig: NodeId,
    pub w_hg: NodeId,
    pub b_g: NodeId,
    // === 输出门参数 ===
    pub w_io: NodeId,
    pub w_ho: NodeId,
    pub b_o: NodeId,
}

/// 创建 LSTM 层
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
/// - `LstmOutput`: 包含所有相关节点 ID
pub fn lstm(
    graph: &mut GraphInner,
    input: NodeId,
    input_size: usize,
    hidden_size: usize,
    batch_size: usize,
    name: Option<&str>,
) -> Result<LstmOutput, GraphError> {
    let prefix = name.unwrap_or("lstm");

    // === 创建状态节点 ===
    let h_prev = graph.new_state_node(
        &[batch_size, hidden_size],
        Some(&format!("{prefix}_h_prev")),
    )?;
    graph.set_node_value(h_prev, Some(&Tensor::zeros(&[batch_size, hidden_size])))?;

    let c_prev = graph.new_state_node(
        &[batch_size, hidden_size],
        Some(&format!("{prefix}_c_prev")),
    )?;
    graph.set_node_value(c_prev, Some(&Tensor::zeros(&[batch_size, hidden_size])))?;

    // === 创建 ones 用于偏置广播 ===
    let ones = graph.new_input_node(&[batch_size, 1], Some(&format!("{prefix}_ones")))?;
    graph.set_node_value(ones, Some(&Tensor::ones(&[batch_size, 1])))?;

    // === 输入门参数 ===
    let w_ii =
        graph.new_parameter_node(&[input_size, hidden_size], Some(&format!("{prefix}_W_ii")))?;
    let w_hi =
        graph.new_parameter_node(&[hidden_size, hidden_size], Some(&format!("{prefix}_W_hi")))?;
    let b_i = graph.new_parameter_node(&[1, hidden_size], Some(&format!("{prefix}_b_i")))?;
    graph.set_node_value(b_i, Some(&Tensor::zeros(&[1, hidden_size])))?;

    // === 遗忘门参数 ===
    let w_if =
        graph.new_parameter_node(&[input_size, hidden_size], Some(&format!("{prefix}_W_if")))?;
    let w_hf =
        graph.new_parameter_node(&[hidden_size, hidden_size], Some(&format!("{prefix}_W_hf")))?;
    let b_f = graph.new_parameter_node(&[1, hidden_size], Some(&format!("{prefix}_b_f")))?;
    // 遗忘门偏置初始化为 1（有助于训练初期记住信息）
    graph.set_node_value(b_f, Some(&Tensor::ones(&[1, hidden_size])))?;

    // === 候选细胞参数 ===
    let w_ig =
        graph.new_parameter_node(&[input_size, hidden_size], Some(&format!("{prefix}_W_ig")))?;
    let w_hg =
        graph.new_parameter_node(&[hidden_size, hidden_size], Some(&format!("{prefix}_W_hg")))?;
    let b_g = graph.new_parameter_node(&[1, hidden_size], Some(&format!("{prefix}_b_g")))?;
    graph.set_node_value(b_g, Some(&Tensor::zeros(&[1, hidden_size])))?;

    // === 输出门参数 ===
    let w_io =
        graph.new_parameter_node(&[input_size, hidden_size], Some(&format!("{prefix}_W_io")))?;
    let w_ho =
        graph.new_parameter_node(&[hidden_size, hidden_size], Some(&format!("{prefix}_W_ho")))?;
    let b_o = graph.new_parameter_node(&[1, hidden_size], Some(&format!("{prefix}_b_o")))?;
    graph.set_node_value(b_o, Some(&Tensor::zeros(&[1, hidden_size])))?;

    // === 输入门计算: i_t = σ(x @ W_ii + h_prev @ W_hi + b_i) ===
    let x_ii = graph.new_mat_mul_node(input, w_ii, Some(&format!("{prefix}_x_ii")))?;
    let h_hi = graph.new_mat_mul_node(h_prev, w_hi, Some(&format!("{prefix}_h_hi")))?;
    let b_i_broadcast = graph.new_mat_mul_node(ones, b_i, Some(&format!("{prefix}_b_i_bc")))?;
    let pre_i = graph.new_add_node(
        &[x_ii, h_hi, b_i_broadcast],
        Some(&format!("{prefix}_pre_i")),
    )?;
    let i_gate = graph.new_sigmoid_node(pre_i, Some(&format!("{prefix}_i_gate")))?;

    // === 遗忘门计算: f_t = σ(x @ W_if + h_prev @ W_hf + b_f) ===
    let x_if = graph.new_mat_mul_node(input, w_if, Some(&format!("{prefix}_x_if")))?;
    let h_hf = graph.new_mat_mul_node(h_prev, w_hf, Some(&format!("{prefix}_h_hf")))?;
    let b_f_broadcast = graph.new_mat_mul_node(ones, b_f, Some(&format!("{prefix}_b_f_bc")))?;
    let pre_f = graph.new_add_node(
        &[x_if, h_hf, b_f_broadcast],
        Some(&format!("{prefix}_pre_f")),
    )?;
    let f_gate = graph.new_sigmoid_node(pre_f, Some(&format!("{prefix}_f_gate")))?;

    // === 候选细胞计算: g_t = tanh(x @ W_ig + h_prev @ W_hg + b_g) ===
    let x_ig = graph.new_mat_mul_node(input, w_ig, Some(&format!("{prefix}_x_ig")))?;
    let h_hg = graph.new_mat_mul_node(h_prev, w_hg, Some(&format!("{prefix}_h_hg")))?;
    let b_g_broadcast = graph.new_mat_mul_node(ones, b_g, Some(&format!("{prefix}_b_g_bc")))?;
    let pre_g = graph.new_add_node(
        &[x_ig, h_hg, b_g_broadcast],
        Some(&format!("{prefix}_pre_g")),
    )?;
    let g_gate = graph.new_tanh_node(pre_g, Some(&format!("{prefix}_g_gate")))?;

    // === 输出门计算: o_t = σ(x @ W_io + h_prev @ W_ho + b_o) ===
    let x_io = graph.new_mat_mul_node(input, w_io, Some(&format!("{prefix}_x_io")))?;
    let h_ho = graph.new_mat_mul_node(h_prev, w_ho, Some(&format!("{prefix}_h_ho")))?;
    let b_o_broadcast = graph.new_mat_mul_node(ones, b_o, Some(&format!("{prefix}_b_o_bc")))?;
    let pre_o = graph.new_add_node(
        &[x_io, h_ho, b_o_broadcast],
        Some(&format!("{prefix}_pre_o")),
    )?;
    let o_gate = graph.new_sigmoid_node(pre_o, Some(&format!("{prefix}_o_gate")))?;

    // === 细胞状态更新: c_t = f_t ⊙ c_prev + i_t ⊙ g_t ===
    let f_c = graph.new_multiply_node(f_gate, c_prev, Some(&format!("{prefix}_f_c")))?;
    let i_g = graph.new_multiply_node(i_gate, g_gate, Some(&format!("{prefix}_i_g")))?;
    let cell = graph.new_add_node(&[f_c, i_g], Some(&format!("{prefix}_c")))?;

    // === 隐藏状态更新: h_t = o_t ⊙ tanh(c_t) ===
    let tanh_c = graph.new_tanh_node(cell, Some(&format!("{prefix}_tanh_c")))?;
    let hidden = graph.new_multiply_node(o_gate, tanh_c, Some(&format!("{prefix}_h")))?;

    // === 建立循环连接 ===
    graph.connect_recurrent(hidden, h_prev)?;
    graph.connect_recurrent(cell, c_prev)?;

    // === 注册层分组（用于可视化） ===
    graph.register_layer_group(
        prefix,
        "LSTM",
        &format!("{input_size}→{hidden_size}"),
        vec![
            // 参数节点
            w_ii,
            w_hi,
            b_i,
            w_if,
            w_hf,
            b_f,
            w_ig,
            w_hg,
            b_g,
            w_io,
            w_ho,
            b_o,
            ones,
            // 输入门
            x_ii,
            h_hi,
            b_i_broadcast,
            pre_i,
            i_gate,
            // 遗忘门
            x_if,
            h_hf,
            b_f_broadcast,
            pre_f,
            f_gate,
            // 候选细胞
            x_ig,
            h_hg,
            b_g_broadcast,
            pre_g,
            g_gate,
            // 输出门
            x_io,
            h_ho,
            b_o_broadcast,
            pre_o,
            o_gate,
            // 状态更新
            f_c,
            i_g,
            cell,
            tanh_c,
            hidden,
        ],
    );

    Ok(LstmOutput {
        hidden,
        cell,
        h_prev,
        c_prev,
        w_ii,
        w_hi,
        b_i,
        w_if,
        w_hf,
        b_f,
        w_ig,
        w_hg,
        b_g,
        w_io,
        w_ho,
        b_o,
    })
}
// 单元测试位于 src/nn/tests/layer_lstm.rs
