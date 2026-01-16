/*
 * @Author       : 老董
 * @Date         : 2025-12-30
 * @Description  : GRU 层 - 便捷函数，组合 Node 构建 GRU 结构
 *
 * 公式:
 *   r_t = σ(x_t @ W_ir + h_{t-1} @ W_hr + b_r)     # 重置门
 *   z_t = σ(x_t @ W_iz + h_{t-1} @ W_hz + b_z)     # 更新门
 *   n_t = tanh(x_t @ W_in + r_t ⊙ (h_{t-1} @ W_hn) + b_n)  # 候选状态
 *   h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}         # 隐藏状态
 *
 * 与 PyTorch nn.GRUCell 对齐:
 * - input: [batch, input_size]
 * - hidden: [batch, hidden_size]
 *
 * GRU 相比 LSTM 更简单（2 个门 vs 4 个门），计算效率更高
 */

use crate::nn::GraphError;
use crate::nn::graph::GraphInner;
use crate::nn::nodes::NodeId;
use crate::tensor::Tensor;

/// GRU 层输出结构
#[derive(Debug, Clone)]
pub struct GruOutput {
    /// 隐藏状态输出节点 `h_t`: [batch, `hidden_size`]
    pub hidden: NodeId,
    /// 上一时间步隐藏状态 (State 节点)
    pub h_prev: NodeId,
    // === 重置门参数 ===
    pub w_ir: NodeId,
    pub w_hr: NodeId,
    pub b_r: NodeId,
    // === 更新门参数 ===
    pub w_iz: NodeId,
    pub w_hz: NodeId,
    pub b_z: NodeId,
    // === 候选状态参数 ===
    pub w_in: NodeId,
    pub w_hn: NodeId,
    pub b_n: NodeId,
}

/// 创建 GRU 层
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
/// - `GruOutput`: 包含所有相关节点 ID
pub fn gru(
    graph: &mut GraphInner,
    input: NodeId,
    input_size: usize,
    hidden_size: usize,
    batch_size: usize,
    name: Option<&str>,
) -> Result<GruOutput, GraphError> {
    let prefix = name.unwrap_or("gru");

    // === 创建状态节点 ===
    let h_prev = graph.new_state_node(
        &[batch_size, hidden_size],
        Some(&format!("{prefix}_h_prev")),
    )?;
    graph.set_node_value(h_prev, Some(&Tensor::zeros(&[batch_size, hidden_size])))?;

    // === 创建 ones 用于偏置广播 ===
    let ones = graph.new_input_node(&[batch_size, 1], Some(&format!("{prefix}_ones")))?;
    graph.set_node_value(ones, Some(&Tensor::ones(&[batch_size, 1])))?;

    // === 创建 one 用于 (1 - z_t) 计算 ===
    let one = graph.new_parameter_node(&[1, 1], Some(&format!("{prefix}_one")))?;
    graph.set_node_value(one, Some(&Tensor::new(&[1.0], &[1, 1])))?;

    let neg_one = graph.new_parameter_node(&[1, 1], Some(&format!("{prefix}_neg_one")))?;
    graph.set_node_value(neg_one, Some(&Tensor::new(&[-1.0], &[1, 1])))?;

    // === 重置门参数 ===
    let w_ir =
        graph.new_parameter_node(&[input_size, hidden_size], Some(&format!("{prefix}_W_ir")))?;
    let w_hr =
        graph.new_parameter_node(&[hidden_size, hidden_size], Some(&format!("{prefix}_W_hr")))?;
    let b_r = graph.new_parameter_node(&[1, hidden_size], Some(&format!("{prefix}_b_r")))?;
    graph.set_node_value(b_r, Some(&Tensor::zeros(&[1, hidden_size])))?;

    // === 更新门参数 ===
    let w_iz =
        graph.new_parameter_node(&[input_size, hidden_size], Some(&format!("{prefix}_W_iz")))?;
    let w_hz =
        graph.new_parameter_node(&[hidden_size, hidden_size], Some(&format!("{prefix}_W_hz")))?;
    let b_z = graph.new_parameter_node(&[1, hidden_size], Some(&format!("{prefix}_b_z")))?;
    graph.set_node_value(b_z, Some(&Tensor::zeros(&[1, hidden_size])))?;

    // === 候选状态参数 ===
    let w_in =
        graph.new_parameter_node(&[input_size, hidden_size], Some(&format!("{prefix}_W_in")))?;
    let w_hn =
        graph.new_parameter_node(&[hidden_size, hidden_size], Some(&format!("{prefix}_W_hn")))?;
    let b_n = graph.new_parameter_node(&[1, hidden_size], Some(&format!("{prefix}_b_n")))?;
    graph.set_node_value(b_n, Some(&Tensor::zeros(&[1, hidden_size])))?;

    // === 重置门计算: r_t = σ(x @ W_ir + h_prev @ W_hr + b_r) ===
    let x_ir = graph.new_mat_mul_node(input, w_ir, Some(&format!("{prefix}_x_ir")))?;
    let h_hr = graph.new_mat_mul_node(h_prev, w_hr, Some(&format!("{prefix}_h_hr")))?;
    let b_r_broadcast = graph.new_mat_mul_node(ones, b_r, Some(&format!("{prefix}_b_r_bc")))?;
    let pre_r = graph.new_add_node(
        &[x_ir, h_hr, b_r_broadcast],
        Some(&format!("{prefix}_pre_r")),
    )?;
    let r_gate = graph.new_sigmoid_node(pre_r, Some(&format!("{prefix}_r_gate")))?;

    // === 更新门计算: z_t = σ(x @ W_iz + h_prev @ W_hz + b_z) ===
    let x_iz = graph.new_mat_mul_node(input, w_iz, Some(&format!("{prefix}_x_iz")))?;
    let h_hz = graph.new_mat_mul_node(h_prev, w_hz, Some(&format!("{prefix}_h_hz")))?;
    let b_z_broadcast = graph.new_mat_mul_node(ones, b_z, Some(&format!("{prefix}_b_z_bc")))?;
    let pre_z = graph.new_add_node(
        &[x_iz, h_hz, b_z_broadcast],
        Some(&format!("{prefix}_pre_z")),
    )?;
    let z_gate = graph.new_sigmoid_node(pre_z, Some(&format!("{prefix}_z_gate")))?;

    // === 候选状态计算: n_t = tanh(x @ W_in + r_t ⊙ (h_prev @ W_hn) + b_n) ===
    let x_in = graph.new_mat_mul_node(input, w_in, Some(&format!("{prefix}_x_in")))?;
    let h_hn = graph.new_mat_mul_node(h_prev, w_hn, Some(&format!("{prefix}_h_hn")))?;
    let r_h = graph.new_multiply_node(r_gate, h_hn, Some(&format!("{prefix}_r_h")))?;
    let b_n_broadcast = graph.new_mat_mul_node(ones, b_n, Some(&format!("{prefix}_b_n_bc")))?;
    let pre_n = graph.new_add_node(
        &[x_in, r_h, b_n_broadcast],
        Some(&format!("{prefix}_pre_n")),
    )?;
    let n_gate = graph.new_tanh_node(pre_n, Some(&format!("{prefix}_n_gate")))?;

    // === 隐藏状态更新: h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_prev ===
    // 分解为: h_t = n_t - z_t ⊙ n_t + z_t ⊙ h_prev
    //       = n_t + z_t ⊙ (h_prev - n_t)
    let neg_n =
        graph.new_scalar_multiply_node(neg_one, n_gate, Some(&format!("{prefix}_neg_n")))?;
    let h_minus_n = graph.new_add_node(&[h_prev, neg_n], Some(&format!("{prefix}_h_minus_n")))?;
    let z_diff = graph.new_multiply_node(z_gate, h_minus_n, Some(&format!("{prefix}_z_diff")))?;
    let hidden = graph.new_add_node(&[n_gate, z_diff], Some(&format!("{prefix}_h")))?;

    // === 建立循环连接 ===
    graph.connect_recurrent(hidden, h_prev)?;

    // === 注册层分组（用于可视化） ===
    graph.register_layer_group(
        prefix,
        "GRU",
        &format!("{input_size}→{hidden_size}"),
        vec![
            // 参数节点
            w_ir,
            w_hr,
            b_r,
            w_iz,
            w_hz,
            b_z,
            w_in,
            w_hn,
            b_n,
            ones,
            one,
            neg_one,
            // 重置门
            x_ir,
            h_hr,
            b_r_broadcast,
            pre_r,
            r_gate,
            // 更新门
            x_iz,
            h_hz,
            b_z_broadcast,
            pre_z,
            z_gate,
            // 候选状态
            x_in,
            h_hn,
            r_h,
            b_n_broadcast,
            pre_n,
            n_gate,
            // 状态更新
            neg_n,
            h_minus_n,
            z_diff,
            hidden,
        ],
    );

    Ok(GruOutput {
        hidden,
        h_prev,
        w_ir,
        w_hr,
        b_r,
        w_iz,
        w_hz,
        b_z,
        w_in,
        w_hn,
        b_n,
    })
}
// 单元测试位于 src/nn/tests/layer_gru.rs
