/*
 * @Author       : 老董
 * @Date         : 2026-01-17
 * @Description  : Gru (门控循环单元) 层 - PyTorch 风格 API
 *
 * 公式:
 *   r_t = σ(x_t @ W_ir + h_{t-1} @ W_hr + b_r)     # 重置门
 *   z_t = σ(x_t @ W_iz + h_{t-1} @ W_hz + b_z)     # 更新门
 *   n_t = tanh(x_t @ W_in + r_t ⊙ (h_{t-1} @ W_hn) + b_n)  # 候选状态
 *   h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}         # 隐藏状态
 *
 * 输入/输出形状：
 * - 输入：[batch_size, input_size]
 * - 输出：[batch_size, hidden_size]
 */

use crate::nn::{Graph, GraphError, Init, Module, Var};
use crate::tensor::Tensor;

// ==================== 新版 Gru 结构体 ====================

/// Gru (门控循环单元) 层
///
/// `PyTorch` 风格的 GRU 层，包含重置门、更新门和候选状态。
/// 比 LSTM 更简单高效（2 个门 vs 4 个门）。
///
/// # 输入/输出形状
/// - 输入：[`batch_size`, `input_size`]
/// - 输出：[`batch_size`, `hidden_size`]
///
/// # 使用示例
/// ```ignore
/// let gru = Gru::new(&graph, 10, 20, 32, "gru1")?;
/// gru.step(&x_t)?;
/// let h_t = gru.hidden().value()?;
/// ```
pub struct Gru {
    // === 重置门参数 ===
    w_ir: Var,
    w_hr: Var,
    b_r: Var,
    // === 更新门参数 ===
    w_iz: Var,
    w_hz: Var,
    b_z: Var,
    // === 候选状态参数 ===
    w_in: Var,
    w_hn: Var,
    b_n: Var,
    // === 状态节点 ===
    hidden_output: Var,
    hidden_input: Var,
    // === 输入节点 ===
    input_node: Var,
    // === Graph 和配置 ===
    graph: Graph,
    input_size: usize,
    hidden_size: usize,
    batch_size: usize,
    #[allow(dead_code)]
    name: String,
}

impl Gru {
    /// 创建新的 Gru 层
    pub fn new(
        graph: &Graph,
        input_size: usize,
        hidden_size: usize,
        batch_size: usize,
        name: &str,
    ) -> Result<Self, GraphError> {
        // 创建参数节点
        let w_ir = graph.parameter(
            &[input_size, hidden_size],
            Init::Kaiming,
            &format!("{name}_W_ir"),
        )?;
        let w_hr = graph.parameter(
            &[hidden_size, hidden_size],
            Init::Kaiming,
            &format!("{name}_W_hr"),
        )?;
        let b_r = graph.parameter(&[1, hidden_size], Init::Zeros, &format!("{name}_b_r"))?;

        let w_iz = graph.parameter(
            &[input_size, hidden_size],
            Init::Kaiming,
            &format!("{name}_W_iz"),
        )?;
        let w_hz = graph.parameter(
            &[hidden_size, hidden_size],
            Init::Kaiming,
            &format!("{name}_W_hz"),
        )?;
        let b_z = graph.parameter(&[1, hidden_size], Init::Zeros, &format!("{name}_b_z"))?;

        let w_in = graph.parameter(
            &[input_size, hidden_size],
            Init::Kaiming,
            &format!("{name}_W_in"),
        )?;
        let w_hn = graph.parameter(
            &[hidden_size, hidden_size],
            Init::Kaiming,
            &format!("{name}_W_hn"),
        )?;
        let b_n = graph.parameter(&[1, hidden_size], Init::Zeros, &format!("{name}_b_n"))?;

        // 创建输入节点
        let input_node = graph.zeros(&[batch_size, input_size])?;

        // 创建状态节点和计算图结构
        let (hidden_input, hidden_output) = {
            let mut g = graph.inner_mut();

            // 创建状态节点
            let h_prev_id =
                g.new_state_node(&[batch_size, hidden_size], Some(&format!("{name}_h_prev")))?;
            g.set_node_value(h_prev_id, Some(&Tensor::zeros(&[batch_size, hidden_size])))?;

            // === 重置门计算 ===
            // Add 支持广播：[batch, hidden] + [batch, hidden] + [1, hidden]
            let x_ir = g.new_mat_mul_node(
                input_node.node_id(),
                w_ir.node_id(),
                Some(&format!("{name}_x_ir")),
            )?;
            let h_hr =
                g.new_mat_mul_node(h_prev_id, w_hr.node_id(), Some(&format!("{name}_h_hr")))?;
            let pre_r =
                g.new_add_node(&[x_ir, h_hr, b_r.node_id()], Some(&format!("{name}_pre_r")))?;
            let r_gate = g.new_sigmoid_node(pre_r, Some(&format!("{name}_r_gate")))?;

            // === 更新门计算 ===
            let x_iz = g.new_mat_mul_node(
                input_node.node_id(),
                w_iz.node_id(),
                Some(&format!("{name}_x_iz")),
            )?;
            let h_hz =
                g.new_mat_mul_node(h_prev_id, w_hz.node_id(), Some(&format!("{name}_h_hz")))?;
            let pre_z =
                g.new_add_node(&[x_iz, h_hz, b_z.node_id()], Some(&format!("{name}_pre_z")))?;
            let z_gate = g.new_sigmoid_node(pre_z, Some(&format!("{name}_z_gate")))?;

            // === 候选状态计算 ===
            let x_in = g.new_mat_mul_node(
                input_node.node_id(),
                w_in.node_id(),
                Some(&format!("{name}_x_in")),
            )?;
            let h_hn =
                g.new_mat_mul_node(h_prev_id, w_hn.node_id(), Some(&format!("{name}_h_hn")))?;
            let r_h_hn = g.new_multiply_node(r_gate, h_hn, Some(&format!("{name}_r_h_hn")))?;
            let pre_n = g.new_add_node(
                &[x_in, r_h_hn, b_n.node_id()],
                Some(&format!("{name}_pre_n")),
            )?;
            let n_gate = g.new_tanh_node(pre_n, Some(&format!("{name}_n_gate")))?;

            // === 隐藏状态更新: h_t = (1-z_t) ⊙ n_t + z_t ⊙ h_{t-1} ===
            // 重写为: h_t = n_t + z_t ⊙ (h_{t-1} - n_t)
            let h_minus_n =
                g.new_subtract_node(h_prev_id, n_gate, Some(&format!("{name}_h_minus_n")))?;
            let z_diff = g.new_multiply_node(z_gate, h_minus_n, Some(&format!("{name}_z_diff")))?;
            let hidden_id = g.new_add_node(&[n_gate, z_diff], Some(&format!("{name}_h")))?;

            // === 建立循环连接 ===
            g.connect_recurrent(hidden_id, h_prev_id)?;

            // 注册层分组
            g.register_layer_group(
                name,
                "Gru",
                &format!("{input_size}→{hidden_size}"),
                vec![
                    w_ir.node_id(),
                    w_hr.node_id(),
                    b_r.node_id(),
                    w_iz.node_id(),
                    w_hz.node_id(),
                    b_z.node_id(),
                    w_in.node_id(),
                    w_hn.node_id(),
                    b_n.node_id(),
                    r_gate,
                    z_gate,
                    n_gate,
                    hidden_id,
                ],
            );

            let h_prev = Var::new(h_prev_id, graph.inner_rc());
            let hidden = Var::new(hidden_id, graph.inner_rc());

            (h_prev, hidden)
        };

        Ok(Self {
            w_ir,
            w_hr,
            b_r,
            w_iz,
            w_hz,
            b_z,
            w_in,
            w_hn,
            b_n,
            hidden_output,
            hidden_input,
            input_node,
            graph: graph.clone(),
            input_size,
            hidden_size,
            batch_size,
            name: name.to_string(),
        })
    }

    /// 单步前向传播
    pub fn step(&self, x: &Tensor) -> Result<&Var, GraphError> {
        self.input_node.set_value(x)?;
        self.graph.inner_mut().step(self.hidden_output.node_id())?;
        Ok(&self.hidden_output)
    }

    /// 完整重置
    pub fn reset(&self) {
        self.graph.inner_mut().reset();
    }

    /// 重置隐藏状态为零
    pub fn reset_hidden(&self) -> Result<(), GraphError> {
        self.hidden_input
            .set_value(&Tensor::zeros(&[self.batch_size, self.hidden_size]))?;
        Ok(())
    }

    // === Getter 方法 ===
    pub const fn hidden(&self) -> &Var {
        &self.hidden_output
    }
    pub const fn hidden_input(&self) -> &Var {
        &self.hidden_input
    }
    pub const fn input(&self) -> &Var {
        &self.input_node
    }

    // 重置门参数
    pub const fn w_ir(&self) -> &Var {
        &self.w_ir
    }
    pub const fn w_hr(&self) -> &Var {
        &self.w_hr
    }
    pub const fn b_r(&self) -> &Var {
        &self.b_r
    }

    // 更新门参数
    pub const fn w_iz(&self) -> &Var {
        &self.w_iz
    }
    pub const fn w_hz(&self) -> &Var {
        &self.w_hz
    }
    pub const fn b_z(&self) -> &Var {
        &self.b_z
    }

    // 候选状态参数
    pub const fn w_in(&self) -> &Var {
        &self.w_in
    }
    pub const fn w_hn(&self) -> &Var {
        &self.w_hn
    }
    pub const fn b_n(&self) -> &Var {
        &self.b_n
    }

    pub const fn input_size(&self) -> usize {
        self.input_size
    }
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }
    pub const fn graph(&self) -> &Graph {
        &self.graph
    }
}

impl Module for Gru {
    fn parameters(&self) -> Vec<Var> {
        vec![
            self.w_ir.clone(),
            self.w_hr.clone(),
            self.b_r.clone(),
            self.w_iz.clone(),
            self.w_hz.clone(),
            self.b_z.clone(),
            self.w_in.clone(),
            self.w_hn.clone(),
            self.b_n.clone(),
        ]
    }
}

// 单元测试位于 src/nn/tests/layer_gru.rs
