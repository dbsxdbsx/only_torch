/*
 * @Author       : 老董
 * @Date         : 2026-01-17
 * @Description  : Lstm (长短期记忆) 层 - PyTorch 风格 API
 *
 * 公式:
 *   i_t = σ(x_t @ W_ii + h_{t-1} @ W_hi + b_i)   # 输入门
 *   f_t = σ(x_t @ W_if + h_{t-1} @ W_hf + b_f)   # 遗忘门
 *   g_t = tanh(x_t @ W_ig + h_{t-1} @ W_hg + b_g) # 候选细胞
 *   o_t = σ(x_t @ W_io + h_{t-1} @ W_ho + b_o)   # 输出门
 *   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t              # 细胞状态
 *   h_t = o_t ⊙ tanh(c_t)                        # 隐藏状态
 *
 * 输入/输出形状：
 * - 输入：[batch_size, input_size]
 * - 输出：hidden [batch_size, hidden_size], cell [batch_size, hidden_size]
 */

use crate::nn::{Graph, GraphError, Init, Module, Var};
use crate::tensor::Tensor;

// ==================== 新版 Lstm 结构体 ====================

/// Lstm (长短期记忆) 层
///
/// PyTorch 风格的 LSTM 层，包含输入门、遗忘门、候选细胞和输出门。
///
/// # 输入/输出形状
/// - 输入：[batch_size, input_size]
/// - 输出：hidden [batch_size, hidden_size], cell [batch_size, hidden_size]
///
/// # 使用示例
/// ```ignore
/// let lstm = Lstm::new(&graph, 10, 20, 32, "lstm1")?;
///
/// // 单步前向传播
/// lstm.step(&x_t)?;
/// let h_t = lstm.hidden().value()?;
/// let c_t = lstm.cell().value()?;
/// ```
pub struct Lstm {
    // === 输入门参数 ===
    w_ii: Var,
    w_hi: Var,
    b_i: Var,
    // === 遗忘门参数 ===
    w_if: Var,
    w_hf: Var,
    b_f: Var,
    // === 候选细胞参数 ===
    w_ig: Var,
    w_hg: Var,
    b_g: Var,
    // === 输出门参数 ===
    w_io: Var,
    w_ho: Var,
    b_o: Var,
    // === 状态节点 ===
    hidden_output: Var,
    cell_output: Var,
    hidden_input: Var,
    cell_input: Var,
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

impl Lstm {
    /// 创建新的 Lstm 层
    pub fn new(
        graph: &Graph,
        input_size: usize,
        hidden_size: usize,
        batch_size: usize,
        name: &str,
    ) -> Result<Self, GraphError> {
        // 创建参数节点
        let w_ii = graph.parameter(&[input_size, hidden_size], Init::Kaiming, &format!("{name}_W_ii"))?;
        let w_hi = graph.parameter(&[hidden_size, hidden_size], Init::Kaiming, &format!("{name}_W_hi"))?;
        let b_i = graph.parameter(&[1, hidden_size], Init::Zeros, &format!("{name}_b_i"))?;

        let w_if = graph.parameter(&[input_size, hidden_size], Init::Kaiming, &format!("{name}_W_if"))?;
        let w_hf = graph.parameter(&[hidden_size, hidden_size], Init::Kaiming, &format!("{name}_W_hf"))?;
        let b_f = graph.parameter(&[1, hidden_size], Init::Ones, &format!("{name}_b_f"))?; // 遗忘门偏置初始化为 1

        let w_ig = graph.parameter(&[input_size, hidden_size], Init::Kaiming, &format!("{name}_W_ig"))?;
        let w_hg = graph.parameter(&[hidden_size, hidden_size], Init::Kaiming, &format!("{name}_W_hg"))?;
        let b_g = graph.parameter(&[1, hidden_size], Init::Zeros, &format!("{name}_b_g"))?;

        let w_io = graph.parameter(&[input_size, hidden_size], Init::Kaiming, &format!("{name}_W_io"))?;
        let w_ho = graph.parameter(&[hidden_size, hidden_size], Init::Kaiming, &format!("{name}_W_ho"))?;
        let b_o = graph.parameter(&[1, hidden_size], Init::Zeros, &format!("{name}_b_o"))?;

        // 创建输入节点
        let input_node = graph.zeros(&[batch_size, input_size])?;

        // 创建 ones 用于偏置广播
        let ones = graph.ones(&[batch_size, 1])?;

        // 创建状态节点和计算图结构
        let (hidden_input, cell_input, hidden_output, cell_output) = {
            let mut g = graph.inner_mut();

            // 创建状态节点
            let h_prev_id = g.new_state_node(&[batch_size, hidden_size], Some(&format!("{name}_h_prev")))?;
            g.set_node_value(h_prev_id, Some(&Tensor::zeros(&[batch_size, hidden_size])))?;

            let c_prev_id = g.new_state_node(&[batch_size, hidden_size], Some(&format!("{name}_c_prev")))?;
            g.set_node_value(c_prev_id, Some(&Tensor::zeros(&[batch_size, hidden_size])))?;

            // === 输入门计算 ===
            let x_ii = g.new_mat_mul_node(input_node.node_id(), w_ii.node_id(), Some(&format!("{name}_x_ii")))?;
            let h_hi = g.new_mat_mul_node(h_prev_id, w_hi.node_id(), Some(&format!("{name}_h_hi")))?;
            let b_i_bc = g.new_mat_mul_node(ones.node_id(), b_i.node_id(), Some(&format!("{name}_b_i_bc")))?;
            let pre_i = g.new_add_node(&[x_ii, h_hi, b_i_bc], Some(&format!("{name}_pre_i")))?;
            let i_gate = g.new_sigmoid_node(pre_i, Some(&format!("{name}_i_gate")))?;

            // === 遗忘门计算 ===
            let x_if = g.new_mat_mul_node(input_node.node_id(), w_if.node_id(), Some(&format!("{name}_x_if")))?;
            let h_hf = g.new_mat_mul_node(h_prev_id, w_hf.node_id(), Some(&format!("{name}_h_hf")))?;
            let b_f_bc = g.new_mat_mul_node(ones.node_id(), b_f.node_id(), Some(&format!("{name}_b_f_bc")))?;
            let pre_f = g.new_add_node(&[x_if, h_hf, b_f_bc], Some(&format!("{name}_pre_f")))?;
            let f_gate = g.new_sigmoid_node(pre_f, Some(&format!("{name}_f_gate")))?;

            // === 候选细胞计算 ===
            let x_ig = g.new_mat_mul_node(input_node.node_id(), w_ig.node_id(), Some(&format!("{name}_x_ig")))?;
            let h_hg = g.new_mat_mul_node(h_prev_id, w_hg.node_id(), Some(&format!("{name}_h_hg")))?;
            let b_g_bc = g.new_mat_mul_node(ones.node_id(), b_g.node_id(), Some(&format!("{name}_b_g_bc")))?;
            let pre_g = g.new_add_node(&[x_ig, h_hg, b_g_bc], Some(&format!("{name}_pre_g")))?;
            let g_gate = g.new_tanh_node(pre_g, Some(&format!("{name}_g_gate")))?;

            // === 输出门计算 ===
            let x_io = g.new_mat_mul_node(input_node.node_id(), w_io.node_id(), Some(&format!("{name}_x_io")))?;
            let h_ho = g.new_mat_mul_node(h_prev_id, w_ho.node_id(), Some(&format!("{name}_h_ho")))?;
            let b_o_bc = g.new_mat_mul_node(ones.node_id(), b_o.node_id(), Some(&format!("{name}_b_o_bc")))?;
            let pre_o = g.new_add_node(&[x_io, h_ho, b_o_bc], Some(&format!("{name}_pre_o")))?;
            let o_gate = g.new_sigmoid_node(pre_o, Some(&format!("{name}_o_gate")))?;

            // === 细胞状态更新 ===
            let f_c = g.new_multiply_node(f_gate, c_prev_id, Some(&format!("{name}_f_c")))?;
            let i_g = g.new_multiply_node(i_gate, g_gate, Some(&format!("{name}_i_g")))?;
            let cell_id = g.new_add_node(&[f_c, i_g], Some(&format!("{name}_c")))?;

            // === 隐藏状态更新 ===
            let tanh_c = g.new_tanh_node(cell_id, Some(&format!("{name}_tanh_c")))?;
            let hidden_id = g.new_multiply_node(o_gate, tanh_c, Some(&format!("{name}_h")))?;

            // === 建立循环连接 ===
            g.connect_recurrent(hidden_id, h_prev_id)?;
            g.connect_recurrent(cell_id, c_prev_id)?;

            // 注册层分组
            g.register_layer_group(
                name,
                "Lstm",
                &format!("{input_size}→{hidden_size}"),
                vec![
                    w_ii.node_id(), w_hi.node_id(), b_i.node_id(),
                    w_if.node_id(), w_hf.node_id(), b_f.node_id(),
                    w_ig.node_id(), w_hg.node_id(), b_g.node_id(),
                    w_io.node_id(), w_ho.node_id(), b_o.node_id(),
                    i_gate, f_gate, g_gate, o_gate,
                    cell_id, hidden_id,
                ],
            );

            // 创建 Var
            let h_prev = Var::new(h_prev_id, graph.inner_rc());
            let c_prev = Var::new(c_prev_id, graph.inner_rc());
            let hidden = Var::new(hidden_id, graph.inner_rc());
            let cell = Var::new(cell_id, graph.inner_rc());

            (h_prev, c_prev, hidden, cell)
        };

        Ok(Self {
            w_ii, w_hi, b_i,
            w_if, w_hf, b_f,
            w_ig, w_hg, b_g,
            w_io, w_ho, b_o,
            hidden_output,
            cell_output,
            hidden_input,
            cell_input,
            input_node,
            graph: graph.clone(),
            input_size,
            hidden_size,
            batch_size,
            name: name.to_string(),
        })
    }

    /// 单步前向传播
    pub fn step(&self, x: &Tensor) -> Result<(&Var, &Var), GraphError> {
        self.input_node.set_value(x)?;
        self.graph.inner_mut().step(self.hidden_output.node_id())?;
        Ok((&self.hidden_output, &self.cell_output))
    }

    /// 完整重置
    pub fn reset(&self) {
        self.graph.inner_mut().reset();
    }

    /// 重置隐藏状态和细胞状态为零
    pub fn reset_state(&self) -> Result<(), GraphError> {
        self.hidden_input.set_value(&Tensor::zeros(&[self.batch_size, self.hidden_size]))?;
        self.cell_input.set_value(&Tensor::zeros(&[self.batch_size, self.hidden_size]))?;
        Ok(())
    }

    // === Getter 方法 ===
    pub fn hidden(&self) -> &Var { &self.hidden_output }
    pub fn cell(&self) -> &Var { &self.cell_output }
    pub fn hidden_input(&self) -> &Var { &self.hidden_input }
    pub fn cell_input(&self) -> &Var { &self.cell_input }
    pub fn input(&self) -> &Var { &self.input_node }

    // 输入门参数
    pub fn w_ii(&self) -> &Var { &self.w_ii }
    pub fn w_hi(&self) -> &Var { &self.w_hi }
    pub fn b_i(&self) -> &Var { &self.b_i }

    // 遗忘门参数
    pub fn w_if(&self) -> &Var { &self.w_if }
    pub fn w_hf(&self) -> &Var { &self.w_hf }
    pub fn b_f(&self) -> &Var { &self.b_f }

    // 候选细胞参数
    pub fn w_ig(&self) -> &Var { &self.w_ig }
    pub fn w_hg(&self) -> &Var { &self.w_hg }
    pub fn b_g(&self) -> &Var { &self.b_g }

    // 输出门参数
    pub fn w_io(&self) -> &Var { &self.w_io }
    pub fn w_ho(&self) -> &Var { &self.w_ho }
    pub fn b_o(&self) -> &Var { &self.b_o }

    pub fn input_size(&self) -> usize { self.input_size }
    pub fn hidden_size(&self) -> usize { self.hidden_size }
    pub fn batch_size(&self) -> usize { self.batch_size }
    pub fn graph(&self) -> &Graph { &self.graph }
}

impl Module for Lstm {
    fn parameters(&self) -> Vec<Var> {
        vec![
            self.w_ii.clone(), self.w_hi.clone(), self.b_i.clone(),
            self.w_if.clone(), self.w_hf.clone(), self.b_f.clone(),
            self.w_ig.clone(), self.w_hg.clone(), self.b_g.clone(),
            self.w_io.clone(), self.w_ho.clone(), self.b_o.clone(),
        ]
    }
}

// 单元测试位于 src/nn/tests/layer_lstm.rs
