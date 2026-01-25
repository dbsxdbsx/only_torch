/*
 * @Author       : 老董
 * @Date         : 2026-01-21
 * @Description  : Rnn (循环神经网络) 层 - 展开式设计（PyTorch 风格 API）
 *
 * 公式: h_t = tanh(x_t @ W_ih + h_{t-1} @ W_hh + b_h)
 *
 * 与 PyTorch nn.RNN 对齐:
 * - 输入: [batch, seq_len, input_size]
 * - 输出: [batch, hidden_size]（最后一个时间步的隐藏状态）
 * - W_ih: [input_size, hidden_size]
 * - W_hh: [hidden_size, hidden_size]
 * - b_h: [1, hidden_size]
 *
 * 展开式设计：
 * - 每次 forward 根据输入序列长度动态展开时间步
 * - 使用 Select 节点从序列中提取每个时间步
 * - BPTT 通过图的反向传播自动完成
 * - 配合 ModelState 实现 PyTorch 风格的 API
 *
 * API 设计：
 * - forward(&Var) 接收 Var，与 Linear 等层保持一致
 * - 缓存由 ModelState 统一管理，层本身不维护缓存
 */

use crate::nn::var_ops::{VarActivationOps, VarMatrixOps, VarShapeOps};
use crate::nn::{Graph, GraphError, Init, Module, Var};

/// Rnn (循环神经网络) 层 - 展开式设计
///
/// `PyTorch` 风格的 RNN 层：`h_t = tanh(x @ W_ih + h_{t-1} @ W_hh + b_h)`
///
/// # 输入/输出形状
/// - 输入：[`batch_size`, `seq_len`, `input_size`]
/// - 输出：[`batch_size`, `hidden_size`]（最后一个时间步）
///
/// # 动态 Batch 支持
/// 初始隐藏状态使用 Input 节点（支持动态 batch），
/// 允许同一个 RNN 层处理不同 `batch_size` 的输入。
///
/// # 使用示例
/// ```ignore
/// // 定义模型
/// pub struct MyRnnModel {
///     rnn: Rnn,
///     fc: Linear,
///     state: ModelState,
/// }
///
/// impl MyRnnModel {
///     pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
///         self.state.try_forward(x, |input| {
///             let h = self.rnn.forward(input)?;
///             Ok(self.fc.forward(&h))
///         })
///     }
/// }
/// ```
pub struct Rnn {
    /// 输入到隐藏权重 `W_ih`: [`input_size`, `hidden_size`]
    w_ih: Var,
    /// 隐藏到隐藏权重 `W_hh`: [`hidden_size`, `hidden_size`]
    w_hh: Var,
    /// 隐藏层偏置 `b_h`: [1, `hidden_size`]
    b_h: Var,
    /// Graph 引用
    graph: Graph,
    /// 配置
    input_size: usize,
    hidden_size: usize,
    #[allow(dead_code)]
    name: String,
}

impl Rnn {
    /// 创建新的 Rnn 层
    ///
    /// # 参数
    /// - `graph`: 计算图句柄
    /// - `input_size`: 输入特征维度
    /// - `hidden_size`: 隐藏状态维度
    /// - `name`: 层名称前缀
    ///
    /// # 返回
    /// Rnn 层实例
    pub fn new(
        graph: &Graph,
        input_size: usize,
        hidden_size: usize,
        name: &str,
    ) -> Result<Self, GraphError> {
        // 创建参数节点
        let w_ih = graph.parameter(
            &[input_size, hidden_size],
            Init::Kaiming,
            &format!("{name}_W_ih"),
        )?;

        let w_hh = graph.parameter(
            &[hidden_size, hidden_size],
            Init::Kaiming,
            &format!("{name}_W_hh"),
        )?;

        let b_h = graph.parameter(&[1, hidden_size], Init::Zeros, &format!("{name}_b_h"))?;
        // 注册循环层元信息（惰性收集：只在可视化时才根据此信息推断完整分组）
        // RNN 每个时间步的节点数：6 (select, matmul_xw, matmul_hw, add1, add2, tanh)
        graph.inner_mut().register_recurrent_layer_meta(
            name,
            "RNN",
            &format!("[?, {input_size}] → [?, {hidden_size}]"),
            vec![w_ih.node_id(), w_hh.node_id(), b_h.node_id()],
            6, // nodes_per_step
        );

        Ok(Self {
            w_ih,
            w_hh,
            b_h,
            graph: graph.clone(),
            input_size,
            hidden_size,
            name: name.to_string(),
        })
    }

    /// 前向传播
    ///
    /// 自动展开所有时间步，返回最后一个时间步的隐藏状态。
    ///
    /// # 参数
    /// - `x`: 输入 Var，形状 [`batch_size`, `seq_len`, `input_size`]
    ///
    /// # 返回
    /// 最后一个时间步的隐藏状态 Var，形状 [`batch_size`, `hidden_size`]
    ///
    /// # 示例
    /// ```ignore
    /// // 与 ModelState 配合使用（推荐）
    /// self.state.try_forward(x, |input| {
    ///     let h = self.rnn.forward(input)?;
    ///     Ok(self.fc.forward(&h))
    /// })
    /// ```
    pub fn forward(&self, x: &Var) -> Result<Var, GraphError> {
        // 使用实际值的形状（支持动态 batch）
        let value = x
            .value()?
            .ok_or_else(|| GraphError::ComputationError("Rnn.forward 需要输入有值".to_string()))?;
        let shape = value.shape();

        if shape.len() != 3 {
            return Err(GraphError::InvalidOperation(format!(
                "Rnn.forward 需要 3D 输入 [batch, seq_len, input], 实际: {shape:?}"
            )));
        }

        let (_, seq_len, input_size) = (shape[0], shape[1], shape[2]);

        // 验证输入维度
        if input_size != self.input_size {
            return Err(GraphError::InvalidOperation(format!(
                "input_size 不匹配: 期望 {}, 实际 {}",
                self.input_size, input_size
            )));
        }

        // 展开所有时间步
        self.unroll(x, seq_len)
    }

    /// 展开 RNN 时间步
    ///
    /// 计算逻辑与可视化信息收集完全分离：
    /// - 此方法只做计算 + 记录最少的必要信息（4 个节点 ID + 1 个数值）
    /// - 完整的分组信息在 `save_visualization` 时惰性推断
    fn unroll(&self, x: &Var, seq_len: usize) -> Result<Var, GraphError> {
        // 创建初始隐藏状态（ZerosLike：根据 x 的 batch_size 动态生成）
        let h0 = self.graph.zeros_like(x, &[self.hidden_size], None)?;
        let init_state_node_ids = vec![h0.node_id()];
        let mut h = h0;

        // 记录第一个时间步的信息（用于惰性推断）
        let mut first_step_start_id = None;
        let mut repr_output_node_ids = Vec::new();

        // 展开所有时间步
        for t in 0..seq_len {
            // 选择第 t 个时间步: x_t = x[:, t, :] -> [batch, input_size]
            let x_t = x.select(1, t)?;

            // 记录第一个时间步的起始节点 ID
            if t == 0 {
                first_step_start_id = Some(x_t.node_id());
            }

            // h_new = tanh(x_t @ W_ih + h @ W_hh + b_h)
            let xw = x_t.matmul(&self.w_ih)?;
            let hw = h.matmul(&self.w_hh)?;
            let sum1 = &xw + &hw;
            let sum2 = &sum1 + &self.b_h;
            h = sum2.tanh();

            // 记录第一个时间步的输出节点 ID（RNN 只有 h）
            if t == 0 {
                repr_output_node_ids.push(h.node_id());
            }
        }

        // 更新循环层的展开信息（只记录 5 个节点 ID + 1 个数值，几乎零开销）
        use crate::nn::graph::RecurrentUnrollInfo;
        self.graph.inner_mut().update_recurrent_layer_unroll_info(
            &self.name,
            RecurrentUnrollInfo {
                steps: seq_len,
                input_node_id: x.node_id(),
                init_state_node_ids,
                first_step_start_id: first_step_start_id.unwrap(),
                repr_output_node_ids,
                real_output_node_id: h.node_id(),
            },
        );

        Ok(h)
    }

    /// 获取 `W_ih` 权重
    pub const fn w_ih(&self) -> &Var {
        &self.w_ih
    }

    /// 获取 `W_hh` 权重
    pub const fn w_hh(&self) -> &Var {
        &self.w_hh
    }

    /// 获取 `b_h` 偏置
    pub const fn b_h(&self) -> &Var {
        &self.b_h
    }

    /// 获取输入维度
    pub const fn input_size(&self) -> usize {
        self.input_size
    }

    /// 获取隐藏维度
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// 获取 Graph 引用
    pub const fn graph(&self) -> &Graph {
        &self.graph
    }
}

impl Module for Rnn {
    fn parameters(&self) -> Vec<Var> {
        vec![self.w_ih.clone(), self.w_hh.clone(), self.b_h.clone()]
    }
}
