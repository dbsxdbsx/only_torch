/*
 * @Author       : 老董
 * @Date         : 2026-01-21
 * @Description  : Lstm (长短期记忆) 层 - 展开式设计（PyTorch 风格 API）
 *
 * 公式:
 *   i_t = σ(x_t @ W_ii + h_{t-1} @ W_hi + b_i)   # 输入门
 *   f_t = σ(x_t @ W_if + h_{t-1} @ W_hf + b_f)   # 遗忘门
 *   g_t = tanh(x_t @ W_ig + h_{t-1} @ W_hg + b_g) # 候选细胞
 *   o_t = σ(x_t @ W_io + h_{t-1} @ W_ho + b_o)   # 输出门
 *   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t              # 细胞状态
 *   h_t = o_t ⊙ tanh(c_t)                        # 隐藏状态
 *
 * 与 PyTorch nn.LSTM 对齐:
 * - 输入: [batch, seq_len, input_size]
 * - 输出: [batch, hidden_size]（最后一个时间步的隐藏状态）
 *
 * 展开式设计：
 * - 每次 forward 根据输入序列长度动态展开时间步
 * - 使用 Select 节点从序列中提取每个时间步
 * - BPTT 通过图的反向传播自动完成
 * - 配合 ModelState 实现 PyTorch 风格的 API
 */

use crate::nn::var_ops::{VarActivationOps, VarMatrixOps, VarShapeOps};
use crate::nn::{Graph, GraphError, Init, Module, Var};

/// Lstm (长短期记忆) 层 - 展开式设计
///
/// `PyTorch` 风格的 LSTM 层，包含输入门、遗忘门、候选细胞和输出门。
///
/// # 输入/输出形状
/// - 输入：[`batch_size`, `seq_len`, `input_size`]
/// - 输出：[`batch_size`, `hidden_size`]（最后一个时间步）
///
/// # 使用示例
/// ```ignore
/// // 定义模型
/// pub struct MyLstmModel {
///     lstm: Lstm,
///     fc: Linear,
///     state: ModelState,
/// }
///
/// impl MyLstmModel {
///     pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
///         self.state.forward(x, |input| {
///             let h = self.lstm.forward(input)?;
///             Ok(self.fc.forward(&h))
///         })
///     }
/// }
/// ```
pub struct Lstm {
    // === 输入门参数 ===
    w_ii: Var, // [input_size, hidden_size]
    w_hi: Var, // [hidden_size, hidden_size]
    b_i: Var,  // [1, hidden_size]
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
    // === Graph 和配置 ===
    graph: Graph,
    input_size: usize,
    hidden_size: usize,
    #[allow(dead_code)]
    name: String,
}

impl Lstm {
    /// 创建新的 Lstm 层
    ///
    /// # 参数
    /// - `graph`: 计算图句柄
    /// - `input_size`: 输入特征维度
    /// - `hidden_size`: 隐藏状态维度
    /// - `name`: 层名称前缀
    ///
    /// # 返回
    /// Lstm 层实例
    pub fn new(
        graph: &Graph,
        input_size: usize,
        hidden_size: usize,
        name: &str,
    ) -> Result<Self, GraphError> {
        // === 输入门参数 ===
        let w_ii = graph.parameter(
            &[input_size, hidden_size],
            Init::Kaiming,
            &format!("{name}_W_ii"),
        )?;
        let w_hi = graph.parameter(
            &[hidden_size, hidden_size],
            Init::Kaiming,
            &format!("{name}_W_hi"),
        )?;
        let b_i = graph.parameter(&[1, hidden_size], Init::Zeros, &format!("{name}_b_i"))?;

        // === 遗忘门参数 ===
        let w_if = graph.parameter(
            &[input_size, hidden_size],
            Init::Kaiming,
            &format!("{name}_W_if"),
        )?;
        let w_hf = graph.parameter(
            &[hidden_size, hidden_size],
            Init::Kaiming,
            &format!("{name}_W_hf"),
        )?;
        // 遗忘门偏置初始化为 1（帮助记忆）
        let b_f = graph.parameter(&[1, hidden_size], Init::Ones, &format!("{name}_b_f"))?;

        // === 候选细胞参数 ===
        let w_ig = graph.parameter(
            &[input_size, hidden_size],
            Init::Kaiming,
            &format!("{name}_W_ig"),
        )?;
        let w_hg = graph.parameter(
            &[hidden_size, hidden_size],
            Init::Kaiming,
            &format!("{name}_W_hg"),
        )?;
        let b_g = graph.parameter(&[1, hidden_size], Init::Zeros, &format!("{name}_b_g"))?;

        // === 输出门参数 ===
        let w_io = graph.parameter(
            &[input_size, hidden_size],
            Init::Kaiming,
            &format!("{name}_W_io"),
        )?;
        let w_ho = graph.parameter(
            &[hidden_size, hidden_size],
            Init::Kaiming,
            &format!("{name}_W_ho"),
        )?;
        let b_o = graph.parameter(&[1, hidden_size], Init::Zeros, &format!("{name}_b_o"))?;

        Ok(Self {
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
    /// self.state.forward(x, |input| {
    ///     let h = self.lstm.forward(input)?;
    ///     Ok(self.fc.forward(&h))
    /// })
    /// ```
    pub fn forward(&self, x: &Var) -> Result<Var, GraphError> {
        // 使用实际值的形状（支持动态 batch）
        let value = x
            .value()?
            .ok_or_else(|| GraphError::ComputationError("Lstm.forward 需要输入有值".to_string()))?;
        let shape = value.shape();

        if shape.len() != 3 {
            return Err(GraphError::InvalidOperation(format!(
                "Lstm.forward 需要 3D 输入 [batch, seq_len, input], 实际: {shape:?}"
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

    /// 展开 LSTM 时间步
    fn unroll(&self, x: &Var, seq_len: usize) -> Result<Var, GraphError> {
        // 创建初始状态（ZerosLike：根据 x 的 batch_size 动态生成）
        // 注意：每次 forward 都创建新的 ZerosLike 节点，确保与当前输入的 batch_size 匹配
        // 这对于变长序列处理（不同桶有不同 batch_size）是必需的
        let h0 = self.graph.zeros_like(x, &[self.hidden_size], None)?;
        let c0 = self.graph.zeros_like(x, &[self.hidden_size], None)?;
        let mut h = h0;
        let mut c = c0;

        // 展开所有时间步
        for t in 0..seq_len {
            // 选择第 t 个时间步: x_t = x[:, t, :] -> [batch, input_size]
            let x_t = x.select(1, t)?;

            // === 输入门 ===
            // i_t = σ(x_t @ W_ii + h @ W_hi + b_i)
            let x_ii = x_t.matmul(&self.w_ii)?;
            let h_hi = h.matmul(&self.w_hi)?;
            let i_gate = (&x_ii + &h_hi + &self.b_i).sigmoid();

            // === 遗忘门 ===
            // f_t = σ(x_t @ W_if + h @ W_hf + b_f)
            let x_if = x_t.matmul(&self.w_if)?;
            let h_hf = h.matmul(&self.w_hf)?;
            let f_gate = (&x_if + &h_hf + &self.b_f).sigmoid();

            // === 候选细胞 ===
            // g_t = tanh(x_t @ W_ig + h @ W_hg + b_g)
            let x_ig = x_t.matmul(&self.w_ig)?;
            let h_hg = h.matmul(&self.w_hg)?;
            let g_gate = (&x_ig + &h_hg + &self.b_g).tanh();

            // === 输出门 ===
            // o_t = σ(x_t @ W_io + h @ W_ho + b_o)
            let x_io = x_t.matmul(&self.w_io)?;
            let h_ho = h.matmul(&self.w_ho)?;
            let o_gate = (&x_io + &h_ho + &self.b_o).sigmoid();

            // === 更新细胞状态 ===
            // c_t = f_t ⊙ c + i_t ⊙ g_t
            c = &f_gate * &c + &i_gate * &g_gate;

            // === 更新隐藏状态 ===
            // h_t = o_t ⊙ tanh(c_t)
            h = &o_gate * &c.tanh();
        }

        Ok(h)
    }

    // === Getter 方法 ===

    /// 获取输入门权重 `W_ii`
    pub const fn w_ii(&self) -> &Var {
        &self.w_ii
    }
    /// 获取输入门权重 `W_hi`
    pub const fn w_hi(&self) -> &Var {
        &self.w_hi
    }
    /// 获取输入门偏置 `b_i`
    pub const fn b_i(&self) -> &Var {
        &self.b_i
    }

    /// 获取遗忘门权重 `W_if`
    pub const fn w_if(&self) -> &Var {
        &self.w_if
    }
    /// 获取遗忘门权重 `W_hf`
    pub const fn w_hf(&self) -> &Var {
        &self.w_hf
    }
    /// 获取遗忘门偏置 `b_f`
    pub const fn b_f(&self) -> &Var {
        &self.b_f
    }

    /// 获取候选细胞权重 `W_ig`
    pub const fn w_ig(&self) -> &Var {
        &self.w_ig
    }
    /// 获取候选细胞权重 `W_hg`
    pub const fn w_hg(&self) -> &Var {
        &self.w_hg
    }
    /// 获取候选细胞偏置 `b_g`
    pub const fn b_g(&self) -> &Var {
        &self.b_g
    }

    /// 获取输出门权重 `W_io`
    pub const fn w_io(&self) -> &Var {
        &self.w_io
    }
    /// 获取输出门权重 `W_ho`
    pub const fn w_ho(&self) -> &Var {
        &self.w_ho
    }
    /// 获取输出门偏置 `b_o`
    pub const fn b_o(&self) -> &Var {
        &self.b_o
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

impl Module for Lstm {
    fn parameters(&self) -> Vec<Var> {
        vec![
            // 输入门
            self.w_ii.clone(),
            self.w_hi.clone(),
            self.b_i.clone(),
            // 遗忘门
            self.w_if.clone(),
            self.w_hf.clone(),
            self.b_f.clone(),
            // 候选细胞
            self.w_ig.clone(),
            self.w_hg.clone(),
            self.b_g.clone(),
            // 输出门
            self.w_io.clone(),
            self.w_ho.clone(),
            self.b_o.clone(),
        ]
    }
}
