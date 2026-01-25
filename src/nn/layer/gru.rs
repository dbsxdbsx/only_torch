/*
 * @Author       : 老董
 * @Date         : 2026-01-21
 * @Description  : Gru (门控循环单元) 层 - 展开式设计（PyTorch 风格 API）
 *
 * 公式:
 *   r_t = σ(x_t @ W_ir + h_{t-1} @ W_hr + b_r)     # 重置门
 *   z_t = σ(x_t @ W_iz + h_{t-1} @ W_hz + b_z)     # 更新门
 *   n_t = tanh(x_t @ W_in + r_t ⊙ (h_{t-1} @ W_hn) + b_n)  # 候选状态
 *   h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}         # 隐藏状态
 *
 * 与 PyTorch nn.GRU 对齐:
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
use crate::nn::{Graph, GraphError, Init, Module, NodeId, Var};
use std::cell::RefCell;
use std::collections::HashMap;

/// Gru (门控循环单元) 层 - 展开式设计
///
/// `PyTorch` 风格的 GRU 层，包含重置门、更新门和候选状态。
/// 比 LSTM 更简单高效（2 个门 vs 4 个门）。
///
/// # 输入/输出形状
/// - 输入：[`batch_size`, `seq_len`, `input_size`]
/// - 输出：[`batch_size`, `hidden_size`]（最后一个时间步）
///
/// # 使用示例
/// ```ignore
/// // 定义模型
/// pub struct MyGruModel {
///     gru: Gru,
///     fc: Linear,
///     state: ModelState,
/// }
///
/// impl MyGruModel {
///     pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
///         self.state.forward(x, |input| {
///             let h = self.gru.forward(input)?;
///             Ok(self.fc.forward(&h))
///         })
///     }
/// }
/// ```
pub struct Gru {
    // === 重置门参数 ===
    w_ir: Var, // [input_size, hidden_size]
    w_hr: Var, // [hidden_size, hidden_size]
    b_r: Var,  // [1, hidden_size]
    // === 更新门参数 ===
    w_iz: Var,
    w_hz: Var,
    b_z: Var,
    // === 候选状态参数 ===
    w_in: Var,
    w_hn: Var,
    b_n: Var,
    // === Graph 和配置 ===
    graph: Graph,
    input_size: usize,
    hidden_size: usize,
    #[allow(dead_code)]
    name: String,
    /// 按 (batch_size, seq_len) 缓存的展开结构 -> 实际输出节点 ID
    /// 注意：必须同时用 batch_size 和 seq_len 作为 key，
    /// 因为 zeros_like 创建的初始状态节点依赖输入的 batch 维度
    unroll_cache: RefCell<HashMap<(usize, usize), NodeId>>,
}

impl Gru {
    /// 创建新的 Gru 层
    ///
    /// # 参数
    /// - `graph`: 计算图句柄
    /// - `input_size`: 输入特征维度
    /// - `hidden_size`: 隐藏状态维度
    /// - `name`: 层名称前缀
    ///
    /// # 返回
    /// Gru 层实例
    pub fn new(
        graph: &Graph,
        input_size: usize,
        hidden_size: usize,
        name: &str,
    ) -> Result<Self, GraphError> {
        // === 重置门参数 ===
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

        // === 更新门参数 ===
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

        // === 候选状态参数 ===
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

        // 注册循环层元信息（惰性收集：只在可视化时才根据此信息推断完整分组）
        // GRU 每个时间步的节点数：20
        // - select: 1
        // - 重置门: 2 matmul + 2 add + 1 sigmoid = 5
        // - 更新门: 2 matmul + 2 add + 1 sigmoid = 5
        // - 候选状态: 2 matmul + 1 multiply + 2 add + 1 tanh = 6
        // - 隐藏更新: 1 subtract + 1 multiply + 1 add = 3
        // 总计: 1+5+5+6+3 = 20
        graph.inner_mut().register_recurrent_layer_meta(
            name,
            "GRU",
            &format!("[?, {input_size}] → [?, {hidden_size}]"),
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
            ],
            20, // nodes_per_step
        );

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
            graph: graph.clone(),
            input_size,
            hidden_size,
            name: name.to_string(),
            unroll_cache: RefCell::new(HashMap::new()),
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
    ///     let h = self.gru.forward(input)?;
    ///     Ok(self.fc.forward(&h))
    /// })
    /// ```
    pub fn forward(&self, x: &Var) -> Result<Var, GraphError> {
        // 使用实际值的形状（支持动态 batch）
        let value = x
            .value()?
            .ok_or_else(|| GraphError::ComputationError("Gru.forward 需要输入有值".to_string()))?;
        let shape = value.shape();

        if shape.len() != 3 {
            return Err(GraphError::InvalidOperation(format!(
                "Gru.forward 需要 3D 输入 [batch, seq_len, input], 实际: {shape:?}"
            )));
        }

        let (batch_size, seq_len, input_size) = (shape[0], shape[1], shape[2]);

        // 验证输入维度
        if input_size != self.input_size {
            return Err(GraphError::InvalidOperation(format!(
                "input_size 不匹配: 期望 {}, 实际 {}",
                self.input_size, input_size
            )));
        }

        // 获取或创建此 (batch_size, seq_len) 的展开结构
        // 注意：必须同时用 batch_size 和 seq_len 作为缓存 key，
        // 因为 zeros_like 创建的初始状态节点依赖输入的 batch 维度
        let cache_key = (batch_size, seq_len);
        let h = {
            let cache = self.unroll_cache.borrow();
            if let Some(&cached_id) = cache.get(&cache_key) {
                // 缓存命中：重新计算该节点
                drop(cache);
                self.graph.inner_mut().forward(cached_id)?;
                Var::new(cached_id, self.graph.inner_rc())
            } else {
                // 缓存未命中：创建新的展开结构
                drop(cache);
                let h = self.unroll(x, seq_len)?;
                let h_id = h.node_id();
                // 触发前向计算
                self.graph.inner_mut().forward(h_id)?;
                self.unroll_cache.borrow_mut().insert(cache_key, h_id);
                h
            }
        };

        Ok(h)
    }

    /// 展开 GRU 时间步
    ///
    /// 计算逻辑与可视化信息收集完全分离：
    /// - 此方法只做计算 + 记录最少的必要信息（4 个节点 ID + 1 个数值）
    /// - 完整的分组信息在 `save_visualization` 时惰性推断
    fn unroll(&self, x: &Var, seq_len: usize) -> Result<Var, GraphError> {
        // 创建初始隐藏状态（ZerosLike：根据 x 的 batch_size 动态生成）
        let h0 = self.graph.zeros_like(x, &[self.hidden_size], None)?;
        let init_state_node_ids = vec![h0.node_id()]; // GRU 只有一个初始状态
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

            // === 重置门 ===
            // r_t = σ(x_t @ W_ir + h @ W_hr + b_r)
            let x_ir = x_t.matmul(&self.w_ir)?;
            let h_hr = h.matmul(&self.w_hr)?;
            let r_gate = (&x_ir + &h_hr + &self.b_r).sigmoid();

            // === 更新门 ===
            // z_t = σ(x_t @ W_iz + h @ W_hz + b_z)
            let x_iz = x_t.matmul(&self.w_iz)?;
            let h_hz = h.matmul(&self.w_hz)?;
            let z_gate = (&x_iz + &h_hz + &self.b_z).sigmoid();

            // === 候选状态 ===
            // n_t = tanh(x_t @ W_in + r_t ⊙ (h @ W_hn) + b_n)
            let x_in = x_t.matmul(&self.w_in)?;
            let h_hn = h.matmul(&self.w_hn)?;
            let r_h_hn = &r_gate * &h_hn;
            let n_gate = (&x_in + &r_h_hn + &self.b_n).tanh();

            // === 更新隐藏状态 ===
            // h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h
            // 重写为: h_t = n_t + z_t ⊙ (h - n_t) 以减少计算
            let h_minus_n = &h - &n_gate;
            let z_diff = &z_gate * &h_minus_n;
            h = &n_gate + &z_diff;

            // 记录第一个时间步的输出节点 ID（GRU 只有 h）
            if t == 0 {
                repr_output_node_ids.push(h.node_id());
            }
        }

        // 更新循环层的展开信息（只记录几个节点 ID + 1 个数值，几乎零开销）
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

    // === Getter 方法 ===

    /// 获取重置门权重 `W_ir`
    pub const fn w_ir(&self) -> &Var {
        &self.w_ir
    }
    /// 获取重置门权重 `W_hr`
    pub const fn w_hr(&self) -> &Var {
        &self.w_hr
    }
    /// 获取重置门偏置 `b_r`
    pub const fn b_r(&self) -> &Var {
        &self.b_r
    }

    /// 获取更新门权重 `W_iz`
    pub const fn w_iz(&self) -> &Var {
        &self.w_iz
    }
    /// 获取更新门权重 `W_hz`
    pub const fn w_hz(&self) -> &Var {
        &self.w_hz
    }
    /// 获取更新门偏置 `b_z`
    pub const fn b_z(&self) -> &Var {
        &self.b_z
    }

    /// 获取候选状态权重 `W_in`
    pub const fn w_in(&self) -> &Var {
        &self.w_in
    }
    /// 获取候选状态权重 `W_hn`
    pub const fn w_hn(&self) -> &Var {
        &self.w_hn
    }
    /// 获取候选状态偏置 `b_n`
    pub const fn b_n(&self) -> &Var {
        &self.b_n
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

impl Module for Gru {
    fn parameters(&self) -> Vec<Var> {
        vec![
            // 重置门
            self.w_ir.clone(),
            self.w_hr.clone(),
            self.b_r.clone(),
            // 更新门
            self.w_iz.clone(),
            self.w_hz.clone(),
            self.b_z.clone(),
            // 候选状态
            self.w_in.clone(),
            self.w_hn.clone(),
            self.b_n.clone(),
        ]
    }
}
