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
 * 展开式设计（无缓存）：
 * - 每次 forward 根据输入序列长度动态展开时间步
 * - 使用 Select 节点从序列中提取每个时间步
 * - BPTT 通过图的反向传播自动完成
 * - 不维护缓存，每次 forward 都创建新节点
 */

use crate::nn::graph::NodeGroupContext;
use crate::nn::var::ops::{VarActivationOps, VarMatrixOps, VarShapeOps};
use crate::nn::{Graph, GraphError, Init, IntoVar, Module, Var};

/// Gru (门控循环单元) 层 - 展开式设计（无缓存）
///
/// `PyTorch` 风格的 GRU 层，包含重置门、更新门和候选状态。
/// 比 LSTM 更简单高效（2 个门 vs 4 个门）。
///
/// # 输入/输出形状
/// - 输入：[`batch_size`, `seq_len`, `input_size`]
/// - 输出：[`batch_size`, `hidden_size`]（最后一个时间步）
///
/// # 无缓存设计
/// 每次 `forward` 都重新创建展开节点，开销可忽略（只是创建节点引用）。
/// 这样节点会在不再引用时自动释放，避免内存泄漏。
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
    /// 分组实例 ID（用于可视化 cluster）
    instance_id: usize,
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
        // 如果 graph 有 model_name，自动拼接为 "ModelName/layer_name" 格式
        let full_name = match graph.model_name() {
            Some(model) => format!("{model}/{name}"),
            None => name.to_string(),
        };

        // === 重置门参数 ===
        let w_ir = graph.parameter(
            &[input_size, hidden_size],
            Init::Kaiming,
            &format!("{full_name}_W_ir"),
        )?;
        let w_hr = graph.parameter(
            &[hidden_size, hidden_size],
            Init::Kaiming,
            &format!("{full_name}_W_hr"),
        )?;
        let b_r = graph.parameter(&[1, hidden_size], Init::Zeros, &format!("{full_name}_b_r"))?;

        // === 更新门参数 ===
        let w_iz = graph.parameter(
            &[input_size, hidden_size],
            Init::Kaiming,
            &format!("{full_name}_W_iz"),
        )?;
        let w_hz = graph.parameter(
            &[hidden_size, hidden_size],
            Init::Kaiming,
            &format!("{full_name}_W_hz"),
        )?;
        let b_z = graph.parameter(&[1, hidden_size], Init::Zeros, &format!("{full_name}_b_z"))?;

        // === 候选状态参数 ===
        let w_in = graph.parameter(
            &[input_size, hidden_size],
            Init::Kaiming,
            &format!("{full_name}_W_in"),
        )?;
        let w_hn = graph.parameter(
            &[hidden_size, hidden_size],
            Init::Kaiming,
            &format!("{full_name}_W_hn"),
        )?;
        let b_n = graph.parameter(&[1, hidden_size], Init::Zeros, &format!("{full_name}_b_n"))?;

        // 注册折叠渲染元信息（仅保留折叠所需的最小信息）
        // GRU 每个时间步的节点数：20
        // - select: 1
        // - 重置门: 2 matmul + 2 add + 1 sigmoid = 5
        // - 更新门: 2 matmul + 2 add + 1 sigmoid = 5
        // - 候选状态: 2 matmul + 1 multiply + 2 add + 1 tanh = 6
        // - 隐藏更新: 1 subtract + 1 multiply + 1 add = 3
        // 总计: 1+5+5+6+3 = 20
        graph
            .inner_mut()
            .register_recurrent_folding_meta(&full_name, 20);

        let instance_id = graph.inner_mut().next_node_group_instance_id();

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
            name: full_name,
            instance_id,
        })
    }

    /// 前向传播
    ///
    /// 自动展开所有时间步，返回最后一个时间步的隐藏状态。
    ///
    /// # 无缓存设计
    /// 每次 forward 都重新创建展开节点。由于节点生命周期由引用计数管理，
    /// 不再引用的节点会自动释放，因此无需缓存。开销可忽略（只是创建节点引用）。
    ///
    /// # 参数
    /// - `x`: 输入 Var，形状 [`batch_size`, `seq_len`, `input_size`]
    ///
    /// # 返回
    /// 最后一个时间步的隐藏状态 Var，形状 [`batch_size`, `hidden_size`]
    pub fn forward(&self, x: impl IntoVar) -> Result<Var, GraphError> {
        let (x, seq_len) = self.validate_input(x)?;
        self.unroll(&x, seq_len, false)
    }

    /// 前向传播（返回所有时间步）
    ///
    /// 与 `forward` 相同，但返回所有时间步的隐藏状态而非仅最后一步。
    ///
    /// # 返回
    /// 所有时间步的隐藏状态 Var，形状 [`batch_size`, `seq_len`, `hidden_size`]
    pub fn forward_seq(&self, x: impl IntoVar) -> Result<Var, GraphError> {
        let (x, seq_len) = self.validate_input(x)?;
        self.unroll(&x, seq_len, true)
    }

    /// 验证输入并返回 (x_var, seq_len)
    fn validate_input(&self, x: impl IntoVar) -> Result<(Var, usize), GraphError> {
        let x = x
            .into_var(&self.w_ir.get_graph())
            .expect("Gru 输入转换失败");
        let value = x
            .value()?
            .ok_or_else(|| GraphError::ComputationError("Gru.forward 需要输入有值".to_string()))?;
        let shape = value.shape();

        if shape.len() != 3 {
            return Err(GraphError::InvalidOperation(format!(
                "Gru.forward 需要 3D 输入 [batch, seq_len, input], 实际: {shape:?}"
            )));
        }

        let (_batch_size, seq_len, input_size) = (shape[0], shape[1], shape[2]);

        if input_size != self.input_size {
            return Err(GraphError::InvalidOperation(format!(
                "input_size 不匹配: 期望 {}, 实际 {}",
                self.input_size, input_size
            )));
        }

        Ok((x, seq_len))
    }

    /// 展开 GRU 时间步
    ///
    /// 计算逻辑与可视化信息收集完全分离：
    /// - 此方法只做计算 + 记录最少的必要信息（4 个节点 ID + 1 个数值）
    /// - 完整的分组信息在 `save_visualization` 时惰性推断
    fn unroll(&self, x: &Var, seq_len: usize, return_sequences: bool) -> Result<Var, GraphError> {
        // 分组上下文：自动标记 unroll 期间创建的节点
        let desc = format!(
            "GRU: [?, {}] → [?, {}] (×{} steps)",
            self.input_size, self.hidden_size, seq_len
        );
        let _guard = NodeGroupContext::for_recurrent(x, "GRU", self.instance_id, &self.name, &desc);
        // 后补标签给参数节点
        _guard.tag_existing(&self.w_ir);
        _guard.tag_existing(&self.w_hr);
        _guard.tag_existing(&self.b_r);
        _guard.tag_existing(&self.w_iz);
        _guard.tag_existing(&self.w_hz);
        _guard.tag_existing(&self.b_z);
        _guard.tag_existing(&self.w_in);
        _guard.tag_existing(&self.w_hn);
        _guard.tag_existing(&self.b_n);

        // 创建初始隐藏状态（ZerosLike：根据 x 的 batch_size 动态生成）
        let h0 = self.graph.zeros_like(x, &[self.hidden_size], None)?;
        _guard.tag_existing(&h0); // 初始状态纳入分组
        let init_state_node_ids = vec![h0.node_id()]; // GRU 只有一个初始状态
        let mut h = h0;

        // 记录第一个时间步的信息（用于折叠渲染）
        let mut first_step_start_id = None;
        let mut repr_output_node_ids = Vec::new();

        // return_sequences 模式下收集所有时间步
        let mut all_h: Vec<Var> = if return_sequences {
            Vec::with_capacity(seq_len)
        } else {
            Vec::new()
        };

        // 展开所有时间步
        for t in 0..seq_len {
            // 步骤 1..N-1 标记为隐藏（可视化时折叠）
            if t > 0 {
                _guard.set_hidden(true);
            }

            // 选择第 t 个时间步: x_t = x[:, t, :] -> [batch, input_size]
            let x_t = x.select(1, t)?;

            // 记录第一个时间步的起始节点 ID
            if t == 0 {
                first_step_start_id = Some(x_t.node_id());
            }

            // === 重置门 ===
            let x_ir = x_t.matmul(&self.w_ir)?;
            let h_hr = h.matmul(&self.w_hr)?;
            let r_gate = (&x_ir + &h_hr + &self.b_r).sigmoid();

            // === 更新门 ===
            let x_iz = x_t.matmul(&self.w_iz)?;
            let h_hz = h.matmul(&self.w_hz)?;
            let z_gate = (&x_iz + &h_hz + &self.b_z).sigmoid();

            // === 候选状态 ===
            let x_in = x_t.matmul(&self.w_in)?;
            let h_hn = h.matmul(&self.w_hn)?;
            let r_h_hn = &r_gate * &h_hn;
            let n_gate = (&x_in + &r_h_hn + &self.b_n).tanh();

            // === 更新隐藏状态 ===
            // h_t = n_t + z_t ⊙ (h - n_t)
            let h_minus_n = &h - &n_gate;
            let z_diff = &z_gate * &h_minus_n;
            h = &n_gate + &z_diff;

            if return_sequences {
                all_h.push(h.clone());
            }

            // 记录第一个时间步的输出节点 ID（GRU 只有 h）
            if t == 0 {
                repr_output_node_ids.push(h.node_id());
            }
        }

        // 更新折叠渲染元信息
        use crate::nn::graph::RecurrentUnrollInfo;
        self.graph.inner_mut().update_recurrent_folding_info(
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

        if return_sequences {
            let refs: Vec<&Var> = all_h.iter().collect();
            Var::stack(&refs, 1)
        } else {
            Ok(h)
        }
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

    /// 从已有参数 Var 创建 Gru 层（基因组 NodeLevel 重建路径专用）
    ///
    /// parents 顺序：[w_ir, w_hr, b_r, w_iz, w_hz, b_z, w_in, w_hn, b_n]
    #[allow(clippy::too_many_arguments)]
    pub fn from_vars(
        w_ir: Var,
        w_hr: Var,
        b_r: Var,
        w_iz: Var,
        w_hz: Var,
        b_z: Var,
        w_in: Var,
        w_hn: Var,
        b_n: Var,
        input_size: usize,
        hidden_size: usize,
    ) -> Self {
        let graph = w_ir.get_graph();
        let name = "gru_rebuilt".to_string();
        graph.inner_mut().register_recurrent_folding_meta(&name, 20);
        let instance_id = graph.inner_mut().next_node_group_instance_id();
        Self {
            w_ir,
            w_hr,
            b_r,
            w_iz,
            w_hz,
            b_z,
            w_in,
            w_hn,
            b_n,
            graph,
            input_size,
            hidden_size,
            name,
            instance_id,
        }
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
