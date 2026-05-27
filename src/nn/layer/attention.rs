/*
 * @Author       : 老董
 * @Date         : 2026-02-15
 * @Description  : MultiHeadAttention 层 - Scaled Dot-Product Attention
 *
 * 组合已有节点（MatMul, Softmax, Reshape, Permute）实现多头注意力。
 *
 * 架构:
 *   Q = x @ W_q, K = x @ W_k, V = x @ W_v
 *   → 分头: [N, T, D] → [N, H, T, D/H]
 *   → attention = softmax(Q @ K^T / sqrt(d_k)) @ V
 *   → 合并: [N, H, T, D/H] → [N, T, D]
 *   → output = concat @ W_o
 *
 * # 双 forward API
 *
 * - `forward(q, k, v)`：无 mask 路径，向后兼容。
 * - `forward_masked(q, k, v, attn_mask)`：显式传 mask，约定见下表。
 *
 * | mask shape       | 语义                                   |
 * |------------------|----------------------------------------|
 * | `[T_q, T_k]`     | 全 batch、全 head 共享（causal mask 典型） |
 * | `[N, T_q, T_k]`  | 每 batch 独立（padding mask 典型）       |
 *
 * mask 中 1.0 位置原样保留，0.0 位置在 softmax 前等效减去 1e9 → 权重 ≈ 0。
 *
 * # 工具方法
 *
 * - [`MultiHeadAttention::causal_mask`]：构造下三角 1 上三角 0 的 `[t, t]` mask。
 * - [`MultiHeadAttention::padding_mask`]：根据每行长度构造 `[N, 1, max_len]`
 *   key padding mask（用户可自行 broadcast 到 `[N, T_q, T_k]`）。
 */

use crate::nn::graph::NodeGroupContext;
use crate::nn::{
    Graph, GraphError, IntoVar, Linear, Module, Var, VarActivationOps, VarMatrixOps, VarShapeOps,
};
use crate::tensor::Tensor;

/// 多头注意力层
///
/// # 使用示例
/// ```ignore
/// let attn = MultiHeadAttention::new(&graph, 256, 8, "attn")?;
/// let output = attn.forward(&x, &x, &x);  // 无 mask self-attention
///
/// // 加 causal mask
/// let mask = MultiHeadAttention::causal_mask(&graph, t)?;
/// let output = attn.forward_masked(&x, &x, &x, &mask);
/// ```
pub struct MultiHeadAttention {
    /// Q 投影
    w_q: Linear,
    /// K 投影
    w_k: Linear,
    /// V 投影
    w_v: Linear,
    /// 输出投影
    w_o: Linear,
    /// 输入特征维（Q/K/V 的输入维度，可与 embed_dim 不同）
    input_size: usize,
    /// 嵌入维度
    embed_dim: usize,
    /// 头数
    num_heads: usize,
    /// 每头维度 d_k = embed_dim / num_heads
    head_dim: usize,
    /// 缩放因子 1/sqrt(d_k)
    scale: f32,
    /// 层名称
    name: String,
    /// 分组实例 ID
    instance_id: usize,
}

impl MultiHeadAttention {
    /// 创建多头注意力层（input 与 embed 维度相同的常规自注意力）
    ///
    /// # 参数
    /// - `graph`: 计算图
    /// - `embed_dim`: 嵌入维度（必须能被 num_heads 整除）
    /// - `num_heads`: 注意力头数
    /// - `name`: 层名称前缀
    pub fn new(
        graph: &Graph,
        embed_dim: usize,
        num_heads: usize,
        name: &str,
    ) -> Result<Self, GraphError> {
        Self::new_with_input_size(graph, embed_dim, embed_dim, num_heads, name)
    }

    /// 创建多头注意力层（允许 input_size 与 embed_dim 不同）
    ///
    /// 演化系统的 `CellAttention` 直接接在 input 后时 input_size 可能很小（例如 1），
    /// 此时 W_q/W_k/W_v 的形状为 `[input_size, embed_dim]`、W_o 仍为 `[embed_dim, embed_dim]`。
    ///
    /// # 参数
    /// - `graph`: 计算图
    /// - `input_size`: Q/K/V 的输入特征维（即上游节点最后一维）
    /// - `embed_dim`: attention 内部嵌入维度（必须能被 num_heads 整除）
    /// - `num_heads`: 注意力头数
    /// - `name`: 层名称前缀
    pub fn new_with_input_size(
        graph: &Graph,
        input_size: usize,
        embed_dim: usize,
        num_heads: usize,
        name: &str,
    ) -> Result<Self, GraphError> {
        assert!(
            embed_dim % num_heads == 0,
            "MultiHeadAttention: embed_dim={embed_dim} 必须能被 num_heads={num_heads} 整除"
        );

        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let w_q = Linear::new(graph, input_size, embed_dim, true, &format!("{name}_q"))?;
        let w_k = Linear::new(graph, input_size, embed_dim, true, &format!("{name}_k"))?;
        let w_v = Linear::new(graph, input_size, embed_dim, true, &format!("{name}_v"))?;
        let w_o = Linear::new(graph, embed_dim, embed_dim, true, &format!("{name}_o"))?;

        let instance_id = graph.inner_mut().next_node_group_instance_id();

        Ok(Self {
            w_q,
            w_k,
            w_v,
            w_o,
            input_size,
            embed_dim,
            num_heads,
            head_dim,
            scale,
            name: name.to_string(),
            instance_id,
        })
    }

    /// 从已有参数 Var 构造（演化 NodeLevel rebuild 路径专用）
    ///
    /// 不创建新参数节点，直接复用传入的 8 个 Var。
    ///
    /// # 参数顺序
    /// 与 `CellAttention` descriptor 的 parents 顺序一致：
    /// `[w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o]`。
    ///
    /// # 形状约束
    /// - `w_q / w_k / w_v`: `[input_size, embed_dim]`
    /// - `w_o`: `[embed_dim, embed_dim]`
    /// - 所有 `b_*`: `[1, embed_dim]`
    #[allow(clippy::too_many_arguments)]
    pub fn from_vars(
        input_size: usize,
        embed_dim: usize,
        num_heads: usize,
        w_q: Var,
        b_q: Var,
        w_k: Var,
        b_k: Var,
        w_v: Var,
        b_v: Var,
        w_o: Var,
        b_o: Var,
    ) -> Self {
        assert!(
            embed_dim % num_heads == 0,
            "MultiHeadAttention::from_vars: embed_dim={embed_dim} 必须能被 num_heads={num_heads} 整除"
        );

        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let w_q_layer = Linear::from_vars(w_q, Some(b_q), input_size, embed_dim);
        let w_k_layer = Linear::from_vars(w_k, Some(b_k), input_size, embed_dim);
        let w_v_layer = Linear::from_vars(w_v, Some(b_v), input_size, embed_dim);
        let w_o_layer = Linear::from_vars(w_o, Some(b_o), embed_dim, embed_dim);

        let graph = w_q_layer.weights().get_graph();
        let instance_id = graph.inner_mut().next_node_group_instance_id();

        Self {
            w_q: w_q_layer,
            w_k: w_k_layer,
            w_v: w_v_layer,
            w_o: w_o_layer,
            input_size,
            embed_dim,
            num_heads,
            head_dim,
            scale,
            name: "attention_rebuilt".to_string(),
            instance_id,
        }
    }

    /// 前向传播（无 mask） — Scaled Dot-Product Multi-Head Attention
    ///
    /// # 参数
    /// - `query`: 查询 [N, T_q, input_size]
    /// - `key`: 键 [N, T_k, input_size]
    /// - `value`: 值 [N, T_k, input_size]
    ///
    /// # 返回
    /// - output: [N, T_q, embed_dim]
    ///
    /// # 说明
    /// self-attention: `forward(&x, &x, &x)`，
    /// cross-attention: `forward(&q, &kv, &kv)`。
    pub fn forward(&self, query: impl IntoVar, key: impl IntoVar, value: impl IntoVar) -> Var {
        self.forward_impl(query, key, value, None)
    }

    /// 前向传播（带 mask） — Scaled Dot-Product Multi-Head Attention
    ///
    /// # 参数
    /// - `query`: 查询 [N, T_q, input_size]
    /// - `key`: 键 [N, T_k, input_size]
    /// - `value`: 值 [N, T_k, input_size]
    /// - `attn_mask`: 注意力 mask
    ///   - shape `[T_q, T_k]`：全 batch、全 head 共享
    ///   - shape `[N, T_q, T_k]`：每 batch 独立
    ///
    /// 1.0 位置保留 scores，0.0 位置在 softmax 前减 1e9（softmax 后趋近 0）。
    pub fn forward_masked(
        &self,
        query: impl IntoVar,
        key: impl IntoVar,
        value: impl IntoVar,
        attn_mask: &Var,
    ) -> Var {
        self.forward_impl(query, key, value, Some(attn_mask))
    }

    /// 内部 forward 实现（mask 可选）
    fn forward_impl(
        &self,
        query: impl IntoVar,
        key: impl IntoVar,
        value: impl IntoVar,
        attn_mask: Option<&Var>,
    ) -> Var {
        let graph_rc = self.w_q.weights().get_graph();
        let query = query.into_var(&graph_rc).expect("Attention query 转换失败");
        let key = key.into_var(&graph_rc).expect("Attention key 转换失败");
        let value = value.into_var(&graph_rc).expect("Attention value 转换失败");

        // 分组上下文
        let desc = format!("D={}, H={}", self.embed_dim, self.num_heads);
        let _guard = NodeGroupContext::for_layer(
            &query,
            "MultiHeadAttention",
            self.instance_id,
            &self.name,
            &desc,
        );

        let q_shape = query.node().shape();
        let n = q_shape[0];
        let t_q = q_shape[1];
        let k_shape = key.node().shape();
        let t_k = k_shape[1];

        // 校验 mask 形状（如有）
        if let Some(mask) = attn_mask {
            let m_shape = mask.node().shape();
            match m_shape.len() {
                2 => assert!(
                    m_shape[0] == t_q && m_shape[1] == t_k,
                    "attn_mask [T_q, T_k] 形状 {:?} 与 (T_q={t_q}, T_k={t_k}) 不匹配",
                    m_shape
                ),
                3 => assert!(
                    m_shape[0] == n && m_shape[1] == t_q && m_shape[2] == t_k,
                    "attn_mask [N, T_q, T_k] 形状 {:?} 与 (N={n}, T_q={t_q}, T_k={t_k}) 不匹配",
                    m_shape
                ),
                _ => panic!(
                    "attn_mask 必须为 2D [T_q, T_k] 或 3D [N, T_q, T_k]，得到 {}D",
                    m_shape.len()
                ),
            }
        }

        // 1. 线性投影（Linear 只支持 2D，先 reshape）
        // [N, T, input_size] → [N*T, input_size] → Linear → [N*T, embed_dim] → [N, T, embed_dim]
        let q_flat = query
            .reshape(&[n * t_q, self.input_size])
            .expect("Q flatten 失败");
        let q = self
            .w_q
            .forward(&q_flat)
            .reshape(&[n, t_q, self.embed_dim])
            .expect("Q reshape 失败");

        let k_flat = key
            .reshape(&[n * t_k, self.input_size])
            .expect("K flatten 失败");
        let k = self
            .w_k
            .forward(&k_flat)
            .reshape(&[n, t_k, self.embed_dim])
            .expect("K reshape 失败");

        let v_flat = value
            .reshape(&[n * t_k, self.input_size])
            .expect("V flatten 失败");
        let v = self
            .w_v
            .forward(&v_flat)
            .reshape(&[n, t_k, self.embed_dim])
            .expect("V reshape 失败");

        // 2. 分头: [N, T, D] → [N*H, T, d_k]（合并 N 和 H 为 batch 维度）
        //    这样就可以用 2D matmul 做注意力计算
        let nh = n * self.num_heads;

        let q = q
            .reshape(&[n, t_q, self.num_heads, self.head_dim])
            .expect("Q reshape 失败")
            .permute(&[0, 2, 1, 3])
            .expect("Q permute 失败")
            .reshape(&[nh, t_q, self.head_dim])
            .expect("Q batch 失败"); // [N*H, T_q, d_k]

        let k = k
            .reshape(&[n, t_k, self.num_heads, self.head_dim])
            .expect("K reshape 失败")
            .permute(&[0, 2, 1, 3])
            .expect("K permute 失败")
            .reshape(&[nh, t_k, self.head_dim])
            .expect("K batch 失败"); // [N*H, T_k, d_k]

        let v = v
            .reshape(&[n, t_k, self.num_heads, self.head_dim])
            .expect("V reshape 失败")
            .permute(&[0, 2, 1, 3])
            .expect("V permute 失败")
            .reshape(&[nh, t_k, self.head_dim])
            .expect("V batch 失败"); // [N*H, T_k, d_k]

        // 3. 逐 head 做 scaled dot-product attention（2D matmul）
        let scale_tensor = Tensor::new(&[self.scale], &[1, 1]);

        let head_outputs: Vec<Var> = (0..nh)
            .map(|h| {
                // Q_h: [T_q, d_k], K_h: [T_k, d_k], V_h: [T_k, d_k]
                let q_h = q
                    .narrow(0, h, 1)
                    .expect("Q narrow 失败")
                    .reshape(&[t_q, self.head_dim])
                    .expect("Q_h reshape 失败");
                let k_h = k
                    .narrow(0, h, 1)
                    .expect("K narrow 失败")
                    .reshape(&[t_k, self.head_dim])
                    .expect("K_h reshape 失败");
                let v_h = v
                    .narrow(0, h, 1)
                    .expect("V narrow 失败")
                    .reshape(&[t_k, self.head_dim])
                    .expect("V_h reshape 失败");

                // Q_h @ K_h^T: [T_q, d_k] @ [d_k, T_k] = [T_q, T_k]
                let k_t = k_h.transpose(0, 1).expect("K transpose 失败");
                let scores = q_h.matmul(&k_t).expect("QK matmul 失败");
                let scores = &scores * &scale_tensor;

                // 应用 mask（如有）：scores + (mask - 1) * 1e9
                let scores = match attn_mask {
                    Some(mask) => {
                        let m_shape = mask.node().shape();
                        let mask_2d = if m_shape.len() == 2 {
                            // 全 head 共享 [T_q, T_k]
                            mask.clone()
                        } else {
                            // [N, T_q, T_k] — 当前 head 对应的 batch 子矩阵
                            let batch_idx = h / self.num_heads;
                            mask.narrow(0, batch_idx, 1)
                                .expect("mask narrow 失败")
                                .reshape(&[t_q, t_k])
                                .expect("mask reshape 失败")
                        };
                        // mask=1 → +0；mask=0 → -1e9
                        let bias = (&mask_2d - 1.0_f32) * 1e9_f32;
                        &scores + &bias
                    }
                    None => scores,
                };

                // softmax → [T_q, T_k]
                let attn_w = scores.softmax();

                // attn_w @ V_h: [T_q, T_k] @ [T_k, d_k] = [T_q, d_k]
                attn_w.matmul(&v_h).expect("attn@V matmul 失败")
            })
            .collect();

        // 4. 拼接 heads: stack([T_q, d_k] * N*H) → [N*H, T_q, d_k]
        //    → [N, H, T_q, d_k] → permute [N, T_q, H, d_k] → [N, T_q, D]
        let head_refs: Vec<&Var> = head_outputs.iter().collect();
        let stacked = Var::stack(&head_refs, 0).expect("stack heads 失败");
        let context = stacked
            .reshape(&[n, self.num_heads, t_q, self.head_dim])
            .expect("context reshape 失败")
            .permute(&[0, 2, 1, 3])
            .expect("context permute 失败")
            .reshape(&[n, t_q, self.embed_dim])
            .expect("context flatten 失败");

        // 8. 输出投影: [N, T_q, D] → [N*T_q, D] → Linear → [N*T_q, D] → [N, T_q, D]
        let context_flat = context
            .reshape(&[n * t_q, self.embed_dim])
            .expect("output flatten 失败");
        self.w_o
            .forward(&context_flat)
            .reshape(&[n, t_q, self.embed_dim])
            .expect("output reshape 失败")
    }

    /// 构造 causal（下三角）attention mask，shape `[t, t]`
    ///
    /// 下三角（含对角线）为 1，上三角为 0。可直接传给 [`forward_masked`] 实现自回归。
    ///
    /// [`forward_masked`]: MultiHeadAttention::forward_masked
    pub fn causal_mask(graph: &Graph, t: usize) -> Result<Var, GraphError> {
        assert!(t > 0, "causal_mask: t 必须 > 0");
        let mut data = vec![0.0_f32; t * t];
        for i in 0..t {
            for j in 0..=i {
                data[i * t + j] = 1.0;
            }
        }
        let tensor = Tensor::new(&data, &[t, t]);
        graph.input_named(&tensor, &format!("causal_mask_{t}"))
    }

    /// 构造 key padding mask，shape `[N, 1, max_len]`
    ///
    /// 第 i 个 batch 的前 `lengths[i]` 列为 1，其余为 0。
    /// 用户可自行 broadcast 或 reshape 到 `[N, T_q, max_len]` 后传给 [`forward_masked`]。
    ///
    /// [`forward_masked`]: MultiHeadAttention::forward_masked
    pub fn padding_mask(
        graph: &Graph,
        lengths: &[usize],
        max_len: usize,
    ) -> Result<Var, GraphError> {
        assert!(!lengths.is_empty(), "padding_mask: lengths 不能为空");
        assert!(max_len > 0, "padding_mask: max_len 必须 > 0");
        let n = lengths.len();
        let mut data = vec![0.0_f32; n * max_len];
        for (i, &len) in lengths.iter().enumerate() {
            assert!(
                len <= max_len,
                "padding_mask: lengths[{i}]={len} > max_len={max_len}"
            );
            for j in 0..len {
                data[i * max_len + j] = 1.0;
            }
        }
        let tensor = Tensor::new(&data, &[n, 1, max_len]);
        graph.input_named(&tensor, "padding_mask")
    }

    /// 获取输入特征维
    pub const fn input_size(&self) -> usize {
        self.input_size
    }

    /// 获取嵌入维度
    pub const fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// 获取头数
    pub const fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// 获取 Q 投影的 Linear（演化系统的参数遍历用）
    pub const fn w_q(&self) -> &Linear {
        &self.w_q
    }

    /// 获取 K 投影
    pub const fn w_k(&self) -> &Linear {
        &self.w_k
    }

    /// 获取 V 投影
    pub const fn w_v(&self) -> &Linear {
        &self.w_v
    }

    /// 获取输出投影
    pub const fn w_o(&self) -> &Linear {
        &self.w_o
    }
}

impl Module for MultiHeadAttention {
    fn parameters(&self) -> Vec<Var> {
        let mut params = Vec::new();
        params.extend(self.w_q.parameters());
        params.extend(self.w_k.parameters());
        params.extend(self.w_v.parameters());
        params.extend(self.w_o.parameters());
        params
    }
}
