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
/// let output = attn.forward(&x, &x, &x, None);  // self-attention
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
    /// 创建多头注意力层
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
        assert!(
            embed_dim % num_heads == 0,
            "MultiHeadAttention: embed_dim={embed_dim} 必须能被 num_heads={num_heads} 整除"
        );

        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let w_q = Linear::new(graph, embed_dim, embed_dim, true, &format!("{name}_q"))?;
        let w_k = Linear::new(graph, embed_dim, embed_dim, true, &format!("{name}_k"))?;
        let w_v = Linear::new(graph, embed_dim, embed_dim, true, &format!("{name}_v"))?;
        let w_o = Linear::new(graph, embed_dim, embed_dim, true, &format!("{name}_o"))?;

        let instance_id = graph.inner_mut().next_node_group_instance_id();

        Ok(Self {
            w_q,
            w_k,
            w_v,
            w_o,
            embed_dim,
            num_heads,
            head_dim,
            scale,
            name: name.to_string(),
            instance_id,
        })
    }

    /// 前向传播 — Scaled Dot-Product Multi-Head Attention
    ///
    /// # 参数
    /// - `query`: 查询 [N, T_q, D]
    /// - `key`: 键 [N, T_k, D]
    /// - `value`: 值 [N, T_k, D]
    ///
    /// # 返回
    /// - output: [N, T_q, D]
    ///
    /// # 说明
    /// self-attention: `forward(&x, &x, &x)`
    /// cross-attention: `forward(&q, &kv, &kv)`
    pub fn forward(&self, query: impl IntoVar, key: impl IntoVar, value: impl IntoVar) -> Var {
        let graph_rc = self.w_q.parameters()[0].get_graph();
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

        // 1. 线性投影（Linear 只支持 2D，先 reshape）
        // [N, T, D] → [N*T, D] → Linear → [N*T, D] → [N, T, D]
        let q_flat = query
            .reshape(&[n * t_q, self.embed_dim])
            .expect("Q flatten 失败");
        let q = self
            .w_q
            .forward(&q_flat)
            .reshape(&[n, t_q, self.embed_dim])
            .expect("Q reshape 失败");

        let k_flat = key
            .reshape(&[n * t_k, self.embed_dim])
            .expect("K flatten 失败");
        let k = self
            .w_k
            .forward(&k_flat)
            .reshape(&[n, t_k, self.embed_dim])
            .expect("K reshape 失败");

        let v_flat = value
            .reshape(&[n * t_k, self.embed_dim])
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

    /// 获取嵌入维度
    pub const fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// 获取头数
    pub const fn num_heads(&self) -> usize {
        self.num_heads
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
