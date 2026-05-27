//! 变长奇偶性检测模型定义（Transformer Encoder，PyTorch 风格）
//!
//! 使用 token embedding + Sinusoidal PE + 多层 Transformer Encoder + sum pooling
//! 完成二进制序列的奇偶判别。
//!
//! ## 网络结构
//! ```text
//! x: [B, T, 1] → reshape → [B, T]
//!              → Embedding(2 → d_model) → [B, T, d_model]
//!              → SinusoidalPositionalEncoding(max_len, d_model)
//!              → TransformerEncoder(num_layers × layer)
//!              → 取最后位置 (narrow(1, t-1, 1).squeeze(1)) → [B, d_model]
//!              → Linear(d_model → 2) → logits [B, 2]
//! ```
//!
//! ## 设计取舍
//! - `Embedding(2, d_model)` 比 `Linear(1, d_model)` 给 0 / 1 各自一组可学向量，
//!   是 transformer 在 parity 等离散 token 任务上的标准做法。
//! - **取最后位置而非 mean / sum pooling**：parity 是 counter 任务，position-wise
//!   pool 后 attention 学到的"按位置区分"信息会被稀释；取最后位置（类似 [CLS]）
//!   让分类头看到"已聚合所有位置的单一向量"，更易在 parity 上收敛。
//! - 多层堆叠 + 适度 d_model 让 attention 有足够表达力学到"按位置累积奇偶"。
//!
//! 与 [`parity_rnn_var_len`](../parity_rnn_var_len/model.rs) /
//! [`parity_lstm_var_len`](../parity_lstm_var_len/model.rs) /
//! [`parity_gru_var_len`](../parity_gru_var_len/model.rs) 同任务、同数据，
//! 用于直观对照"序列建模算子"的差异。

use only_torch::nn::{
    Embedding, Graph, GraphError, Linear, Module, SinusoidalPositionalEncoding,
    TransformerEncoder, Var, VarShapeOps,
};
use only_torch::tensor::Tensor;

/// 变长奇偶性检测 Transformer 模型
pub struct ParityTransformer {
    token_emb: Embedding,
    pos_enc: SinusoidalPositionalEncoding,
    encoder: TransformerEncoder,
    head: Linear,
}

impl ParityTransformer {
    /// 构造模型
    ///
    /// # 参数
    /// - `graph`: 计算图
    /// - `max_len`: 序列最大长度（位置编码预算）
    /// - `d_model`: Transformer 隐藏维度
    /// - `num_heads`: 注意力头数
    /// - `d_ff`: FFN 中间维度
    /// - `num_layers`: encoder 堆叠层数
    pub fn new(
        graph: &Graph,
        max_len: usize,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        num_layers: usize,
    ) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("ParityTransformer");
        let token_emb = Embedding::new(&graph, 2, d_model, "tok_emb")?;
        let pos_enc = SinusoidalPositionalEncoding::new(&graph, max_len, d_model, "pe")?;
        let encoder =
            TransformerEncoder::new(&graph, num_layers, d_model, num_heads, d_ff, 0.0, "te")?;
        let head = Linear::new(&graph, d_model, 2, true, "head")?;
        Ok(Self {
            token_emb,
            pos_enc,
            encoder,
            head,
        })
    }

    /// 前向传播：接收 `[B, T, 1]` 形状的 Tensor（值域 0 / 1）
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        let shape = x.shape();
        assert!(
            shape.len() == 3 && shape[2] == 1,
            "ParityTransformer: 输入需为 [B, T, 1]，得到 {shape:?}"
        );
        let b = shape[0];
        let t = shape[1];

        // 1) 把 [B, T, 1] reshape 成 [B, T]，作为整数索引送入 Embedding
        let graph = self.token_emb.weight().get_graph();
        let x_var = graph.input(x)?;
        let x_idx = x_var.reshape(&[b, t])?;
        let h = self.token_emb.forward(&x_idx);

        // 2) 加位置编码
        let h = self.pos_enc.forward(&h);

        // 3) Transformer Encoder（无 mask，桶式同长度免 padding mask）
        let h = self.encoder.forward(&h);

        // 4) 取最后位置作为聚合：narrow(1, t-1, 1) → squeeze(1) → [B, d_model]
        let h = h.narrow(1, t - 1, 1)?.squeeze(Some(1))?;

        // 5) 分类头
        Ok(self.head.forward(&h))
    }
}

impl Module for ParityTransformer {
    fn parameters(&self) -> Vec<Var> {
        let mut params = Vec::new();
        params.extend(self.token_emb.parameters());
        params.extend(self.pos_enc.parameters());
        params.extend(self.encoder.parameters());
        params.extend(self.head.parameters());
        params
    }
}
