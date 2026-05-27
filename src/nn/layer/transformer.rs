/*
 * @Author       : 老董
 * @Date         : 2026-05-02
 * @Description  : Transformer Encoder 层 - Pre-LN 风格
 *
 * 单个 TransformerEncoderLayer 计算流：
 *   x  = x + Dropout(MHA(LN(x)))
 *   x  = x + Dropout(FFN(LN(x)))
 *
 *   FFN(h) = Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model)
 *
 * Pre-LN 比 Post-LN 训练更稳定（无需 warmup），现代 LLM 默认走 Pre-LN。
 *
 * 与 attention 层一致，提供 forward / forward_masked 双 API：
 * - `forward(x)`：无 mask 路径
 * - `forward_masked(x, mask)`：透传 mask 给底层 MHA
 *
 * `TransformerEncoder` 是 N 层 `TransformerEncoderLayer` 的堆叠。
 */

use crate::nn::graph::NodeGroupContext;
use crate::nn::{
    Graph, GraphError, IntoVar, LayerNorm, Linear, Module, MultiHeadAttention, Var,
    VarActivationOps, VarRegularizationOps, VarShapeOps,
};

/// 单层 Transformer Encoder（Pre-LN 风格）
pub struct TransformerEncoderLayer {
    /// MHA 前的 LayerNorm
    ln1: LayerNorm,
    /// 自注意力子层
    mha: MultiHeadAttention,
    /// FFN 前的 LayerNorm
    ln2: LayerNorm,
    /// FFN 第一个 Linear: d_model → d_ff
    ff1: Linear,
    /// FFN 第二个 Linear: d_ff → d_model
    ff2: Linear,
    /// 模型维度
    d_model: usize,
    /// 注意力头数
    num_heads: usize,
    /// FFN 中间维度
    d_ff: usize,
    /// dropout 概率
    dropout: f32,
    /// 层名称
    name: String,
    /// 分组实例 ID
    instance_id: usize,
}

impl TransformerEncoderLayer {
    /// 创建一个 TransformerEncoderLayer
    ///
    /// # 参数
    /// - `graph`: 计算图
    /// - `d_model`: 模型隐藏维度（也是输入/输出最后一维）
    /// - `num_heads`: 注意力头数（必须能整除 `d_model`）
    /// - `d_ff`: FFN 中间维度（典型值 4 × d_model）
    /// - `dropout`: dropout 概率（推理或不需要时传 0.0）
    /// - `name`: 层名称前缀
    pub fn new(
        graph: &Graph,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout: f32,
        name: &str,
    ) -> Result<Self, GraphError> {
        assert!(d_model > 0, "TransformerEncoderLayer: d_model 必须 > 0");
        assert!(num_heads > 0, "TransformerEncoderLayer: num_heads 必须 > 0");
        assert!(
            d_model % num_heads == 0,
            "TransformerEncoderLayer: d_model={d_model} 必须被 num_heads={num_heads} 整除"
        );
        assert!(d_ff > 0, "TransformerEncoderLayer: d_ff 必须 > 0");
        assert!(
            (0.0..1.0).contains(&dropout),
            "TransformerEncoderLayer: dropout={dropout} 须在 [0, 1)"
        );

        let ln1 = LayerNorm::new(graph, &[d_model], 1e-5, &format!("{name}_ln1"))?;
        let mha = MultiHeadAttention::new(graph, d_model, num_heads, &format!("{name}_mha"))?;
        let ln2 = LayerNorm::new(graph, &[d_model], 1e-5, &format!("{name}_ln2"))?;
        let ff1 = Linear::new(graph, d_model, d_ff, true, &format!("{name}_ff1"))?;
        let ff2 = Linear::new(graph, d_ff, d_model, true, &format!("{name}_ff2"))?;

        let instance_id = graph.inner_mut().next_node_group_instance_id();

        Ok(Self {
            ln1,
            mha,
            ln2,
            ff1,
            ff2,
            d_model,
            num_heads,
            d_ff,
            dropout,
            name: name.to_string(),
            instance_id,
        })
    }

    /// 前向（无 mask）
    ///
    /// # 参数
    /// - `x`: 输入 `[B, T, d_model]`
    pub fn forward(&self, x: impl IntoVar) -> Var {
        self.forward_impl(x, None)
    }

    /// 前向（带 mask） — mask 透传给底层 MHA
    pub fn forward_masked(&self, x: impl IntoVar, mask: &Var) -> Var {
        self.forward_impl(x, Some(mask))
    }

    fn forward_impl(&self, x: impl IntoVar, mask: Option<&Var>) -> Var {
        let x = x
            .into_var(&self.ln1.gamma().get_graph())
            .expect("TransformerEncoderLayer 输入转换失败");

        let desc = format!(
            "d_model={}, H={}, d_ff={}",
            self.d_model, self.num_heads, self.d_ff
        );
        let _guard = NodeGroupContext::for_layer(
            &x,
            "TransformerEncoderLayer",
            self.instance_id,
            &self.name,
            &desc,
        );

        let x_shape = x.node().shape();
        assert!(
            x_shape.len() == 3 && x_shape[2] == self.d_model,
            "TransformerEncoderLayer: 输入需为 [B, T, {}]，得到 {:?}",
            self.d_model,
            x_shape
        );
        let b = x_shape[0];
        let t = x_shape[1];

        // ===== 子层 1：Pre-LN + MHA + Dropout + Residual =====
        let n1 = self.ln1.forward(&x);
        let attn_out = match mask {
            Some(m) => self.mha.forward_masked(&n1, &n1, &n1, m),
            None => self.mha.forward(&n1, &n1, &n1),
        };
        let attn_out = if self.dropout > 0.0 {
            attn_out
                .dropout(self.dropout)
                .expect("Attention dropout 失败")
        } else {
            attn_out
        };
        let x1 = &x + &attn_out;

        // ===== 子层 2：Pre-LN + FFN + Dropout + Residual =====
        let n2 = self.ln2.forward(&x1);
        // FFN：Linear 只接受 2D，先 flatten
        let n2_flat = n2
            .reshape(&[b * t, self.d_model])
            .expect("FFN flatten 失败");
        let h = self.ff1.forward(&n2_flat).gelu();
        let ff_out = self
            .ff2
            .forward(&h)
            .reshape(&[b, t, self.d_model])
            .expect("FFN reshape 失败");
        let ff_out = if self.dropout > 0.0 {
            ff_out.dropout(self.dropout).expect("FFN dropout 失败")
        } else {
            ff_out
        };
        &x1 + &ff_out
    }

    /// 模型维度
    pub const fn d_model(&self) -> usize {
        self.d_model
    }

    /// 注意力头数
    pub const fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// FFN 中间维度
    pub const fn d_ff(&self) -> usize {
        self.d_ff
    }
}

impl Module for TransformerEncoderLayer {
    fn parameters(&self) -> Vec<Var> {
        let mut params = Vec::new();
        params.extend(self.ln1.parameters());
        params.extend(self.mha.parameters());
        params.extend(self.ln2.parameters());
        params.extend(self.ff1.parameters());
        params.extend(self.ff2.parameters());
        params
    }
}

/// 多层 Transformer Encoder
pub struct TransformerEncoder {
    layers: Vec<TransformerEncoderLayer>,
}

impl TransformerEncoder {
    /// 创建 N 层 Transformer Encoder
    ///
    /// # 参数
    /// - `graph`: 计算图
    /// - `num_layers`: 堆叠层数
    /// - `d_model`、`num_heads`、`d_ff`、`dropout`: 与 [`TransformerEncoderLayer::new`] 同义
    /// - `name`: 层名称前缀（每层会附加 `_layer{i}`）
    pub fn new(
        graph: &Graph,
        num_layers: usize,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout: f32,
        name: &str,
    ) -> Result<Self, GraphError> {
        assert!(num_layers > 0, "TransformerEncoder: num_layers 必须 > 0");
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(TransformerEncoderLayer::new(
                graph,
                d_model,
                num_heads,
                d_ff,
                dropout,
                &format!("{name}_layer{i}"),
            )?);
        }
        Ok(Self { layers })
    }

    /// 前向（无 mask） — 顺序通过所有层
    pub fn forward(&self, x: impl IntoVar) -> Var {
        let graph_rc = self.layers[0].ln1.gamma().get_graph();
        let mut h = x
            .into_var(&graph_rc)
            .expect("TransformerEncoder 输入转换失败");
        for layer in &self.layers {
            h = layer.forward(&h);
        }
        h
    }

    /// 前向（带 mask） — 每层都接收同一个 mask
    pub fn forward_masked(&self, x: impl IntoVar, mask: &Var) -> Var {
        let graph_rc = self.layers[0].ln1.gamma().get_graph();
        let mut h = x
            .into_var(&graph_rc)
            .expect("TransformerEncoder 输入转换失败");
        for layer in &self.layers {
            h = layer.forward_masked(&h, mask);
        }
        h
    }

    /// 层数
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl Module for TransformerEncoder {
    fn parameters(&self) -> Vec<Var> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }
}
