/*
 * @Author       : 老董
 * @Date         : 2026-05-02
 * @Description  : Positional Encoding 层 - Sinusoidal + LearnableAbsolute
 *
 * 把"序列中位置"信息以加法方式注入 token embedding：
 *   y = x + PE
 *
 * - SinusoidalPositionalEncoding：固定的 sin/cos 编码（无可训练参数）
 * - LearnableAbsolutePositionalEncoding：参数化 lookup table，初始 N(0, 0.02)
 *
 * 输入形状约定：[B, T, embed_dim]，T <= max_len。
 * 内部用 narrow + 加法广播得到 [B, T, embed_dim]。
 */

use crate::nn::graph::NodeGroupContext;
use crate::nn::{Graph, GraphError, Init, IntoVar, Module, Var, VarShapeOps};
use crate::tensor::Tensor;

/// Sinusoidal Positional Encoding（无参数）
///
/// 标准 Transformer 公式（"Attention Is All You Need"）：
/// - `PE(pos, 2i)   = sin(pos / 10000^(2i / D))`
/// - `PE(pos, 2i+1) = cos(pos / 10000^(2i / D))`
///
/// # 使用示例
/// ```ignore
/// let pe = SinusoidalPositionalEncoding::new(&graph, 512, 64, "pe")?;
/// let h = pe.forward(&x);  // x: [B, T, 64]
/// ```
pub struct SinusoidalPositionalEncoding {
    /// 预计算的 PE 张量节点 [1, max_len, embed_dim]
    pe: Var,
    /// 序列最大长度
    max_len: usize,
    /// 嵌入维度
    embed_dim: usize,
    /// 层名称
    name: String,
    /// 分组实例 ID
    instance_id: usize,
}

impl SinusoidalPositionalEncoding {
    /// 创建 Sinusoidal Positional Encoding 层
    ///
    /// # 参数
    /// - `graph`: 计算图
    /// - `max_len`: 序列最大长度
    /// - `embed_dim`: 嵌入维度
    /// - `name`: 层名称前缀
    pub fn new(
        graph: &Graph,
        max_len: usize,
        embed_dim: usize,
        name: &str,
    ) -> Result<Self, GraphError> {
        assert!(
            max_len > 0,
            "SinusoidalPositionalEncoding: max_len 必须 > 0"
        );
        assert!(
            embed_dim > 0,
            "SinusoidalPositionalEncoding: embed_dim 必须 > 0"
        );

        // 预计算 PE 张量 [1, max_len, embed_dim]
        let mut data = vec![0.0f32; max_len * embed_dim];
        let ln_base = 10000.0_f32.ln();
        for pos in 0..max_len {
            for i in 0..embed_dim {
                // 偶数维 sin / 奇数维 cos：div_term 的指数取 2*floor(i/2)/D
                let two_i = (i & !1) as f32;
                let div_term = (two_i / embed_dim as f32 * ln_base).exp();
                let angle = (pos as f32) / div_term;
                data[pos * embed_dim + i] = if i & 1 == 0 { angle.sin() } else { angle.cos() };
            }
        }
        let pe_tensor = Tensor::new(&data, &[1, max_len, embed_dim]);
        let pe = graph.input_named(&pe_tensor, &format!("{name}_pe"))?;

        let instance_id = graph.inner_mut().next_node_group_instance_id();

        Ok(Self {
            pe,
            max_len,
            embed_dim,
            name: name.to_string(),
            instance_id,
        })
    }

    /// 前向：把 PE 加到输入上
    ///
    /// # 参数
    /// - `x`: 输入 `[B, T, embed_dim]`，要求 `T <= max_len`
    ///
    /// # 返回
    /// `[B, T, embed_dim]`
    pub fn forward(&self, x: impl IntoVar) -> Var {
        let x = x
            .into_var(&self.pe.get_graph())
            .expect("SinusoidalPositionalEncoding 输入转换失败");

        let desc = format!("max_len={}, D={}", self.max_len, self.embed_dim);
        let _guard = NodeGroupContext::for_layer(
            &x,
            "SinusoidalPositionalEncoding",
            self.instance_id,
            &self.name,
            &desc,
        );
        _guard.tag_existing(&self.pe);

        let x_shape = x.node().shape();
        assert!(
            x_shape.len() == 3,
            "SinusoidalPositionalEncoding: 输入需为 3D [B, T, D]，得到 {}D",
            x_shape.len()
        );
        let t = x_shape[1];
        assert!(
            x_shape[2] == self.embed_dim,
            "SinusoidalPositionalEncoding: 输入最后一维 {} 与 embed_dim {} 不匹配",
            x_shape[2],
            self.embed_dim
        );
        assert!(
            t <= self.max_len,
            "SinusoidalPositionalEncoding: 序列长度 {t} 超出 max_len {}",
            self.max_len
        );

        // 从 [1, max_len, D] 截出 [1, T, D]，与 [B, T, D] 广播相加
        let pe_slice = self
            .pe
            .narrow(1, 0, t)
            .expect("SinusoidalPositionalEncoding narrow 失败");
        &x + &pe_slice
    }

    /// 获取最大序列长度
    pub const fn max_len(&self) -> usize {
        self.max_len
    }

    /// 获取嵌入维度
    pub const fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

impl Module for SinusoidalPositionalEncoding {
    fn parameters(&self) -> Vec<Var> {
        vec![]
    }
}

/// Learnable Absolute Positional Encoding（可训练 lookup table）
///
/// 参数形状 `[max_len, embed_dim]`，初始 `Normal(0, 0.02)`。
/// 与 BERT 等模型采用的 "learned absolute" 位置编码一致。
///
/// # 使用示例
/// ```ignore
/// let pe = LearnableAbsolutePositionalEncoding::new(&graph, 128, 64, "lpe")?;
/// let h = pe.forward(&x);
/// ```
pub struct LearnableAbsolutePositionalEncoding {
    /// 可训练 lookup table [max_len, embed_dim]
    weight: Var,
    /// 序列最大长度
    max_len: usize,
    /// 嵌入维度
    embed_dim: usize,
    /// 层名称
    name: String,
    /// 分组实例 ID
    instance_id: usize,
}

impl LearnableAbsolutePositionalEncoding {
    /// 创建可训练绝对位置编码层
    pub fn new(
        graph: &Graph,
        max_len: usize,
        embed_dim: usize,
        name: &str,
    ) -> Result<Self, GraphError> {
        assert!(
            max_len > 0,
            "LearnableAbsolutePositionalEncoding: max_len 必须 > 0"
        );
        assert!(
            embed_dim > 0,
            "LearnableAbsolutePositionalEncoding: embed_dim 必须 > 0"
        );

        let weight = graph.parameter(
            &[max_len, embed_dim],
            Init::Normal {
                mean: 0.0,
                std: 0.02,
            },
            &format!("{name}_weight"),
        )?;

        let instance_id = graph.inner_mut().next_node_group_instance_id();

        Ok(Self {
            weight,
            max_len,
            embed_dim,
            name: name.to_string(),
            instance_id,
        })
    }

    /// 前向：把对应位置的 lookup 向量加到输入上
    ///
    /// # 参数
    /// - `x`: 输入 `[B, T, embed_dim]`，要求 `T <= max_len`
    pub fn forward(&self, x: impl IntoVar) -> Var {
        let x = x
            .into_var(&self.weight.get_graph())
            .expect("LearnableAbsolutePositionalEncoding 输入转换失败");

        let desc = format!("max_len={}, D={}", self.max_len, self.embed_dim);
        let _guard = NodeGroupContext::for_layer(
            &x,
            "LearnableAbsolutePositionalEncoding",
            self.instance_id,
            &self.name,
            &desc,
        );
        _guard.tag_existing(&self.weight);

        let x_shape = x.node().shape();
        assert!(
            x_shape.len() == 3,
            "LearnableAbsolutePositionalEncoding: 输入需为 3D [B, T, D]，得到 {}D",
            x_shape.len()
        );
        let t = x_shape[1];
        assert!(
            x_shape[2] == self.embed_dim,
            "LearnableAbsolutePositionalEncoding: 输入最后一维 {} 与 embed_dim {} 不匹配",
            x_shape[2],
            self.embed_dim
        );
        assert!(
            t <= self.max_len,
            "LearnableAbsolutePositionalEncoding: 序列长度 {t} 超出 max_len {}",
            self.max_len
        );

        // [max_len, D] → narrow → [t, D] → unsqueeze(0) → [1, t, D]，再广播加 [B, T, D]
        let pe_slice = self
            .weight
            .narrow(0, 0, t)
            .expect("LearnableAbsolutePositionalEncoding narrow 失败")
            .unsqueeze(0)
            .expect("LearnableAbsolutePositionalEncoding unsqueeze 失败");
        &x + &pe_slice
    }

    /// 获取最大序列长度
    pub const fn max_len(&self) -> usize {
        self.max_len
    }

    /// 获取嵌入维度
    pub const fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// 获取参数 weight
    pub const fn weight(&self) -> &Var {
        &self.weight
    }
}

impl Module for LearnableAbsolutePositionalEncoding {
    fn parameters(&self) -> Vec<Var> {
        vec![self.weight.clone()]
    }
}
