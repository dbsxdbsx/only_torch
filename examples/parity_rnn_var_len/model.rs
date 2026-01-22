//! 变长奇偶性检测模型定义（展开式 RNN，PyTorch 风格）
//!
//! 使用 RNN 层判断二进制序列中 1 的个数是奇数还是偶数。
//! **支持变长序列**：使用智能缓存自动处理不同长度的输入。
//!
//! ## 与 `fixed_len` 的区别
//! - 输入序列长度可以变化
//! - 模型定义完全相同（使用标准 `ModelState`）
//! - 智能缓存自动处理不同形状
//!
//! ## 网络结构
//! ```text
//! x: [batch, seq_len, 1] → RNN → h: [batch, hidden] → Linear → [batch, 2]
//! ```
//! `注意：seq_len` 可以在不同批次之间变化！

use only_torch::nn::{Graph, GraphError, Linear, ModelState, Module, Rnn, Var};
use only_torch::tensor::Tensor;

/// 变长奇偶性检测 RNN 模型（PyTorch 风格）
///
/// 使用 `ModelState` 自动管理不同形状的计算图缓存，
/// 代码风格与 `fixed_len` 版本完全一致！
pub struct ParityRNN {
    rnn: Rnn,
    fc: Linear,
    state: ModelState,
}

impl ParityRNN {
    /// 创建奇偶性检测模型
    ///
    /// # 参数
    /// - `graph`: 计算图
    /// - `hidden_size`: RNN 隐藏层大小
    pub fn new(graph: &Graph, hidden_size: usize) -> Result<Self, GraphError> {
        // RNN: input_size=1 (单个 bit), hidden_size 由参数指定
        let rnn = Rnn::new(graph, 1, hidden_size, "rnn")?;

        // Linear: hidden_size -> 2 (二分类：偶数/奇数)
        let fc = Linear::new(graph, hidden_size, 2, true, "fc")?;

        // ModelState 自动处理变长缓存
        let state = ModelState::new(graph);

        Ok(Self { rnn, fc, state })
    }

    /// 前向传播（PyTorch 风格，支持变长）
    ///
    /// # 参数
    /// - `x`: 输入张量 `[batch, seq_len, 1]`（`seq_len` 可变）
    ///
    /// # 返回
    /// 2 类 logits，形状 `[batch, 2]`
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        self.state.forward(x, |input| {
            let h = self.rnn.forward(input)?;
            Ok(self.fc.forward(&h))
        })
    }

    /// 获取缓存的形状数量
    pub fn cache_size(&self) -> usize {
        self.state.cache_size()
    }
}

impl Module for ParityRNN {
    fn parameters(&self) -> Vec<Var> {
        let mut params = self.rnn.parameters();
        params.extend(self.fc.parameters());
        params
    }
}
