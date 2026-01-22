//! 变长奇偶性检测模型定义（GRU，PyTorch 风格）
//!
//! 使用 GRU 层判断二进制序列中 1 的个数是奇数还是偶数。
//! **支持变长序列**：使用智能缓存自动处理不同长度的输入。
//!
//! ## 网络结构
//! ```text
//! x: [batch, seq_len, 1] → GRU → h: [batch, hidden] → Linear → [batch, 2]
//! ```
//! `注意：seq_len` 可以在不同批次之间变化！

use only_torch::nn::{Graph, GraphError, Gru, Linear, ModelState, Module, Var};
use only_torch::tensor::Tensor;

/// 变长奇偶性检测 GRU 模型（PyTorch 风格）
pub struct ParityGRU {
    gru: Gru,
    fc: Linear,
    state: ModelState,
}

impl ParityGRU {
    /// 创建奇偶性检测模型
    pub fn new(graph: &Graph, hidden_size: usize) -> Result<Self, GraphError> {
        let gru = Gru::new(graph, 1, hidden_size, "gru")?;
        let fc = Linear::new(graph, hidden_size, 2, true, "fc")?;
        let state = ModelState::new(graph);

        Ok(Self { gru, fc, state })
    }

    /// 前向传播（PyTorch 风格，支持变长）
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        self.state.forward(x, |input| {
            let h = self.gru.forward(input)?;
            Ok(self.fc.forward(&h))
        })
    }

    /// 获取缓存的形状数量
    pub fn cache_size(&self) -> usize {
        self.state.cache_size()
    }
}

impl Module for ParityGRU {
    fn parameters(&self) -> Vec<Var> {
        let mut params = self.gru.parameters();
        params.extend(self.fc.parameters());
        params
    }
}
