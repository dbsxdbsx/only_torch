//! 变长奇偶性检测模型定义（LSTM，PyTorch 风格）
//!
//! 使用 LSTM 层判断二进制序列中 1 的个数是奇数还是偶数。
//! **支持变长序列**：动态图下每次 forward 自动重新展开，天然支持变长。
//!
//! ## 网络结构
//! ```text
//! x: [batch, seq_len, 1] → LSTM → h: [batch, hidden] → Linear → [batch, 2]
//! ```
//! `注意：seq_len` 可以在不同批次之间变化！

use only_torch::nn::{Graph, GraphError, Linear, Lstm, Module, Var};
use only_torch::tensor::Tensor;

/// 变长奇偶性检测 LSTM 模型
pub struct ParityLSTM {
    lstm: Lstm,
    fc: Linear,
}

impl ParityLSTM {
    pub fn new(graph: &Graph, hidden_size: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("ParityLSTM");
        let lstm = Lstm::new(&graph, 1, hidden_size, "lstm")?;
        let fc = Linear::new(&graph, hidden_size, 2, true, "fc")?;
        Ok(Self { lstm, fc })
    }

    /// 前向传播：接收 &Tensor，LSTM 层自动转为 Var
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        let h = self.lstm.forward(x)?;
        Ok(self.fc.forward(&h))
    }
}

impl Module for ParityLSTM {
    fn parameters(&self) -> Vec<Var> {
        [self.lstm.parameters(), self.fc.parameters()].concat()
    }
}
