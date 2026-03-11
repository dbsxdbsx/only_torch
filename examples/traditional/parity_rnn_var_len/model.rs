//! 变长奇偶性检测模型定义（展开式 RNN，PyTorch 风格）
//!
//! 使用 RNN 层判断二进制序列中 1 的个数是奇数还是偶数。
//! **支持变长序列**：动态图下每次 forward 自动重新展开，天然支持变长。
//!
//! ## 网络结构
//! ```text
//! x: [batch, seq_len, 1] → RNN → h: [batch, hidden] → Linear → [batch, 2]
//! ```
//! `注意：seq_len` 可以在不同批次之间变化！

use only_torch::nn::{Graph, GraphError, Linear, Module, Rnn, Var};
use only_torch::tensor::Tensor;

/// 变长奇偶性检测 RNN 模型
pub struct ParityRNN {
    rnn: Rnn,
    fc: Linear,
}

impl ParityRNN {
    pub fn new(graph: &Graph, hidden_size: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("ParityRNN");
        let rnn = Rnn::new(&graph, 1, hidden_size, "rnn")?;
        let fc = Linear::new(&graph, hidden_size, 2, true, "fc")?;
        Ok(Self { rnn, fc })
    }

    /// 前向传播：接收 &Tensor，RNN 层自动转为 Var
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        let h = self.rnn.forward(x)?;
        Ok(self.fc.forward(&h))
    }
}

impl Module for ParityRNN {
    fn parameters(&self) -> Vec<Var> {
        [self.rnn.parameters(), self.fc.parameters()].concat()
    }
}
