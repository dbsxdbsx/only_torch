//! 奇偶性检测模型定义（展开式 RNN，PyTorch 风格）
//!
//! 使用 RNN 层判断二进制序列中 1 的个数是奇数还是偶数。
//!
//! ## 任务说明
//! - 输入：长度为 N 的 0/1 序列
//! - 输出：2 类分类（偶数=类0，奇数=类1）
//!
//! ## 网络结构
//! ```text
//! x: [batch, seq_len, 1] → RNN → h: [batch, hidden] → Linear → [batch, 2]
//! ```

use only_torch::nn::{Graph, GraphError, Linear, Module, Rnn, Var};
use only_torch::tensor::Tensor;

/// 奇偶性检测 RNN 模型（PyTorch 风格）
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
        let h = self.rnn.forward(x)?; // RNN 内部自动 Tensor → Var
        Ok(self.fc.forward(&h))
    }
}

impl Module for ParityRNN {
    fn parameters(&self) -> Vec<Var> {
        [self.rnn.parameters(), self.fc.parameters()].concat()
    }
}
