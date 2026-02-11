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
    graph: Graph,
    rnn: Rnn,
    fc: Linear,
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

        Ok(Self {
            graph: graph.clone(),
            rnn,
            fc,
        })
    }

    /// 前向传播（PyTorch 风格）
    ///
    /// # 参数
    /// - `x`: 输入张量 `[batch, seq_len, 1]`
    ///
    /// # 返回
    /// 2 类 logits，形状 `[batch, 2]`
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        let input = self.graph.input(x)?;
        // RNN 处理序列
        let h = self.rnn.forward(&input)?;
        // Linear 输出分类 logits
        Ok(self.fc.forward(&h))
    }
}

impl Module for ParityRNN {
    fn parameters(&self) -> Vec<Var> {
        [self.rnn.parameters(), self.fc.parameters()].concat()
    }
}
