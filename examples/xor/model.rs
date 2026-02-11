//! XOR 模型定义
//!
//! 使用 Linear 层的 PyTorch 风格实现（Phase 3 新 API）。
//!
//! ## 网络结构
//! ```text
//! Input(2) -> Linear(4, Tanh) -> Linear(2) -> Softmax
//! ```

use only_torch::nn::{Graph, GraphError, Linear, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

/// XOR 多层感知机
pub struct XorMLP {
    fc1: Linear,
    fc2: Linear,
}

impl XorMLP {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        Ok(Self {
            fc1: Linear::new(graph, 2, 4, true, "fc1")?,
            fc2: Linear::new(graph, 4, 2, true, "fc2")?,
        })
    }

    /// PyTorch 风格 forward：接收 &Tensor，Linear 自动转为 Var
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        let h1 = self.fc1.forward(x).tanh(); // Linear 内部自动 Tensor → Var
        let out = self.fc2.forward(&h1);
        Ok(out)
    }
}

impl Module for XorMLP {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}
