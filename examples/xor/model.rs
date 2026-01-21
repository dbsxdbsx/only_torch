//! XOR 模型定义
//!
//! 使用 Linear 层 + ModelState 的 PyTorch 风格实现。
//!
//! ## 网络结构
//! ```text
//! Input(2) -> Linear(4, Tanh) -> Linear(2) -> Softmax
//! ```

use only_torch::nn::{Graph, GraphError, Linear, ModelState, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

/// XOR 多层感知机
pub struct XorMLP {
    fc1: Linear,
    fc2: Linear,
    state: ModelState,
}

impl XorMLP {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        Ok(Self {
            fc1: Linear::new(graph, 2, 4, true, "fc1")?,
            fc2: Linear::new(graph, 4, 2, true, "fc2")?,
            state: ModelState::new(graph),
        })
    }

    /// PyTorch 风格 forward：直接接收 Tensor
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        self.state.forward(x, |input| {
            Ok(self.fc2.forward(&self.fc1.forward(input).tanh()))
        })
    }
}

impl Module for XorMLP {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}
