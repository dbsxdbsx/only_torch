//! 正弦函数拟合模型
//!
//! 使用 MLP 拟合 y = sin(x)（PyTorch 风格）

use only_torch::nn::{Graph, GraphError, Linear, ModelState, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

/// 正弦拟合 MLP
///
/// 网络结构: Input(1) -> Linear(32, Tanh) -> Linear(1)
pub struct SineMLP {
    fc1: Linear,
    fc2: Linear,
    state: ModelState,
}

impl SineMLP {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        Ok(Self {
            fc1: Linear::new(graph, 1, 32, true, "fc1")?,
            fc2: Linear::new(graph, 32, 1, true, "fc2")?,
            state: ModelState::new(graph),
        })
    }

    /// `PyTorch` 风格 forward：直接接收 Tensor
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        self.state.forward(x, |input| {
            Ok(self.fc2.forward(&self.fc1.forward(input).tanh()))
        })
    }
}

impl Module for SineMLP {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}
