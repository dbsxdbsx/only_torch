//! Iris 分类模型
//!
//! 三层 MLP 用于三分类任务（PyTorch 风格）

use only_torch::nn::{Graph, GraphError, Linear, ModelState, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

/// Iris 分类 MLP
///
/// 网络结构: Input(4) -> Linear(10, Tanh) -> Linear(10, Tanh) -> Linear(3)
pub struct IrisMLP {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    state: ModelState,
}

impl IrisMLP {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        Ok(Self {
            fc1: Linear::new(graph, 4, 10, true, "fc1")?,
            fc2: Linear::new(graph, 10, 10, true, "fc2")?,
            fc3: Linear::new(graph, 10, 3, true, "fc3")?,
            state: ModelState::new(graph),
        })
    }

    /// `PyTorch` 风格 forward：直接接收 Tensor
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        self.state.forward(x, |input| {
            let h1 = self.fc1.forward(input).tanh();
            let h2 = self.fc2.forward(&h1).tanh();
            Ok(self.fc3.forward(&h2))
        })
    }
}

impl Module for IrisMLP {
    fn parameters(&self) -> Vec<Var> {
        [
            self.fc1.parameters(),
            self.fc2.parameters(),
            self.fc3.parameters(),
        ]
        .concat()
    }
}
