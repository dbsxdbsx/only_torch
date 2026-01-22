//! MNIST MLP 模型
//!
//! 两层全连接网络用于手写数字识别（PyTorch 风格）

use only_torch::nn::{Graph, GraphError, Linear, ModelState, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

/// MNIST MLP
///
/// 网络结构: Input(784) -> Linear(128, Softplus) -> Linear(10)
pub struct MnistMLP {
    fc1: Linear,
    fc2: Linear,
    state: ModelState,
}

impl MnistMLP {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        Ok(Self {
            // 784 = 28x28 (MNIST 图片展平后的维度)
            fc1: Linear::new(graph, 784, 128, true, "fc1")?,
            fc2: Linear::new(graph, 128, 10, true, "fc2")?,
            state: ModelState::new(graph),
        })
    }

    /// `PyTorch` 风格 forward：直接接收 Tensor
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        self.state.forward(x, |input| {
            let h1 = self.fc1.forward(input).softplus();
            Ok(self.fc2.forward(&h1))
        })
    }
}

impl Module for MnistMLP {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}
