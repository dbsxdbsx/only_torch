//! MNIST MLP 模型
//!
//! 两层全连接网络用于手写数字识别

use only_torch::nn::{Graph, GraphError, Linear, Module, Var, VarActivationOps};

/// MNIST MLP
///
/// 网络结构: Input(784) -> Linear(128, Softplus) -> Linear(10)
pub struct MnistMLP {
    fc1: Linear,
    fc2: Linear,
}

impl MnistMLP {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        Ok(Self {
            // 784 = 28x28 (MNIST 图片展平后的维度)
            fc1: Linear::new_seeded(graph, 784, 128, true, "fc1", 100)?,
            fc2: Linear::new_seeded(graph, 128, 10, true, "fc2", 200)?,
        })
    }

    pub fn forward(&self, x: &Var) -> Var {
        let h1 = self.fc1.forward(x).softplus();
        self.fc2.forward(&h1)
    }
}

impl Module for MnistMLP {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}
