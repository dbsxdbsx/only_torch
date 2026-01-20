//! 正弦函数拟合模型
//!
//! 使用 MLP 拟合 y = sin(x)

use only_torch::nn::{Graph, GraphError, Linear, Module, Var, VarActivationOps};

/// 正弦拟合 MLP
///
/// 网络结构: Input(1) -> Linear(32, Tanh) -> Linear(1)
pub struct SineMLP {
    fc1: Linear,
    fc2: Linear,
}

impl SineMLP {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        Ok(Self {
            fc1: Linear::new(graph, 1, 32, true, "fc1")?,
            fc2: Linear::new(graph, 32, 1, true, "fc2")?,
        })
    }

    pub fn forward(&self, x: &Var) -> Var {
        self.fc2.forward(&self.fc1.forward(x).tanh())
    }
}

impl Module for SineMLP {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}
