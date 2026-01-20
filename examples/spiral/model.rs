//! 螺旋分类模型
//!
//! 使用较深的 MLP 学习非线性决策边界

use only_torch::nn::{Graph, GraphError, Linear, Module, Var, VarActivationOps};

/// 螺旋分类 MLP
///
/// 网络结构: Input(2) -> Linear(16, Tanh) -> Linear(16, Tanh) -> Linear(2)
pub struct SpiralMLP {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl SpiralMLP {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        Ok(Self {
            fc1: Linear::new(graph, 2, 16, true, "fc1")?,
            fc2: Linear::new(graph, 16, 16, true, "fc2")?,
            fc3: Linear::new(graph, 16, 2, true, "fc3")?,
        })
    }

    pub fn forward(&self, x: &Var) -> Var {
        let h1 = self.fc1.forward(x).tanh();
        let h2 = self.fc2.forward(&h1).tanh();
        self.fc3.forward(&h2)
    }
}

impl Module for SpiralMLP {
    fn parameters(&self) -> Vec<Var> {
        [
            self.fc1.parameters(),
            self.fc2.parameters(),
            self.fc3.parameters(),
        ]
        .concat()
    }
}
