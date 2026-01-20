//! XOR 模型定义
//!
//! 使用 Linear 层的简洁实现。
//!
//! ## 网络结构
//! ```text
//! Input(2) -> Linear(4, Tanh) -> Linear(2) -> Softmax
//! ```

use only_torch::nn::{Graph, GraphError, Linear, Module, Var, VarActivationOps};

/// XOR 多层感知机
pub struct XorMLP {
    fc1: Linear,
    fc2: Linear,
}

impl XorMLP {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        Ok(Self {
            fc1: Linear::new_seeded(graph, 2, 4, true, "fc1", 1)?,
            fc2: Linear::new_seeded(graph, 4, 2, true, "fc2", 2)?,
        })
    }

    pub fn forward(&self, x: &Var) -> Var {
        self.fc2.forward(&self.fc1.forward(x).tanh())
    }
}

impl Module for XorMLP {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}
