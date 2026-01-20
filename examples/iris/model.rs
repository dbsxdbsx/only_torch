//! Iris 分类模型
//!
//! 三层 MLP 用于三分类任务

use only_torch::nn::{Graph, GraphError, Linear, Module, Var, VarActivationOps};

/// Iris 分类 MLP
///
/// 网络结构: Input(4) -> Linear(10, Tanh) -> Linear(10, Tanh) -> Linear(3)
pub struct IrisMLP {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl IrisMLP {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        Ok(Self {
            fc1: Linear::new_seeded(graph, 4, 10, true, "fc1", 100)?,
            fc2: Linear::new_seeded(graph, 10, 10, true, "fc2", 200)?,
            fc3: Linear::new_seeded(graph, 10, 3, true, "fc3", 300)?,
        })
    }

    pub fn forward(&self, x: &Var) -> Var {
        let h1 = self.fc1.forward(x).tanh();
        let h2 = self.fc2.forward(&h1).tanh();
        self.fc3.forward(&h2)
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
