//! California Housing 房价回归模型
//!
//! 使用 MLP 预测加州房价（PyTorch 风格）

use only_torch::nn::{Graph, GraphError, Linear, ModelState, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

/// California Housing 回归 MLP
///
/// 网络结构: Input(8) -> Linear(128, Softplus) -> Linear(64, Softplus)
///                    -> Linear(32, Softplus) -> Linear(1)
pub struct CaliforniaHousingMLP {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    fc4: Linear,
    state: ModelState,
}

impl CaliforniaHousingMLP {
    /// 创建模型
    ///
    /// # 参数
    /// - `graph`: 计算图（使用 `Graph::new_with_seed` 确保可复现）
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        Ok(Self {
            fc1: Linear::new(graph, 8, 128, true, "fc1")?,
            fc2: Linear::new(graph, 128, 64, true, "fc2")?,
            fc3: Linear::new(graph, 64, 32, true, "fc3")?,
            fc4: Linear::new(graph, 32, 1, true, "fc4")?,
            state: ModelState::new_for::<Self>(graph),
        })
    }

    /// `PyTorch` 风格 forward：直接接收 Tensor
    ///
    /// # 参数
    /// - `x`: 输入特征，形状 `[batch, 8]`
    ///
    /// # 返回
    /// 预测房价，形状 `[batch, 1]`
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        self.state.forward(x, |input| {
            let a1 = self.fc1.forward(input).softplus();
            let a2 = self.fc2.forward(&a1).softplus();
            let a3 = self.fc3.forward(&a2).softplus();
            Ok(self.fc4.forward(&a3))
        })
    }
}

impl Module for CaliforniaHousingMLP {
    fn parameters(&self) -> Vec<Var> {
        [
            self.fc1.parameters(),
            self.fc2.parameters(),
            self.fc3.parameters(),
            self.fc4.parameters(),
        ]
        .concat()
    }
}
