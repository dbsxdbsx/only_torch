//! 双输入加法模型定义
//!
//! 展示 `ModelState::forward2` 的使用方式。
//!
//! ## 网络结构
//! ```text
//! Input1(1) -> Linear(4) -+
//!                         +--> Stack(concat) -> Linear(1)
//! Input2(1) -> Linear(4) -+
//! ```

use only_torch::nn::{Graph, GraphError, Linear, ModelState, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

/// 双输入加法模型
///
/// 学习预测两个数的和
pub struct DualInputAdder {
    fc1: Linear,
    fc2: Linear,
    fc_out: Linear,
    state: ModelState,
}

impl DualInputAdder {
    pub fn new(graph: &Graph) -> Result<Self, GraphError> {
        Ok(Self {
            fc1: Linear::new(graph, 1, 4, true, "fc1")?,
            fc2: Linear::new(graph, 1, 4, true, "fc2")?,
            fc_out: Linear::new(graph, 8, 1, true, "fc_out")?,
            state: ModelState::new_for::<Self>(graph),
        })
    }

    /// 双输入 forward：接收两个 Tensor，预测它们的和
    pub fn forward(&self, x1: &Tensor, x2: &Tensor) -> Result<Var, GraphError> {
        self.state.forward2(x1, x2, |a, b| {
            let h1 = self.fc1.forward(a).relu();
            let h2 = self.fc2.forward(b).relu();
            // 拼接两个分支的特征
            let combined = Var::stack(&[&h1, &h2], 1, false)?;
            Ok(self.fc_out.forward(&combined))
        })
    }
}

impl Module for DualInputAdder {
    fn parameters(&self) -> Vec<Var> {
        [
            self.fc1.parameters(),
            self.fc2.parameters(),
            self.fc_out.parameters(),
        ]
        .concat()
    }
}
