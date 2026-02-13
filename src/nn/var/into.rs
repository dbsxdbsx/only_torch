use super::Var;
use crate::nn::{Graph, GraphError};
use crate::tensor::Tensor;

// ==================== IntoVar ====================

/// 前向传播输入类型转换 trait
///
/// 允许模型的 forward 方法同时接受 `&Tensor` 和 `&Var`，
/// 与 PyTorch 中统一使用 Tensor 的体验类似。
///
/// # 示例
/// ```ignore
/// // 模型定义
/// impl MyModel {
///     pub fn forward(&self, x: impl IntoVar) -> Result<Var, GraphError> {
///         let input = x.into_var(&self.graph)?;
///         let h = self.fc1.forward(&input).relu();
///         Ok(self.fc2.forward(&h))
///     }
/// }
///
/// // 使用时：Tensor 和 Var 都可以
/// let out1 = model.forward(&tensor)?;  // &Tensor
/// let out2 = model.forward(&var)?;     // &Var
/// let out3 = model.forward(var)?;      // Var
/// ```
pub trait IntoVar {
    fn into_var(self, graph: &Graph) -> Result<Var, GraphError>;
}

impl IntoVar for &Tensor {
    fn into_var(self, graph: &Graph) -> Result<Var, GraphError> {
        graph.input(self)
    }
}

impl IntoVar for Tensor {
    fn into_var(self, graph: &Graph) -> Result<Var, GraphError> {
        graph.input(&self)
    }
}

impl IntoVar for &Var {
    fn into_var(self, _graph: &Graph) -> Result<Var, GraphError> {
        Ok(self.clone())
    }
}

impl IntoVar for Var {
    fn into_var(self, _graph: &Graph) -> Result<Var, GraphError> {
        Ok(self)
    }
}
