mod input;
mod loss;
mod ops;
mod parameter;

pub(super) use input::Input;
pub(super) use loss::{MSELoss, PerceptionLoss, SoftmaxCrossEntropy};
pub use loss::Reduction;
pub(super) use ops::*;
pub(super) use parameter::Parameter;

use enum_dispatch::enum_dispatch;

#[enum_dispatch]
#[derive(Clone)]
pub(in crate::nn) enum NodeType {
    Input(Input),
    Parameter(Parameter),
    Add(Add),
    AvgPool2d(AvgPool2d),
    Conv2d(Conv2d),
    Flatten(Flatten),
    MatMul(MatMul),
    MaxPool2d(MaxPool2d),
    MSELoss(MSELoss),
    Multiply(Multiply),
    Reshape(Reshape),
    ScalarMultiply(ScalarMultiply),
    LeakyReLU(LeakyReLU),
    Sigmoid(Sigmoid),
    SoftPlus(SoftPlus),
    Step(Step),
    Tanh(Tanh),
    PerceptionLoss(PerceptionLoss),
    SoftmaxCrossEntropy(SoftmaxCrossEntropy),
}

use super::{GraphError, NodeHandle, NodeId};
use crate::nn::format_node_display;
use crate::tensor::Tensor;
use std::any::type_name;

#[enum_dispatch(NodeType)]
pub(in crate::nn::nodes) trait TraitNode {
    fn id(&self) -> NodeId;

    fn set_id(&mut self, id: NodeId);

    fn name(&self) -> &str;

    fn set_name(&mut self, name: &str);

    fn get_type_name(&self) -> &'static str {
        type_name::<Self>().split("::").last().unwrap_or("Unknown")
    }

    fn display_node(&self) -> String {
        format_node_display(self.id(), self.name(), self.get_type_name())
    }

    // 根据父节点的值计算本节点的值（注意：由于该接口只在Graph中使用，所以实现时不用关心父节点的值是否已被计算，所有父节点的值可以已预先被计算过了）
    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError>;

    fn value(&self) -> Option<&Tensor>;

    fn set_value(&mut self, _value: Option<&Tensor>) -> Result<(), GraphError> {
        Err(GraphError::InvalidOperation(format!(
            "{}的值只能通过前向传播计算得到，不能直接设置",
            self.display_node()
        )))
    }

    /// 清除节点的值（用于释放内存）
    ///
    /// 与 `set_value(None)` 不同，此方法专门用于内存管理，
    /// 对于不允许直接设置值的节点（如运算节点）也能正常清除。
    fn clear_value(&mut self) -> Result<(), GraphError>;

    // ========== 单样本模式（Jacobi-based）==========

    /// 计算本节点对父节点的雅可比矩阵（单样本模式）
    fn calc_jacobi_to_a_parent(
        &self,
        target_parent: &NodeHandle,
        assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError>;

    fn jacobi(&self) -> Option<&Tensor>;

    fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError>;

    fn clear_jacobi(&mut self) -> Result<(), GraphError> {
        self.set_jacobi(None)
    }

    // ========== Batch 模式（Gradient-based）==========

    /// 计算本节点对父节点的梯度（Batch 模式）
    ///
    /// # 参数
    /// - `target_parent`: 目标父节点
    /// - `upstream_grad`: 从下游传来的梯度，shape 与本节点 value 相同
    /// - `assistant_parent`: 辅助父节点（用于双父节点如 MatMul）
    ///
    /// # 返回
    /// 对 target_parent 的梯度，shape 与 target_parent.value 相同
    ///
    /// # 默认实现
    /// 返回错误，需要各节点自行实现
    fn calc_grad_to_parent(
        &self,
        target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        let _ = (target_parent, upstream_grad, assistant_parent);
        Err(GraphError::InvalidOperation(format!(
            "{}尚未实现 calc_grad_to_parent（Batch 模式）",
            self.display_node()
        )))
    }

    /// 获取节点的梯度（Batch 模式）
    fn grad(&self) -> Option<&Tensor> {
        None // 默认不支持，需要各节点实现
    }

    /// 设置节点的梯度（Batch 模式）
    fn set_grad(&mut self, _grad: Option<&Tensor>) -> Result<(), GraphError> {
        Err(GraphError::InvalidOperation(format!(
            "{}尚未实现 set_grad（Batch 模式）",
            self.display_node()
        )))
    }

    /// 清除节点的梯度
    fn clear_grad(&mut self) -> Result<(), GraphError> {
        self.set_grad(None)
    }

    // ========== 通用方法 ==========

    fn is_inited(&self) -> bool {
        self.value().is_some()
    }

    /// 返回节点的预期输出形状
    /// 这个形状在节点创建时就已确定，存储在节点中
    fn value_expected_shape(&self) -> &[usize];
}
