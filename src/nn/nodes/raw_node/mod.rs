mod input;
mod loss;
mod ops;
mod parameter;

pub(super) use input::Input;
pub(super) use loss::*;
pub(super) use ops::*;
pub(super) use parameter::Parameter;

use enum_dispatch::enum_dispatch;

#[enum_dispatch]
#[derive(Clone)]
pub(in crate::nn) enum NodeType {
    Input(Input),
    Parameter(Parameter),
    Add(Add),
    MatMul(MatMul),
    Multiply(Multiply),
    ScalarMultiply(ScalarMultiply),
    Sigmoid(Sigmoid),
    Step(Step),
    Tanh(Tanh),
    PerceptionLoss(PerceptionLoss),
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

    /// 计算本节点对父节点的雅可比矩阵
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

    fn is_inited(&self) -> bool {
        self.value().is_some()
    }

    /// 返回节点的预期输出形状
    /// 这个形状在节点创建时就已确定，存储在节点中
    fn value_expected_shape(&self) -> &[usize];
}
