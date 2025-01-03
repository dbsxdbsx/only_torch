mod loss;
mod ops;
mod variable;

pub(super) use loss::*;
pub(super) use ops::*;
pub(super) use variable::Variable;

use enum_dispatch::enum_dispatch;

#[enum_dispatch]
#[derive(Clone)]
pub(in crate::nn) enum NodeType {
    Variable(Variable),
    Add(Add),
    MatMul(MatMul),
    Step(Step),
    PerceptionLoss(PerceptionLoss),
}

use super::{GraphError, NodeHandle};
use crate::tensor::Tensor;
use std::any::type_name;

#[enum_dispatch(NodeType)]
pub(in crate::nn::nodes) trait TraitNode {
    fn name(&self) -> &str;

    // 根据父节点的值计算本节点的值（注意：由于该接口只在Graph中使用，所以实现时不用关心父节点的值是否已被计算，所有父节点的值可以已预先被计算过了）
    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError>;

    fn value(&self) -> Option<&Tensor>;

    fn set_value(&mut self, _value: Option<&Tensor>) -> Result<(), GraphError> {
        let type_name = type_name::<Self>().split("::").last().unwrap_or("Unknown");
        Err(GraphError::InvalidOperation(format!(
            "{}节点的值只能通过前向传播计算得到，不能直接设置",
            type_name
        )))
    }

    fn calc_jacobi_to_a_parent(
        &self,
        target_parent: &NodeHandle,
        another_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError>;

    fn jacobi(&self) -> Option<&Tensor>;

    fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError>;

    fn clear_jacobi(&mut self) -> Result<(), GraphError> {
        self.set_jacobi(None)
    }

    /// 返回该节点的参数是否应该在训练过程中被更新
    fn is_trainable(&self) -> bool;

    /// 设置该节点的参数是否应该在训练过程中被更新
    fn set_trainable(&mut self, trainable: bool) -> Result<(), GraphError>;

    fn is_inited(&self) -> bool {
        self.value().is_some()
    }

    /// 返回节点的预期输出形状
    /// 这个形状在节点创建时就已确定，存储在节点中
    fn value_expected_shape(&self) -> &[usize];
}
