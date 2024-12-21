mod loss;
mod ops;
mod variable;

pub(super) use loss::*;
pub(super) use ops::*;
pub(super) use variable::Variable;

use enum_dispatch::enum_dispatch;

#[enum_dispatch]
pub(in crate::nn::nodes) enum NodeType {
    Variable(Variable),
    Add(Add),
    MatMul(MatMul),
    Step(Step),
    PerceptionLoss(PerceptionLoss),
}

use super::{GraphError, NodeHandle};
use crate::tensor::Tensor;

#[enum_dispatch(NodeType)]
pub(in crate::nn::nodes) trait TraitNode {
    fn name(&self) -> &str;

    // 根据父节点的值计算本节点的值（注意：由于该接口只在Graph中使用，所以实现时不用关心父节点的值是否已被计算，所有父节点的值可以已预先被计算过了）
    fn calc_value_by_parents(&mut self, parents: &[&NodeHandle]) -> Result<(), GraphError>;

    fn value(&self) -> Option<&Tensor>;

    fn set_value(&mut self, _value: Option<&Tensor>) -> Result<(), GraphError> {
        Err(GraphError::InvalidOperation(
            "该类型节点的值不应该被手动设置",
        ))
    }

    fn calc_jacobi_to_a_parent(&self, parent: &NodeHandle) -> Result<Tensor, GraphError>;

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
}
