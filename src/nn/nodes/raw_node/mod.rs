pub mod variable;

pub use variable::Variable;

pub mod ops;

pub use ops::*;
pub mod loss;
pub use loss::*;

use enum_dispatch::enum_dispatch;

#[enum_dispatch]
pub enum NodeType {
    Variable(Variable),
    Add(Add),
    MatMul(MatMul),
    Step(Step),
    PerceptionLoss(PerceptionLoss),
}

use super::{GraphError, NodeHandle, NodeId};
use crate::Tensor;

#[enum_dispatch(NodeType)]
pub trait TraitNode {
    fn name(&self) -> &str;

    // 根据父节点的值计算本节点的值（注意：由于该接口只在Graph中使用，所以实现时不用关心父节点的值是否已被计算，所有父节点的值可以已预先被计算过了）
    fn compute_value(&mut self, parents: &[&NodeHandle]) -> Result<(), GraphError>;

    fn value(&self) -> Option<&Tensor>;

    fn set_value(&mut self, value: Option<&Tensor>) -> Result<(), GraphError>;

    fn calc_jacobi_to_a_parent(&self, parent: &NodeHandle) -> Result<Tensor, GraphError>;

    fn jacobi(&self) -> Option<&Tensor>;

    fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError>;

    fn clear_jacobi(&mut self) -> Result<(), GraphError> {
        self.set_jacobi(None)
    }

    fn parents_ids(&self) -> &[NodeId];

    fn children_ids(&self) -> &[NodeId];

    fn is_trainable(&self) -> bool;
}
