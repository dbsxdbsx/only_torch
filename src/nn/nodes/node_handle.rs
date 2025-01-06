use super::super::graph::GraphError;
use super::raw_node::{Add, MatMul, PerceptionLoss, Step, Variable};
use super::{NodeType, TraitNode};
use crate::nn::format_node_display;
use crate::tensor::Tensor;

/// 为Graph所管理的节点提供一个句柄，所有内置数据类型必须为二阶张量
#[derive(Clone)]
pub(in crate::nn) struct NodeHandle {
    raw_node: NodeType,
    /// 节点最后一次计算的前向传播次数
    forward_cnt: u64,
}

impl std::fmt::Display for NodeHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.raw_node.display_node())
    }
}

impl NodeHandle {
    pub(in crate::nn) fn new<T: Into<NodeType>>(raw_node: T) -> Result<Self, GraphError> {
        let raw_node = raw_node.into();
        Ok(Self {
            raw_node,
            forward_cnt: 0,
        })
    }

    pub(in crate::nn) fn node_type(&self) -> &NodeType {
        &self.raw_node
    }

    fn check_shape_consistency(&self, new_tensor: &Tensor) -> Result<(), GraphError> {
        if let Some(current_value) = self.raw_node.value() {
            if new_tensor.shape() != current_value.shape() {
                return Err(GraphError::ShapeMismatch {
                    expected: current_value.shape().to_vec(),
                    got: new_tensor.shape().to_vec(),
                    message: format!(
                        "新张量的形状 {:?} 与节点 '{}' 现有张量的形状 {:?} 不匹配。",
                        new_tensor.shape(),
                        self.name(),
                        current_value.shape(),
                    ),
                });
            }
        }
        Ok(())
    }

    pub(in crate::nn) fn bind_id_and_name(&mut self, id: NodeId, name: &str) {
        self.raw_node.set_id(id);
        self.raw_node.set_name(name);
    }

    pub(in crate::nn) fn id(&self) -> NodeId {
        self.raw_node.id()
    }

    pub(in crate::nn) fn name(&self) -> &str {
        self.raw_node.name()
    }

    pub(in crate::nn) fn is_trainable(&self) -> bool {
        self.raw_node.is_trainable()
    }

    pub(in crate::nn) fn set_trainable(&mut self, trainable: bool) -> Result<(), GraphError> {
        self.raw_node.set_trainable(trainable)
    }

    pub(in crate::nn) fn is_inited(&self) -> bool {
        self.raw_node.is_inited()
    }

    pub(in crate::nn) fn value(&self) -> Option<&Tensor> {
        self.raw_node.value()
    }

    pub(in crate::nn) fn set_value(&mut self, value: Option<&Tensor>) -> Result<(), GraphError> {
        if let Some(tensor) = value {
            self.check_shape_consistency(tensor)?;
        }
        self.raw_node.set_value(value)
    }

    pub(in crate::nn) fn jacobi(&self) -> Option<&Tensor> {
        self.raw_node.jacobi()
    }

    pub(in crate::nn) fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError> {
        self.raw_node.set_jacobi(jacobi)
    }

    pub(in crate::nn) fn clear_jacobi(&mut self) -> Result<(), GraphError> {
        self.raw_node.clear_jacobi()
    }

    pub(in crate::nn) fn new_variable(
        shape: &[usize],
        init: bool,
        trainable: bool,
    ) -> Result<Self, GraphError> {
        let variable = Variable::new(shape, init, trainable)?;
        Ok(Self {
            raw_node: NodeType::Variable(variable),
            forward_cnt: 0,
        })
    }

    pub(in crate::nn) fn new_add(
        parents: &[&NodeHandle],
        trainable: bool,
    ) -> Result<Self, GraphError> {
        Self::new(Add::new(parents, trainable)?)
    }

    pub(in crate::nn) fn new_mat_mul(
        parents: &[&NodeHandle],
        trainable: bool,
    ) -> Result<Self, GraphError> {
        Self::new(MatMul::new(parents, trainable)?)
    }

    pub(in crate::nn) fn new_step(
        parents: &[&NodeHandle],
        trainable: bool,
    ) -> Result<Self, GraphError> {
        Self::new(Step::new(parents, trainable)?)
    }

    pub(in crate::nn) fn new_perception_loss(
        parents: &[&NodeHandle],
        trainable: bool,
    ) -> Result<Self, GraphError> {
        Self::new(PerceptionLoss::new(parents, trainable)?)
    }

    pub(in crate::nn) fn calc_value_by_parents(
        &mut self,
        parents: &[NodeHandle],
    ) -> Result<(), GraphError> {
        self.raw_node.calc_value_by_parents(parents)
    }

    /// 计算本节点对父节点的雅可比矩阵
    pub(in crate::nn) fn calc_jacobi_to_a_parent(
        &self,
        parent: &NodeHandle,
        assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        self.raw_node.calc_jacobi_to_a_parent(parent, assistant_parent)
    }

    pub(in crate::nn) fn value_expected_shape(&self) -> &[usize] {
        self.raw_node.value_expected_shape()
    }

    pub(in crate::nn) fn has_value(&self) -> bool {
        self.raw_node.value().is_some()
    }

    pub(in crate::nn) fn forward_cnt(&self) -> u64 {
        self.forward_cnt
    }

    pub(in crate::nn) fn set_forward_cnt(&mut self, cnt: u64) {
        self.forward_cnt = cnt;
    }
}

/// 节点ID
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct NodeId(pub(in crate::nn) u64);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
