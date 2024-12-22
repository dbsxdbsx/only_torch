use super::super::graph::GraphError;
use super::raw_node::{Add, MatMul, PerceptionLoss, Step, Variable};
use super::{NodeType, TraitNode};
use crate::tensor::Tensor;

/// 为Graph所管理的节点提供一个句柄，所有内置数据类型必须为二阶张量
#[derive(Clone)]
pub(in crate::nn) struct NodeHandle {
    id: Option<NodeId>,
    name: Option<String>,
    raw_node: NodeType,
    parents_ids: Vec<NodeId>,
    children_ids: Vec<NodeId>,
}

impl NodeHandle {
    fn check_tensor_dimension(tensor: &Tensor) -> Result<(), GraphError> {
        if tensor.dimension() != 2 {
            return Err(GraphError::ShapeMismatch {
                expected: vec![2],
                got: vec![tensor.dimension()],
                message: format!(
                    "神经网络中的张量必须是二维的（矩阵），但收到的张量维度是 {} 维。",
                    tensor.dimension(),
                ),
            });
        }
        Ok(())
    }

    fn check_shape_compatibility(&self, new_tensor: &Tensor) -> Result<(), GraphError> {
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

    pub(in crate::nn) fn bind_id_and_name(
        &mut self,
        id: NodeId,
        name: &str,
    ) -> Result<(), GraphError> {
        if self.id.is_some() || self.name.is_some() {
            return Err(GraphError::InvalidOperation("节点已经绑定了ID和名称"));
        }
        self.id = Some(id);
        self.name = Some(name.to_string());
        Ok(())
    }

    pub fn id(&self) -> NodeId {
        self.id.expect("节点ID未初始化，这是一个内部错误")
    }

    pub fn name(&self) -> &str {
        self.name
            .as_ref()
            .expect("节点名称未初始化，这是一个内部错误")
    }

    pub fn is_trainable(&self) -> bool {
        self.raw_node.is_trainable()
    }

    pub fn set_trainable(&mut self, trainable: bool) -> Result<(), GraphError> {
        self.raw_node.set_trainable(trainable)
    }

    pub fn is_inited(&self) -> bool {
        self.raw_node.is_inited()
    }

    pub fn value(&self) -> Option<&Tensor> {
        self.raw_node.value()
    }

    pub fn set_value(&mut self, value: Option<&Tensor>) -> Result<(), GraphError> {
        if let Some(tensor) = value {
            self.check_shape_compatibility(tensor)?;
        }
        self.raw_node.set_value(value)
    }

    pub fn jacobi(&self) -> Option<&Tensor> {
        self.raw_node.jacobi()
    }

    pub fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError> {
        if let Some(tensor) = jacobi {
            Self::check_tensor_dimension(tensor)?;
        }
        self.raw_node.set_jacobi(jacobi)
    }

    pub fn clear_jacobi(&mut self) -> Result<(), GraphError> {
        self.raw_node.clear_jacobi()
    }

    pub fn add_parent_id(&mut self, parent_id: NodeId) {
        if !self.parents_ids.contains(&parent_id) {
            self.parents_ids.push(parent_id);
        }
    }

    pub fn add_child_id(&mut self, child_id: NodeId) {
        if !self.children_ids.contains(&child_id) {
            self.children_ids.push(child_id);
        }
    }

    pub fn remove_parent_id(&mut self, parent_id: NodeId) {
        if let Some(pos) = self.parents_ids.iter().position(|&x| x == parent_id) {
            self.parents_ids.remove(pos);
        }
    }

    pub fn remove_child_id(&mut self, child_id: NodeId) {
        if let Some(pos) = self.children_ids.iter().position(|&x| x == child_id) {
            self.children_ids.remove(pos);
        }
    }

    pub fn parents_ids(&self) -> &[NodeId] {
        &self.parents_ids
    }

    pub fn children_ids(&self) -> &[NodeId] {
        &self.children_ids
    }

    pub fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
        self.raw_node.calc_value_by_parents(parents)
    }

    pub fn calc_jacobi_to_a_parent(
        &self,
        parent: &NodeHandle,
        another_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        self.raw_node
            .calc_jacobi_to_a_parent(parent, another_parent)
    }

    fn set_parents_and_validate(&mut self, parents: &[NodeId]) -> Result<(), GraphError> {
        self.parents_ids = parents.to_vec();

        match &self.raw_node {
            NodeType::Add(_) => self.validate_add_parents(),
            NodeType::MatMul(_) => self.validate_mat_mul_parents(),
            NodeType::Step(_) => self.validate_step_parents(),
            NodeType::PerceptionLoss(_) => self.validate_perception_loss_parents(),
            NodeType::Variable(_) => self.validate_variable_parents(),
        }
    }

    fn validate_add_parents(&self) -> Result<(), GraphError> {
        if self.parents_ids.len() < 2 {
            return Err(GraphError::InvalidOperation("Add节点至少需2个父节点"));
        }
        Ok(())
    }

    fn validate_mat_mul_parents(&self) -> Result<(), GraphError> {
        if self.parents_ids.len() != 2 {
            return Err(GraphError::InvalidOperation("MatMul节点需要正好2个父节点"));
        }
        Ok(())
    }

    fn validate_step_parents(&self) -> Result<(), GraphError> {
        if self.parents_ids.len() != 1 {
            return Err(GraphError::InvalidOperation("Step节点只需要1个父节点"));
        }
        Ok(())
    }

    fn validate_perception_loss_parents(&self) -> Result<(), GraphError> {
        if self.parents_ids.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "PerceptionLoss节点只需要1个父节点",
            ));
        }
        Ok(())
    }

    fn validate_variable_parents(&self) -> Result<(), GraphError> {
        if !self.parents_ids.is_empty() {
            return Err(GraphError::InvalidOperation("Variable节点不能有父节点"));
        }
        Ok(())
    }

    pub(in crate::nn) fn new_variable(
        shape: &[usize],
        init: bool,
        trainable: bool,
    ) -> Result<Self, GraphError> {
        Self::new(Variable::new(shape, init, trainable, ""), &[])
    }

    pub(in crate::nn) fn new_add(parents: &[NodeId], trainable: bool) -> Result<Self, GraphError> {
        Self::new(Add::new("", trainable), parents)
    }

    pub(in crate::nn) fn new_mat_mul(
        parents: &[NodeId],
        trainable: bool,
    ) -> Result<Self, GraphError> {
        Self::new(MatMul::new("", trainable, parents), parents)
    }

    pub(in crate::nn) fn new_step(parents: &[NodeId], trainable: bool) -> Result<Self, GraphError> {
        Self::new(Step::new("", trainable), parents)
    }

    pub(in crate::nn) fn new_perception_loss(
        parents: &[NodeId],
        trainable: bool,
    ) -> Result<Self, GraphError> {
        Self::new(PerceptionLoss::new("", trainable), parents)
    }

    pub fn node_type(&self) -> &NodeType {
        &self.raw_node
    }

    pub(in crate::nn) fn new<T: Into<NodeType>>(
        raw_node: T,
        parents: &[NodeId],
    ) -> Result<Self, GraphError> {
        // 检查节点当前值的维度
        let raw_node = raw_node.into();
        if let Some(value) = raw_node.value() {
            if let Err(e) = Self::check_tensor_dimension(value) {
                return Err(GraphError::DimensionMismatch {
                    expected: 2,
                    got: value.dimension(),
                    message: format!("NodeHandle要求所有张量必须为二阶张量: {:?}", e),
                });
            }
        }

        // 设置并验证父节点关系
        let mut handle = Self {
            id: None,
            name: None,
            raw_node,
            parents_ids: Vec::new(),
            children_ids: Vec::new(),
        };
        handle.set_parents_and_validate(parents)?;

        Ok(handle)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);
