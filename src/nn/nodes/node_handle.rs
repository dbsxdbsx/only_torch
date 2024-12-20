use super::super::graph::{GraphError, GraphId};
use super::raw_node::{Add, MatMul, PerceptionLoss, Step, Variable};
use super::{NodeType, TraitNode};
use crate::nn::graph::init_or_get_graph_registry;
use crate::tensor::Tensor;

/// 为Graph所管理的节点提供一个句柄，所有内置数据类型必须为二阶张量
pub struct NodeHandle {
    id: NodeId,
    graph_id: GraphId,
    raw_node: NodeType,
    parents_ids: Vec<NodeId>,
    children_ids: Vec<NodeId>,
}

impl NodeHandle {
    /// 检查张量是否为二阶
    fn check_tensor_shape(tensor: &Tensor) -> Result<(), GraphError> {
        if tensor.shape().len() != 2 {
            return Err(GraphError::ShapeMismatch {
                expected: vec![2],
                got: vec![tensor.shape().len()],
            });
        }
        Ok(())
    }

    pub(crate) fn new<T: Into<NodeType>>(
        id: NodeId,
        graph_id: GraphId,
        raw_node: T,
        parents: &[NodeId],
    ) -> Result<Self, GraphError> {
        let raw_node = raw_node.into();

        // 检查节点当前值的维度
        if let Some(value) = raw_node.value() {
            if let Err(e) = Self::check_tensor_shape(value) {
                panic!("NodeHandle要求所有张量必须为二阶张量: {:?}", e);
            }
        }

        let mut handle = Self {
            id,
            graph_id,
            raw_node,
            parents_ids: Vec::new(),
            children_ids: Vec::new(),
        };

        // 设置并验证父节点关系
        handle.set_parents_and_validate(parents)?;

        Ok(handle)
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn graph_id(&self) -> GraphId {
        self.graph_id
    }

    pub fn name(&self) -> &str {
        self.raw_node.name()
    }

    pub fn is_trainable(&self) -> bool {
        self.raw_node.is_trainable()
    }

    pub fn is_inited(&self) -> bool {
        self.raw_node.is_inited()
    }

    pub fn value(&self) -> Option<&Tensor> {
        self.raw_node.value()
    }

    pub fn set_value(&mut self, value: Option<&Tensor>) -> Result<(), GraphError> {
        if let Some(tensor) = value {
            Self::check_tensor_shape(tensor)?;
        }
        self.raw_node.set_value(value)
    }

    pub fn jacobi(&self) -> Option<&Tensor> {
        self.raw_node.jacobi()
    }

    pub fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError> {
        if let Some(tensor) = jacobi {
            Self::check_tensor_shape(tensor)?;
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

    pub fn calc_value_by_parents(&mut self, parents_ids: &[NodeId]) -> Result<(), GraphError> {
        let registry = init_or_get_graph_registry();
        let map = registry.lock().unwrap();
        let graph = map.get(&self.graph_id).unwrap();

        let parent_handles: Vec<&NodeHandle> = parents_ids
            .iter()
            .map(|id| graph.get_node(*id))
            .collect::<Result<Vec<_>, _>>()?;

        self.raw_node.calc_value_by_parents(&parent_handles)
    }

    pub fn calc_jacobi_to_a_parent(&self, parent_id: NodeId) -> Result<Tensor, GraphError> {
        let registry = init_or_get_graph_registry();
        let map = registry.lock().unwrap();
        let graph = map.get(&self.graph_id).unwrap();

        let parent_handle = graph.get_node(parent_id)?;
        self.raw_node.calc_jacobi_to_a_parent(parent_handle)
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

    // Node creation methods
    pub(crate) fn new_variable(
        id: NodeId,
        graph_id: GraphId,
        shape: &[usize],
        init: bool,
        trainable: bool,
        name: &str,
    ) -> Result<Self, GraphError> {
        Self::new(
            id,
            graph_id,
            Variable::new(shape, init, trainable, name),
            &[],
        )
    }

    pub(crate) fn new_add(
        id: NodeId,
        graph_id: GraphId,
        name: &str,
        parents: &[NodeId],
    ) -> Result<Self, GraphError> {
        Self::new(id, graph_id, Add::new(name), parents)
    }

    pub(crate) fn new_mat_mul(
        id: NodeId,
        graph_id: GraphId,
        name: &str,
        parents: &[NodeId],
    ) -> Result<Self, GraphError> {
        Self::new(id, graph_id, MatMul::new(name), parents)
    }

    pub(crate) fn new_step(
        id: NodeId,
        graph_id: GraphId,
        name: &str,
        parents: &[NodeId],
    ) -> Result<Self, GraphError> {
        Self::new(id, graph_id, Step::new(name), parents)
    }

    pub(crate) fn new_perception_loss(
        id: NodeId,
        graph_id: GraphId,
        name: &str,
        parents: &[NodeId],
    ) -> Result<Self, GraphError> {
        Self::new(id, graph_id, PerceptionLoss::new(name), parents)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);
