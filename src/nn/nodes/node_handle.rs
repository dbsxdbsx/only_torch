use super::super::graph::{GraphError, GraphId};
use super::raw_node::NodeType;
use super::Graph;
use crate::nn::graph::init_or_get_graph_registry;
use crate::nn::nodes::raw_node::TraitNode;
use crate::tensor::Tensor;

pub struct NodeHandle {
    id: NodeId,
    graph_id: GraphId,
    raw_node: NodeType,
}

impl NodeHandle {
    pub fn new<T: Into<NodeType>>(id: NodeId, graph_id: GraphId, raw_node: T) -> Self {
        Self {
            id,
            graph_id,
            raw_node: raw_node.into(),
        }
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn graph_id(&self) -> GraphId {
        self.graph_id
    }

    pub fn value(&self) -> Option<&Tensor> {
        self.raw_node.value()
    }

    pub fn set_value(&mut self, value: Option<&Tensor>) -> Result<(), GraphError> {
        self.raw_node.set_value(value)
    }

    pub fn jacobi(&self) -> Option<&Tensor> {
        self.raw_node.jacobi()
    }

    pub fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError> {
        self.raw_node.set_jacobi(jacobi)
    }

    pub fn clear_jacobi(&mut self) -> Result<(), GraphError> {
        self.raw_node.clear_jacobi()
    }

    pub fn parents(&self) -> &[NodeId] {
        self.raw_node.parents_ids()
    }

    pub fn children(&self) -> &[NodeId] {
        self.raw_node.children_ids()
    }

    pub fn compute_value(&mut self, parent_ids: &[NodeId]) -> Result<(), GraphError> {
        let registry = init_or_get_graph_registry();
        let map = registry.lock().unwrap();
        let graph = map.get(&self.graph_id).unwrap();

        let parent_handles: Vec<&NodeHandle> = parent_ids
            .iter()
            .map(|id| graph.get_node(*id))
            .collect::<Result<Vec<_>, _>>()?;

        self.raw_node.compute_value(&parent_handles)
    }

    pub fn calc_jacobi_to_a_parent(&self, parent_id: NodeId) -> Result<Tensor, GraphError> {
        let registry = init_or_get_graph_registry();
        let map = registry.lock().unwrap();
        let graph = map.get(&self.graph_id).unwrap();

        let parent_handle = graph.get_node(parent_id)?;
        self.raw_node.calc_jacobi_to_a_parent(parent_handle)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);
