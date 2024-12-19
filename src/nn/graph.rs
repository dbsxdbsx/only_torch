/*
 * @Author       : 老董
 * @Date         : 2024-01-31 17:57:13
 * @LastEditors  : 老董
 * @LastEditTime : 2024-12-19 15:53:32
 * @Description  : 神经网络模型的计算图
 */

use super::nodes::{Add, MatMul, Variable};
use super::{NodeHandle, NodeId, TraitNode};
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// 图的完整定义
pub struct Graph {
    nodes: HashMap<NodeId, NodeHandle>,
    edges: HashMap<NodeId, HashSet<NodeId>>,
    next_id: u64,
    is_training: bool,
}

impl Graph {
    // 基本
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            next_id: 0,
            is_training: true, // 默认为训练模式
        }
    }
    pub fn nodes(&self) -> &HashMap<NodeId, NodeHandle> {
        &self.nodes
    }

    pub fn is_forwarded(&self, node_id: NodeId) -> Result<bool, GraphError> {
        self.nodes
            .get(&node_id)
            .map(|node| node.value().is_some())
            .ok_or(GraphError::NodeNotFound(node_id))
    }

    // 前向传播：
    pub fn forward_node(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        // 1. 如果已经计算过且不需要重新计算，直接返回
        if self.is_forwarded(node_id)? {
            return Ok(());
        }

        // 2. 递归计算所有父节点
        let parents_ids = &self.get_node(node_id)?.parents().to_vec();
        for parent_id in parents_ids {
            self.forward_node(*parent_id)?;
        }

        // 3. 计算当前节点
        let node = self.get_node_mut(node_id)?;
        node.compute_value(parents_ids)?;

        // 4.返回
        Ok(())
    }

    /// 验证父子节点关系
    fn validate_parent_child(&self, child_id: NodeId, parent_id: NodeId) -> Result<(), GraphError> {
        let child = self.get_node(child_id)?;
        if !child.parents().contains(&parent_id) {
            return Err(GraphError::InvalidOperation("无效的父子节点关系"));
        }
        Ok(())
    }

    /// 反向传播：计算结果节点对本节点的雅可比矩阵
    /// NOTE: 这里的逻辑参考了https://github.com/zc911/MatrixSlow/blob/a6db0d38802004449941e6644e609a2455b26327/matrixslow/core/node.py#L83
    pub fn backward_node(&mut self, node_id: NodeId, result_id: NodeId) -> Result<(), GraphError> {
        // 1. 如果已经计算过，则直接返回
        if self.get_node(node_id)?.jacobi().is_some() {
            return Ok(());
        }

        // 2. 如果节点是结果节点（是自身），则自己对自己的雅可比为单位矩阵
        if node_id == result_id {
            let dim = self
                .get_node(node_id)?
                .value()
                .ok_or_else(|| GraphError::ComputationError("节点没有值".to_string()))?
                .size();
            let eye = Tensor::eyes(dim);
            self.get_node_mut(node_id)?.set_jacobi(Some(&eye))?;
            return Ok(());
        }

        // 3. 其他情况的雅可比矩阵计算
        // 3.1 先将雅可比矩阵初始化为零矩阵
        {
            let (result_dim, node_dim) = {
                let result_node = self.get_node(result_id)?;
                let node = self.get_node(node_id)?;
                (
                    result_node.value().map(|v| v.size()).ok_or_else(|| {
                        GraphError::ComputationError("结果节点没有值".to_string())
                    })?,
                    node.value()
                        .map(|v| v.size())
                        .ok_or_else(|| GraphError::ComputationError("节点没有值".to_string()))?,
                )
            };
            let zeros = Tensor::zeros(&[result_dim, node_dim]);
            self.get_node_mut(node_id)?.set_jacobi(Some(&zeros))?;
        }

        // 3.2 计算所有子节点的梯度（雅可比矩阵）对当前节点的贡献
        let child_ids = self.get_node(node_id)?.children().to_vec();
        for child_id in child_ids {
            // 3.2.1 先计算子节点对结果节点的梯度（雅可比矩阵）
            self.backward_node(child_id, result_id)?;

            // 3.2.2 计算子节点对当前节点的梯度（雅可比矩阵）贡献
            let contribution = {
                self.validate_parent_child(child_id, node_id)?;
                let child = self.get_node(child_id)?;
                let local_jacobi = child.calc_jacobi_to_a_parent(node_id)?;
                let child_jacobi = child.jacobi().unwrap();
                child_jacobi * local_jacobi
            };

            // 3.2.3 更新当前节点的梯度（雅可比矩阵）
            {
                let node = self.get_node_mut(node_id)?;
                let current = node.jacobi().unwrap();
                node.set_jacobi(Some(&(current + contribution)))?;
            }
        }

        Ok(())
    }

    pub fn zero_grad(&mut self) -> Result<(), GraphError> {
        for node in self.nodes.values_mut() {
            node.clear_jacobi()?;
        }
        Ok(())
    }

    pub fn reset_from(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        // 1. 获取所有下游节点
        let mut to_reset = Vec::new();
        self.collect_downstream_nodes(node_id, &mut to_reset)?;

        // 2. 重置所有下游节点值
        for id in to_reset {
            let node = self
                .nodes
                .get_mut(&id)
                .ok_or(GraphError::NodeNotFound(id))?;
            node.set_value(None)?;
        }

        Ok(())
    }

    // TODO: 若碰到rnn咋办，
    // 参考matrixslow下的def reset_value(self, recursive=True):
    fn collect_downstream_nodes(
        &self,
        start: NodeId,
        result: &mut Vec<NodeId>,
    ) -> Result<(), GraphError> {
        // 1. 如果已经包含，则直接返回
        if result.contains(&start) {
            return Ok(());
        }

        // 2. 将当前节点加入结果
        result.push(start);

        // 3. 所有子节点递归处理
        if let Some(children) = self.edges.get(&start) {
            for child in children {
                self.collect_downstream_nodes(*child, result)?;
            }
        }

        Ok(())
    }

    fn generate_node_id(&mut self) -> NodeId {
        // 生成唯一的节点ID
        self.next_id += 1;
        NodeId(self.next_id)
    }

    // 用于调试
    pub fn nodes_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get a reference to a node by its ID
    pub fn get_node(&self, id: NodeId) -> Result<&NodeHandle, GraphError> {
        self.nodes.get(&id).ok_or(GraphError::NodeNotFound(id))
    }

    /// Get a mutable reference to a node by its ID
    pub fn get_node_mut(&mut self, id: NodeId) -> Result<&mut NodeHandle, GraphError> {
        self.nodes.get_mut(&id).ok_or(GraphError::NodeNotFound(id))
    }

    /// Get a node's value by its ID, handling both node lookup and value extraction
    pub fn get_node_value(&self, id: NodeId) -> Result<&Tensor, GraphError> {
        self.get_node(id)?
            .value()
            .ok_or_else(|| GraphError::ComputationError(format!("节点 {} 没有值", id.0)))
    }
}

// 图模式相关
impl Graph {
    // 训练模式控制方法
    pub fn train(&mut self) {
        self.is_training = true;
    }

    pub fn eval(&mut self) {
        self.is_training = false;
    }

    pub fn is_training(&self) -> bool {
        self.is_training
    }
}

// 便捷的节点构建方法
impl Graph {
    pub fn add_node(&mut self, node: impl Into<NodeType> + TraitNode + 'static) -> &NodeHandle {
        let id = self.generate_node_id();
        let graph_id = GraphId::new();

        // Create and store node handle
        let handle = NodeHandle::new(id, graph_id, node);

        // Register parent-child relationships
        for parent_id in handle.parents() {
            self.edges.entry(*parent_id).or_default().insert(id);
        }

        self.nodes.insert(id, handle);
        self.nodes.get(&id).unwrap()
    }

    pub fn variable(
        &mut self,
        shape: &[usize],
        init: bool,
        trainable: bool,
        name: Option<&str>,
    ) -> &NodeHandle {
        self.add_node(Variable::new(shape, init, trainable, name))
    }

    pub fn add(&mut self, parents: &[&NodeHandle], name: Option<&str>) -> &NodeHandle {
        self.add_node(Add::new(
            &parents.iter().map(|h| h.id()).collect::<Vec<_>>(),
            name,
        ))
    }

    pub fn mat_mul(&mut self, a: &NodeHandle, b: &NodeHandle, name: Option<&str>) -> &NodeHandle {
        self.add_node(MatMul::new(&[a.id(), b.id()], name))
    }
}

/// 图错误类型
#[derive(Debug)]
pub enum GraphError {
    GraphNotFound(String),
    NodeNotFound(NodeId),
    InvalidOperation(&'static str),
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    ComputationError(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphId(pub u64);

static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(0);
static GRAPH_REGISTRY: OnceLock<Mutex<HashMap<GraphId, Graph>>> = OnceLock::new();

impl GraphId {
    pub fn new() -> Self {
        GraphId(NEXT_GRAPH_ID.fetch_add(1, Ordering::SeqCst))
    }
}

// 初始化 GRAPH_REGISTRY
pub(crate) fn init_or_get_graph_registry() -> &'static Mutex<HashMap<GraphId, Graph>> {
    GRAPH_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_registry_init_and_get() {
        // 使用示例
        let registry = init_or_get_graph_registry();
        let mut map = registry.lock().unwrap();
        let graph_id = GraphId::new();
        map.insert(graph_id, Graph::new());

        // 验证插入成功
        assert!(map.contains_key(&graph_id));

        // 验证图的初始状态
        let graph = map.get(&graph_id).unwrap();
        assert!(graph.nodes().is_empty());
        assert!(graph.is_training);
    }
}
