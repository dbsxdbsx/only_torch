/*
 * @Author       : 老董
 * @Date         : 2024-01-31 17:57:13
 * @LastEditors  : 老董
 * @LastEditTime : 2024-12-21 16:25:25
 * @Description  : 神经网络模型的计算图
 */

use super::{NodeHandle, NodeId};
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// 图的完整定义
pub struct Graph {
    id: GraphId,
    name: String,
    nodes: HashMap<NodeId, NodeHandle>,
    edges: HashMap<NodeId, HashSet<NodeId>>,
    next_id: u64,
    is_training: bool,
}

impl Graph {
    fn check_duplicate_id(id: GraphId) -> Result<(), GraphError> {
        let registry = init_or_get_graph_registry();
        let guard = registry.lock().unwrap();

        // 检查是否有重复ID的图
        if guard.contains_key(&id) {
            return Err(GraphError::DuplicateId(id));
        }
        Ok(())
    }

    fn check_duplicate_name(name: &str) -> Result<(), GraphError> {
        let registry = init_or_get_graph_registry();
        let guard = registry.lock().unwrap();

        // 检查是否有重名的图
        if guard.values().any(|g| g.name() == name) {
            return Err(GraphError::DuplicateName(name.to_string()));
        }
        Ok(())
    }

    fn check_duplicate_node_name(&self, name: &str) -> Result<(), GraphError> {
        if self.nodes.values().any(|node| node.name() == name) {
            return Err(GraphError::DuplicateNodeName(format!(
                "节点{}在图{}中重复",
                name,
                self.name()
            )));
        }
        Ok(())
    }

    fn generate_node_name(&self, base_name: &str, node_type: &str) -> String {
        if !base_name.is_empty() {
            let name = format!("{}_{}", self.name(), base_name);
            if self.check_duplicate_node_name(&name).is_ok() {
                return name;
            }
        }

        let mut counter = 1;
        loop {
            let name = format!("{}_{}_{}", self.name(), node_type, counter);
            if self.check_duplicate_node_name(&name).is_ok() {
                return name;
            }
            counter += 1;
        }
    }

    // 基本
    pub fn new() -> Self {
        Self::with_name("default_graph").unwrap()
    }

    pub fn with_name(name: &str) -> Result<Self, GraphError> {
        // 1. 生成新ID并检查是否重复
        let id = GraphId::new();
        Self::check_duplicate_id(id)?;

        // 2. 检查名称是否重复
        Self::check_duplicate_name(name)?;

        // 3. 创建新图
        Ok(Self {
            id,
            name: name.to_string(),
            nodes: HashMap::new(),
            edges: HashMap::new(),
            next_id: 0,
            is_training: true,
        })
    }

    pub fn id(&self) -> GraphId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
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
        let parents_ids = &self.get_node(node_id)?.parents_ids().to_vec();
        for parent_id in parents_ids {
            self.forward_node(*parent_id)?;
        }

        // 3. 计算当前节点
        let node = self.get_node_mut(node_id)?;
        node.calc_value_by_parents(parents_ids)?;

        // 4.返回
        Ok(())
    }

    /// 验证父子节点关系
    fn validate_parent_child(&self, child_id: NodeId, parent_id: NodeId) -> Result<(), GraphError> {
        let child = self.get_node(child_id)?;
        if !child.parents_ids().contains(&parent_id) {
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
        let child_ids = self.get_node(node_id)?.children_ids().to_vec();
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

    pub fn clear_jacobi(&mut self) -> Result<(), GraphError> {
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

    /// Set a node's value by its ID
    pub fn set_node_value(&mut self, id: NodeId, value: Option<&Tensor>) -> Result<(), GraphError> {
        self.get_node_mut(id)?.set_value(value)
    }

    /// Get a node's jacobi matrix by its ID
    pub fn get_node_jacobi(&self, id: NodeId) -> Result<&Tensor, GraphError> {
        self.get_node(id)?
            .jacobi()
            .ok_or_else(|| GraphError::ComputationError(format!("节点 {} 没有雅可比矩阵", id.0)))
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
    fn add_node(&mut self, node_handle: NodeHandle) -> Result<NodeId, GraphError> {
        let node_id = node_handle.id();

        // 1. 检查节点名称是否重复
        if let Err(e) = self.check_duplicate_node_name(node_handle.name()) {
            return Err(e);
        }

        // 2. 注册父-子关系
        for parent_id in node_handle.parents_ids() {
            self.edges.entry(*parent_id).or_default().insert(node_id);
        }

        // 3. 将 node_handle 插入到 nodes 中，并返回 id
        self.nodes.insert(node_id, node_handle);
        Ok(node_id)
    }

    pub fn new_variable(
        &mut self,
        shape: &[usize],
        init: bool,
        trainable: bool,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let node_name = self.generate_node_name(name.unwrap_or(""), "var");
        let node_id = self.generate_node_id();
        let handle =
            NodeHandle::new_variable(node_id, self.id, shape, init, trainable, &node_name)?;
        self.add_node(handle)
    }

    pub fn new_add(
        &mut self,
        parents_ids: &[NodeId],
        name: Option<&str>,
        trainable: bool,
    ) -> Result<NodeId, GraphError> {
        let node_name = self.generate_node_name(name.unwrap_or(""), "add");
        let node_id = self.generate_node_id();
        let handle = NodeHandle::new_add(node_id, self.id, &node_name, parents_ids, trainable)?;
        self.add_node(handle)
    }

    pub fn new_mat_mul(
        &mut self,
        left_node_id: NodeId,
        right_node_id: NodeId,
        name: Option<&str>,
        trainable: bool,
    ) -> Result<NodeId, GraphError> {
        let node_name = self.generate_node_name(name.unwrap_or(""), "matmul");
        let node_id = self.generate_node_id();
        let handle = NodeHandle::new_mat_mul(
            node_id,
            self.id,
            &node_name,
            &[left_node_id, right_node_id],
            trainable,
        )?;
        self.add_node(handle)
    }

    pub fn new_step(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
        trainable: bool,
    ) -> Result<NodeId, GraphError> {
        let node_name = self.generate_node_name(name.unwrap_or(""), "step");
        let node_id = self.generate_node_id();
        let handle = NodeHandle::new_step(node_id, self.id, &node_name, &[parent_id], trainable)?;
        self.add_node(handle)
    }

    pub fn new_perception_loss(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
        trainable: bool,
    ) -> Result<NodeId, GraphError> {
        let node_name = self.generate_node_name(name.unwrap_or(""), "perception_loss");
        let node_id = self.generate_node_id();
        let handle =
            NodeHandle::new_perception_loss(node_id, self.id, &node_name, &[parent_id], trainable)?;
        self.add_node(handle)
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
        message: String,
    },
    ComputationError(String),
    DuplicateName(String),
    DuplicateId(GraphId),
    DuplicateNodeName(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphId(pub u64);

static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(0);
static GRAPH: OnceLock<Mutex<HashMap<GraphId, Graph>>> = OnceLock::new();

impl GraphId {
    pub fn new() -> Self {
        GraphId(NEXT_GRAPH_ID.fetch_add(1, Ordering::SeqCst))
    }
}

// 初始化全局GRAPH
pub(crate) fn init_or_get_graph_registry() -> &'static Mutex<HashMap<GraphId, Graph>> {
    GRAPH.get_or_init(|| Mutex::new(HashMap::new()))
}

#[macro_export]
macro_rules! with_graph {
    ($graph_id:expr, $f:expr) => {{
        let registry = $crate::nn::graph::init_or_get_graph_registry();
        let guard = registry.lock().unwrap();
        let graph = guard.get(&$graph_id).unwrap();
        $f(graph)
    }};
}

#[cfg(test)]
mod with_graph_tests {
    use super::*;

    #[test]
    fn test_with_graph() {
        // 1. 准备测试数据
        let registry = init_or_get_graph_registry();
        let graph_id = GraphId::new();
        {
            let mut map = registry.lock().unwrap();
            map.insert(graph_id, Graph::new());
        }

        // 2. 测试宏
        with_graph!(graph_id, |graph: &Graph| {
            assert!(graph.nodes().is_empty());
            assert!(graph.is_training());
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_registry_init_and_get() {
        // 1. 清理全局注册表
        {
            let registry = init_or_get_graph_registry();
            let mut map = registry.lock().unwrap();
            map.clear();
        }

        // 2. 验证插入新图
        let registry = init_or_get_graph_registry();
        let mut map = registry.lock().unwrap();
        let graph = Graph::new();
        let graph_id = graph.id();
        map.insert(graph_id, graph);
        assert!(map.contains_key(&graph_id));

        // 3. 从全局图中验证指定图的初始状态
        let graph = map.get_mut(&graph_id).unwrap();
        assert!(graph.nodes().is_empty());
        assert!(graph.is_training);

        // 4. 对该图做修改后再从全局图中检验修改是否成功
        let node_id = graph
            .new_variable(&[2, 2], true, true, Some("test"))
            .unwrap();
        let new_graph = map.get_mut(&graph_id).unwrap();
        assert!(new_graph.nodes().contains_key(&node_id));
    }

    #[test]
    fn test_graph_name_and_id_management() {
        // 1. 清理全局注册表
        {
            let registry = init_or_get_graph_registry();
            let mut map = registry.lock().unwrap();
            map.clear();
        }

        // 2. 测试创建默认名称的图
        let graph1 = Graph::new();
        assert_eq!(graph1.name(), "default_graph");

        // 3. 测试创建自定义名称的图
        let graph2 = Graph::with_name("custom_graph").unwrap();
        assert_eq!(graph2.name(), "custom_graph");

        // 4. 测试创建重复名称的图
        assert!(matches!(
            Graph::with_name("custom_graph"),
            Err(GraphError::DuplicateName(_))
        ));

        // 5. 测试创建另一个不同名称的图
        let graph3 = Graph::with_name("another_graph").unwrap();
        assert_eq!(graph3.name(), "another_graph");

        // 6. 测试ID唯一性
        let registry = init_or_get_graph_registry();
        let mut map = registry.lock().unwrap();
        map.clear();

        // 7. 创建第一个图并插入注册
        let graph1 = Graph::new();
        let id1 = graph1.id();
        map.insert(id1, graph1);

        // 8. 尝试创建使用相同ID的图（通过动设置ID来模拟）
        let graph2 = Graph {
            id: id1,
            name: "test".to_string(),
            nodes: HashMap::new(),
            edges: HashMap::new(),
            next_id: 0,
            is_training: true,
        };
        assert!(matches!(
            Graph::check_duplicate_id(graph2.id()),
            Err(GraphError::DuplicateId(_))
        ));
    }
}
