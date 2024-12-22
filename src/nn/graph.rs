/*
 * @Author       : 老董
 * @Date         : 2024-01-31 17:57:13
 * @LastEditors  : 老董
 * @LastEditTime : 2024-12-22 14:59:02
 * @Description  : 神经网络模型的计算图
 */

use super::nodes::NodeHandle;
use super::nodes::NodeType;
use super::NodeId;
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

/// 图的完整定义
pub struct Graph {
    name: String,
    nodes: HashMap<NodeId, NodeHandle>,
    edges: HashMap<NodeId, HashSet<NodeId>>,
    next_id: u64,
    is_eval_mode: bool,
}

impl Graph {
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

    fn generate_new_node_name(&self, base_name: &str, node_type: &str) -> String {
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

    // 基本操作
    pub fn new() -> Self {
        Self::with_name("default_graph").unwrap_or_else(|_| panic!("创建默认图失败"))
    }

    pub fn with_name(name: &str) -> Result<Self, GraphError> {
        Ok(Self {
            name: name.to_string(),
            nodes: HashMap::new(),
            edges: HashMap::new(),
            next_id: 0,
            is_eval_mode: false,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn nodes(&self) -> Vec<NodeId> {
        self.nodes.keys().cloned().collect()
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
        let parents_ids = self.get_node(node_id)?.parents_ids().to_vec();
        for parent_id in &parents_ids {
            self.forward_node(*parent_id)?;
        }

        // 3. 先收集所有父节点的值
        let mut parent_values = Vec::new();
        for id in &parents_ids {
            let value = self
                .get_node(*id)?
                .value()
                .ok_or_else(|| GraphError::ComputationError("父节点没有值".to_string()))?
                .clone();
            parent_values.push(value);
        }

        // 4. 创建临时的父节点句柄，不持有self的引用(避免等会计算值时借用检查问题)
        let parent_nodes = parents_ids
            .iter()
            .map(|id| self.get_node(*id).unwrap().clone())
            .collect::<Vec<NodeHandle>>();

        // 5. 计算当前节点
        let node = self.get_node_mut(node_id)?;
        node.calc_value_by_parents(&parent_nodes)?;

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
                let parent = self.get_node(node_id)?;

                // 根据节点类型决定是否需要另一个父节点
                let other_parent = match child.node_type() {
                    NodeType::MatMul(_) => {
                        // 找到另一个父节点
                        let other_parent_id = child
                            .parents_ids()
                            .iter()
                            .find(|&&id| id != node_id)
                            .ok_or_else(|| {
                                GraphError::ComputationError(
                                    "MatMul节点缺少另一个父节点".to_string(),
                                )
                            })?;
                        Some(self.get_node(*other_parent_id)?)
                    }
                    _ => None,
                };

                let local_jacobi = child.calc_jacobi_to_a_parent(parent, other_parent)?;
                let child_jacobi = child.jacobi().ok_or_else(|| {
                    GraphError::ComputationError("子节点没有雅可比矩阵".to_string())
                })?;
                child_jacobi * local_jacobi
            };

            // 3.2.3 更新当前节点的梯度（雅可比矩阵）
            {
                let node = self.get_node_mut(node_id)?;
                let current = node.jacobi().ok_or_else(|| {
                    GraphError::ComputationError("节点没有雅可比矩阵".to_string())
                })?;
                node.set_jacobi(Some(&(current + contribution)))?;
            }
        }

        // 4. 返回
        Ok(())
    }

    pub fn clear_jacobi(&mut self) -> Result<(), GraphError> {
        for node in self.nodes.values_mut() {
            node.clear_jacobi()?;
        }
        Ok(())
    }

    fn reset_from(&mut self, node_id: NodeId) -> Result<(), GraphError> {
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

    /// 根据ID获取节点的引用
    pub(in crate::nn) fn get_node(&self, id: NodeId) -> Result<&NodeHandle, GraphError> {
        self.nodes.get(&id).ok_or(GraphError::NodeNotFound(id))
    }

    /// 根据ID获取节点的可变引用
    pub(in crate::nn) fn get_node_mut(
        &mut self,
        id: NodeId,
    ) -> Result<&mut NodeHandle, GraphError> {
        self.nodes.get_mut(&id).ok_or(GraphError::NodeNotFound(id))
    }

    /// 根据ID获取节点的值，处理节点查找和值提取
    pub fn get_node_value(&self, id: NodeId) -> Result<&Tensor, GraphError> {
        self.get_node(id)?
            .value()
            .ok_or_else(|| GraphError::ComputationError(format!("节点 {} 没有值", id.0)))
    }

    /// 根据ID设置节点的值
    pub fn set_node_value(&mut self, id: NodeId, value: Option<&Tensor>) -> Result<(), GraphError> {
        self.get_node_mut(id)?.set_value(value)
    }

    /// 根据ID获取节点的雅可比矩阵
    pub fn get_node_jacobi(&self, id: NodeId) -> Result<&Tensor, GraphError> {
        self.get_node(id)?
            .jacobi()
            .ok_or_else(|| GraphError::ComputationError(format!("节点 {} 没有雅可比矩阵", id.0)))
    }
}

// 图模式相关
impl Graph {
    pub fn set_train_mode(&mut self) {
        self.is_eval_mode = false;
    }

    pub fn set_eval_mode(&mut self) {
        self.is_eval_mode = true;
    }

    pub fn is_train_mode(&self) -> bool {
        !self.is_eval_mode
    }
}

// 便捷的节点构建方法
impl Graph {
    fn add_node_to_list(
        &mut self,
        mut node_handle: NodeHandle,
        name: Option<&str>,
        node_type: &str,
    ) -> Result<NodeId, GraphError> {
        // 1. 生成节点ID和名称
        let node_id = self.generate_node_id();
        let node_name = self.generate_new_node_name(name.unwrap_or(""), node_type);

        // 2. 绑定ID和名称
        node_handle.bind_id_and_name(node_id, &node_name)?;

        // 3. 注册父-子关系
        for parent_id in node_handle.parents_ids() {
            self.edges.entry(*parent_id).or_default().insert(node_id);
        }

        // 4. 将节点句柄插入到节点列表中，并返回ID
        self.nodes.insert(node_id, node_handle);
        Ok(node_id)
    }

    pub fn new_variable_node(
        &mut self,
        shape: &[usize],
        init: bool,
        trainable: bool,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_variable(shape, init, trainable)?;
        self.add_node_to_list(handle, name, "variable")
    }

    pub fn new_add_node(
        &mut self,
        parents_ids: &[NodeId],
        name: Option<&str>,
        trainable: bool,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_add(parents_ids, trainable)?;
        self.add_node_to_list(handle, name, "add")
    }

    pub fn new_mat_mul_node(
        &mut self,
        left_node_id: NodeId,
        right_node_id: NodeId,
        name: Option<&str>,
        trainable: bool,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_mat_mul(&[left_node_id, right_node_id], trainable)?;
        self.add_node_to_list(handle, name, "mat_mul")
    }

    pub fn new_step_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
        trainable: bool,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_step(&[parent_id], trainable)?;
        self.add_node_to_list(handle, name, "step")
    }

    pub fn new_perception_loss_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
        trainable: bool,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_perception_loss(&[parent_id], trainable)?;
        self.add_node_to_list(handle, name, "perception_loss")
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
    DimensionMismatch {
        expected: usize,
        got: usize,
        message: String,
    },
    ComputationError(String),
    DuplicateName(String),
    DuplicateNodeName(String),
}
