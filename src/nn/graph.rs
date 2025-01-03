/*
 * @Author       : 老董
 * @Date         : 2024-01-31 17:57:13
 * @LastEditors  : 老董
 * @LastEditTime : 2025-01-03 17:15:12
 * @Description  : 神经网络模型的计算图
 */

use super::nodes::{NodeHandle, NodeType};
use super::NodeId;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// 图的完整定义
pub struct Graph {
    name: String,
    nodes: HashMap<NodeId, NodeHandle>,
    /// 正向边：parent_id -> child_ids（父节点指向子节点）
    forward_edges: HashMap<NodeId, Vec<NodeId>>,
    /// 反向边：child_id -> parent_ids（子节点指向父节点）
    backward_edges: HashMap<NodeId, Vec<NodeId>>,
    /// 当前前向传播的次数
    forward_cnt: u64,
    next_id: u64,
    is_eval_mode: bool,
}

impl Graph {
    pub(in crate::nn) fn forward_cnt(&self) -> u64 {
        self.forward_cnt
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

    fn generate_valid_new_node_name(
        &self,
        base_name: &str,
        node_type: &str,
    ) -> Result<String, GraphError> {
        // 若用户提供了名称，检查重复并直接返回错误
        if !base_name.is_empty() {
            self.check_duplicate_node_name(base_name)?;
            return Ok(base_name.to_string());
        }

        // 自动生成名称（只有在用户未提供名称时才进行）
        let mut counter = 1;
        loop {
            let name = format!("{}_{}", node_type, counter);
            if self.check_duplicate_node_name(&name).is_ok() {
                return Ok(name);
            }
            counter += 1;
        }
    }

    // 基本操作
    pub fn new() -> Self {
        Self::with_name("default_graph")
    }

    pub fn with_name(name: &str) -> Self {
        Self {
            name: name.to_string(),
            nodes: HashMap::new(),
            forward_edges: HashMap::new(),
            backward_edges: HashMap::new(),
            forward_cnt: 0,
            next_id: 0,
            is_eval_mode: false,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn nodes(&self) -> Vec<NodeId> {
        self.nodes.keys().cloned().collect()
    }

    pub fn has_node_value(&self, node_id: NodeId) -> Result<bool, GraphError> {
        self.nodes
            .get(&node_id)
            .map(|node| node.has_value())
            .ok_or(GraphError::NodeNotFound(node_id))
    }

    // 前向传播：
    pub fn forward_node(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        // 1. 检查节点类型
        let node = self.get_node(node_id)?;
        if let NodeType::Variable(_) = node.node_type() {
            return Err(GraphError::InvalidOperation(
                "Variable节点是输入节点，其值应通过set_value设置，而不是通过父节点前向传播计算"
                    .to_string(),
            ));
        }

        // 2. 增加前向传播次数
        self.forward_cnt += 1;

        // 3. 通过内部方法执行完整的前向传播
        self.forward_node_internal(node_id).map_err(|e| {
            // 如果出错则回滚forward_cnt
            self.forward_cnt -= 1;
            e
        })
    }

    // 前向传播的内部实现
    pub(in crate::nn) fn forward_node_internal(
        &mut self,
        node_id: NodeId,
    ) -> Result<(), GraphError> {
        let graph_forward_cnt = self.forward_cnt.clone();

        // 1. 必要检查
        // 1.1 若节点已经在本代计算过，则直接返回
        let node = self.get_node(node_id)?;
        if node.forward_cnt() == graph_forward_cnt {
            return Ok(());
        }

        // 1.2 检查Variable节点是否在本代已经计算过
        if let NodeType::Variable(_) = node.node_type() {
            return Err(GraphError::InvalidOperation(format!(
                "Variable节点[{}]不能直接前向传播。问题节点的前向传播次数为{}，而图的前向传播次数为{}",
                node.id(),
                node.forward_cnt(),
                graph_forward_cnt
            )));
        }

        // 2. 递归计算所有父节点
        let parents_ids = self.get_node_parents(node_id)?.to_vec();
        for parent_id in &parents_ids {
            self.forward_node_internal(*parent_id)?;
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

        // 6. 更新节点的前向传播次数为当前次数
        node.set_forward_cnt(graph_forward_cnt);

        // 7. 返回
        Ok(())
    }

    /// 反向传播：计算结果节点对本节点的雅可比矩阵
    /// NOTE: 这里的逻辑参考了https://github.com/zc911/MatrixSlow/blob/a6db0d38802004449941e6644e609a2455b26327/matrixslow/core/node.py#L83
    pub fn backward_node(&mut self, node_id: NodeId, result_id: NodeId) -> Result<(), GraphError> {
        // 1. 检查节点是否可训练
        if !self.is_node_trainable(node_id)? {
            return Err(GraphError::InvalidOperation(
                "不能对不可训练的节点进行反向传播".to_string(),
            ));
        }

        // 2. 若已经计算过，则直接返回
        if self.get_node(node_id)?.jacobi().is_some() {
            return Ok(());
        }

        // 3. 若节点是结果节点（是自身），则自己对自己的雅可比为单位矩阵
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

        // 4. 其他情况的雅可比矩阵计算
        // 4.1 若节点没有子节点且不是结果节点，则返回错误
        let children = self.get_node_children(node_id)?;
        if children.is_empty() {
            return Err(GraphError::InvalidOperation(
                "无法对没有子节点的节点进行反向传播".to_string(),
            ));
        }

        // 4.2 先将雅可比矩阵初始化为零矩阵
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

        // 4.3 计算所有子节点的梯度（雅可比矩阵）对当前节点的贡献
        for child_id in children {
            // 4.3.1 先计算子节点对结果节点的梯度（雅可比矩阵）
            self.backward_node(child_id, result_id)?;

            // 4.3.2 计算子节点对当前节点的梯度（雅可比矩阵）贡献
            let contribution = {
                let child = self.get_node(child_id)?;
                let parent = self.get_node(node_id)?;

                // 根据节点类型决定是否需要另一个父节点
                let other_parent = match child.node_type() {
                    NodeType::MatMul(_) => {
                        // 找到另一个父节点
                        let parents = self.get_node_parents(child_id)?;
                        let other_parent_id =
                            parents.iter().find(|&&id| id != node_id).ok_or_else(|| {
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

            // 4.3.3 更新当前节点的梯度（雅可比矩阵）
            {
                let node = self.get_node_mut(node_id)?;
                let current = node
                    .jacobi()
                    .ok_or_else(|| GraphError::ComputationError("节点没有可比矩阵".to_string()))?;
                node.set_jacobi(Some(&(current + contribution)))?;
            }
        }

        // 5. 返回
        Ok(())
    }

    pub fn clear_jacobi(&mut self) -> Result<(), GraphError> {
        for node in self.nodes.values_mut() {
            node.clear_jacobi()?;
        }
        Ok(())
    }

    fn generate_valid_node_id(&mut self) -> NodeId {
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

    /// 根据ID获取节点的变引用
    fn get_node_mut(&mut self, id: NodeId) -> Result<&mut NodeHandle, GraphError> {
        self.nodes.get_mut(&id).ok_or(GraphError::NodeNotFound(id))
    }

    /// 获取多个节点的引用
    fn get_nodes<'a>(&'a self, ids: &[NodeId]) -> Result<Vec<&'a NodeHandle>, GraphError> {
        ids.iter().map(|id| self.get_node(*id)).collect()
    }

    /// 根据ID获取节点的值，处理节点查找和值提取
    pub fn get_node_value(&self, id: NodeId) -> Result<Option<&Tensor>, GraphError> {
        Ok(self.get_node(id)?.value())
    }

    /// 根据ID设置节点的值
    pub fn set_node_value(&mut self, id: NodeId, value: Option<&Tensor>) -> Result<(), GraphError> {
        let forward_cnt = self.forward_cnt.clone();
        // 1. 设置节点值
        let node = self.get_node_mut(id)?;
        node.set_value(value)?;

        // 2. 更新节点的前向传播次数为当前图的次数+1
        node.set_forward_cnt(forward_cnt + 1);

        // 3. 返回
        Ok(())
    }

    /// 根据ID获取节点的雅可比矩阵
    pub fn get_node_jacobi(&self, id: NodeId) -> Result<Option<&Tensor>, GraphError> {
        Ok(self.get_node(id)?.jacobi())
    }

    // Add new public methods for node information access
    pub fn get_node_name(&self, id: NodeId) -> Result<&str, GraphError> {
        self.get_node(id).map(|node| node.name())
    }

    pub fn get_node_parents(&self, id: NodeId) -> Result<Vec<NodeId>, GraphError> {
        self.get_node(id)?; // 确保节点存在
        Ok(self
            .backward_edges
            .get(&id)
            .map(|vec| vec.clone())
            .unwrap_or_default())
    }

    pub fn get_node_children(&self, id: NodeId) -> Result<Vec<NodeId>, GraphError> {
        self.get_node(id)?; // 确保节点存在
        Ok(self
            .forward_edges
            .get(&id)
            .map(|vec| vec.clone())
            .unwrap_or_default())
    }

    pub fn is_node_inited(&self, id: NodeId) -> Result<bool, GraphError> {
        self.get_node(id).map(|node| node.is_inited())
    }

    pub fn get_node_value_shape(&self, id: NodeId) -> Result<Option<&[usize]>, GraphError> {
        Ok(self.get_node(id)?.value().map(|v| v.shape()))
    }

    pub fn get_node_value_expected_shape(&self, id: NodeId) -> Result<&[usize], GraphError> {
        Ok(self.get_node(id)?.value_expected_shape())
    }

    pub fn get_node_value_size(&self, id: NodeId) -> Result<Option<usize>, GraphError> {
        Ok(self.get_node(id)?.value().map(|v| v.size()))
    }

    pub fn get_node_jacobi_shape(&self, id: NodeId) -> Result<Option<&[usize]>, GraphError> {
        Ok(self.get_node(id)?.jacobi().map(|j| j.shape()))
    }

    pub fn get_node_jacobi_size(&self, id: NodeId) -> Result<Option<usize>, GraphError> {
        Ok(self.get_node(id)?.jacobi().map(|j| j.size()))
    }

    pub fn is_node_trainable(&self, id: NodeId) -> Result<bool, GraphError> {
        self.get_node(id).map(|node| node.is_trainable())
    }

    pub fn set_node_trainable(&mut self, id: NodeId, trainable: bool) -> Result<(), GraphError> {
        self.get_node_mut(id)?.set_trainable(trainable)
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
        parents: &[NodeId],
    ) -> Result<NodeId, GraphError> {
        // 1. 生成节点ID和名称
        let node_id = self.generate_valid_node_id();
        let node_name = self.generate_valid_new_node_name(name.unwrap_or(""), node_type)?;

        // 2. 更新父子关系
        // 2.1 更新正向边：父节点 -> 子节点
        for &parent_id in parents {
            self.forward_edges
                .entry(parent_id)
                .or_insert_with(Vec::new)
                .push(node_id);
        }
        // 2.2 更新反向边：子节点 -> 父节点
        self.backward_edges
            .entry(node_id)
            .or_insert_with(Vec::new)
            .extend(parents);

        // 3. 绑定ID和名称
        node_handle.bind_id_and_name(node_id, &node_name)?;

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
        let mut node = NodeHandle::new_variable(shape, init, trainable)?;
        if init {
            node.set_forward_cnt(self.forward_cnt + 1);
        }
        self.add_node_to_list(node, name, "variable", &[])
    }

    pub fn new_add_node(
        &mut self,
        parents: &[NodeId],
        name: Option<&str>,
        trainable: bool,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_add(&self.get_nodes(parents)?, trainable)?;
        self.add_node_to_list(handle, name, "add", parents)
    }

    pub fn new_mat_mul_node(
        &mut self,
        left_node_id: NodeId,
        right_node_id: NodeId,
        name: Option<&str>,
        trainable: bool,
    ) -> Result<NodeId, GraphError> {
        let handle =
            NodeHandle::new_mat_mul(&self.get_nodes(&[left_node_id, right_node_id])?, trainable)?;
        self.add_node_to_list(handle, name, "mat_mul", &[left_node_id, right_node_id])
    }

    pub fn new_step_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
        trainable: bool,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_step(&self.get_nodes(&[parent_id])?, trainable)?;
        self.add_node_to_list(handle, name, "step", &[parent_id])
    }

    pub fn new_perception_loss_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
        trainable: bool,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_perception_loss(&self.get_nodes(&[parent_id])?, trainable)?;
        self.add_node_to_list(handle, name, "perception_loss", &[parent_id])
    }
}

/// 图错误类型
#[derive(Debug, PartialEq)]
pub enum GraphError {
    GraphNotFound(String),
    NodeNotFound(NodeId),
    InvalidOperation(String),
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
