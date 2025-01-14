/*
 * @Author       : 老董
 * @Date         : 2024-01-31 17:57:13
 * @LastEditors  : 老董
 * @LastEditTime : 2025-01-14 16:13:53
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
    /// 最后一次前向传播的id
    last_forward_pass_id: u64,
    next_id: u64,
    is_eval_mode: bool,
}

impl Graph {
    #[cfg(test)]
    pub(in crate::nn) fn last_forward_pass_id(&self) -> u64 {
        self.last_forward_pass_id
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
            last_forward_pass_id: 0,
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
        match node.node_type() {
            NodeType::Input(_) | NodeType::Parameter(_) => {
                return Err(GraphError::InvalidOperation(format!(
                    "{}是输入或参数节点，其值应通过set_value设置，而不是通过父节点前向传播计算",
                    node
                )));
            }
            _ => {}
        }

        // 2. 为图本次的前向传播设置新id
        self.last_forward_pass_id += 1;

        // 3. 通过内部方法执行完整的前向传播
        self.forward_node_internal(node_id).map_err(|e| {
            // 如果出错则回滚forward_pass_id
            self.last_forward_pass_id -= 1;
            e
        })
    }

    // 前向传播的内部实现
    pub(in crate::nn) fn forward_node_internal(
        &mut self,
        node_id: NodeId,
    ) -> Result<(), GraphError> {
        let graph_forward_pass_id = self.last_forward_pass_id.clone();

        // 1. 必要检查
        let node = self.get_node_mut(node_id)?;

        // 1.1 检查节点类型和状态
        match node.node_type() {
            // 1.1.1 输入和参数节点
            NodeType::Input(_) | NodeType::Parameter(_) => {
                if node.has_value() {
                    node.set_last_forward_pass_id(graph_forward_pass_id);
                    return Ok(());
                }
                return Err(GraphError::InvalidOperation(format!(
                    "{}不能直接前向传播（须通过set_value或初始化时设置`init`为true来增加前向传播次数）。问题节点的前向传播次数为{}，而图的前向传播次数为{}",
                    node,
                    node.last_forward_pass_id(),
                    graph_forward_pass_id
                )));
            }
            _ => {
                // 1.1.2 其他类型节点，若已在本代计算过则直接返回
                if node.last_forward_pass_id() == graph_forward_pass_id {
                    return Ok(());
                }
            }
        }

        // 2. 递归计算所有父节点
        let parents_ids = self.get_node_parents(node_id)?.to_vec();
        for parent_id in &parents_ids {
            self.forward_node_internal(*parent_id)?;
        }

        // 3. 创建临时的父节点句柄，不持有self的引用(避免等会计算值时借用检查问题)
        let parent_nodes = parents_ids
            .iter()
            .map(|id| self.get_node(*id).unwrap().clone())
            .collect::<Vec<NodeHandle>>();

        // 4. 计算当前节点
        let node = self.get_node_mut(node_id)?;
        node.calc_value_by_parents(&parent_nodes)?;

        // 5. 更新节点的前向传播次数为当前次数
        node.set_last_forward_pass_id(graph_forward_pass_id);

        // 6. 返回
        Ok(())
    }

    /// 反向传播：计算结果节点对本节点的雅可比矩阵
    /// NOTE: 这里的逻辑参考了https://github.com/zc911/MatrixSlow/blob/a6db0d38802004449941e6644e609a2455b26327/matrixslow/core/node.py#L83
    pub fn backward_node(&mut self, node_id: NodeId, result_id: NodeId) -> Result<(), GraphError> {
        let target_node = self.get_node(node_id)?;
        let result_node = self.get_node(result_id)?;
        // println!("反向传播：{} -> {}", target_node, result_node);
        // 1. 若已经计算过，则直接返回
        if target_node.jacobi().is_some() {
            return Ok(());
        }

        // 2. 若节点是结果节点（是自身），则自己对自己的雅可比矩阵为单位矩阵
        if node_id == result_id {
            let element_number = target_node
                .value()
                .ok_or_else(|| {
                    let node = self.get_node(node_id).unwrap();
                    GraphError::ComputationError(format!("反向传播：{}没有值", node))
                })?
                .size();
            let eye = Tensor::eyes(element_number);
            self.get_node_mut(node_id)?.set_jacobi(Some(&eye))?;
            return Ok(());
        }

        // 3. 其他情况的雅可比矩阵计算
        // 3.1 若节点没有子节点且不是结果节点，则返回错误
        let children_ids = self.get_node_children(node_id)?;
        if children_ids.is_empty() {
            return Err(GraphError::InvalidOperation(format!(
                "无法对没有子节点的{}进行反向传播",
                self.get_node(node_id).unwrap()
            )));
        }

        // 3.2 先将雅可比矩阵初始化为零矩阵
        {
            let (result_dim, node_dim) = {
                let node = self.get_node(node_id)?;
                (
                    result_node.value().map(|v| v.size()).ok_or_else(|| {
                        GraphError::ComputationError(format!("反向传播：结果{}没有值", result_node))
                    })?,
                    node.value().map(|v| v.size()).ok_or_else(|| {
                        GraphError::ComputationError(format!("反向传播：{}没有值", target_node))
                    })?,
                )
            };
            let zeros = Tensor::zeros(&[result_dim, node_dim]);
            self.get_node_mut(node_id)?.set_jacobi(Some(&zeros))?;
        }

        // 3.3 计算所有子节点的梯度（雅可比矩阵）对当前节点的贡献
        for child_id in children_ids {
            // 若子节点的前向传播id不等于图的前向传播id，
            // 说明该节点未前向传播或值已过时，则跳过该子节点
            let child = self.get_node(child_id)?;
            if child.last_forward_pass_id() != self.last_forward_pass_id {
                continue;
            }
            // 3.3.1 先计算结果节点对子节点的梯度（雅可比矩阵）
            self.backward_node(child_id, result_id)?;

            // 3.3.2 计算子节点对当前节点的梯度（雅可比矩阵）贡献
            let contribution = {
                let child = self.get_node(child_id)?;

                // 根据节点类型决定是否需要另一个父节点
                let assistant_parent = match child.node_type() {
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

                let parent = self.get_node(node_id).unwrap();
                let local_jacobi = child
                    .calc_jacobi_to_a_parent(parent, assistant_parent)
                    .unwrap();
                child.jacobi().unwrap().mat_mul(&local_jacobi)
            };
            // 3.3.3 更新当前节点的梯度（雅可比矩阵）
            {
                let current = {
                    let node = self.get_node(node_id)?;
                    node.jacobi()
                        .ok_or_else(|| {
                            GraphError::ComputationError(format!(
                                "反向传播：{}没有雅可比矩阵",
                                node
                            ))
                        })?
                        .clone()
                };
                let node = self.get_node_mut(node_id)?;
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
        self.get_node_mut(id)?.set_value(value)
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
        node_handle.bind_id_and_name(node_id, &node_name);

        // 4. 将节点句柄插入到节点列表中，并返回ID
        self.nodes.insert(node_id, node_handle);
        Ok(node_id)
    }

    pub fn new_input_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let node = NodeHandle::new_input(shape)?;
        self.add_node_to_list(node, name, "input", &[])
    }

    pub fn new_parameter_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let node = NodeHandle::new_parameter(shape)?;
        self.add_node_to_list(node, name, "parameter", &[])
    }

    pub fn new_add_node(
        &mut self,
        parents: &[NodeId],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_add(&self.get_nodes(parents)?)?;
        self.add_node_to_list(handle, name, "add", parents)
    }

    pub fn new_mat_mul_node(
        &mut self,
        left_node_id: NodeId,
        right_node_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_mat_mul(&self.get_nodes(&[left_node_id, right_node_id])?)?;
        self.add_node_to_list(handle, name, "mat_mul", &[left_node_id, right_node_id])
    }

    pub fn new_step_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_step(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "step", &[parent_id])
    }

    pub fn new_perception_loss_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_perception_loss(&self.get_nodes(&[parent_id])?)?;
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
