/*
 * @Author       : 老董
 * @Date         : 2024-01-31 17:57:13
 * @LastEditors  : 老董
 * @LastEditTime : 2025-01-15 16:41:45
 * @Description  : 神经网络模型的计算图
 */

use super::NodeId;
use super::nodes::{NodeHandle, NodeType};
use crate::tensor::Tensor;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;

/// 图的完整定义
pub struct Graph {
    name: String,
    nodes: HashMap<NodeId, NodeHandle>,
    /// `正向边：parent_id` -> `child_ids（父节点指向子节点`）
    forward_edges: HashMap<NodeId, Vec<NodeId>>,
    /// `反向边：child_id` -> `parent_ids（子节点指向父节点`）
    backward_edges: HashMap<NodeId, Vec<NodeId>>,
    /// 最后一次前向传播的id
    last_forward_pass_id: u64,
    /// 最后一次反向传播的id
    last_backward_pass_id: u64,
    next_id: u64,
    is_eval_mode: bool,
    /// 图级别的随机数生成器（用于参数初始化等）
    /// None 表示使用默认的 thread_rng（非确定性）
    rng: Option<StdRng>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    #[cfg(test)]
    pub(in crate::nn) fn last_forward_pass_id(&self) -> u64 {
        self.last_forward_pass_id
    }

    pub(in crate::nn) const fn last_backward_pass_id(&self) -> u64 {
        self.last_backward_pass_id
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
            let name = format!("{node_type}_{counter}");
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

    /// 创建一个带固定种子的计算图（确保可重复性）
    ///
    /// 使用此方法创建的图会有一个独立的随机数生成器，
    /// 所有通过 `new_parameter_node()` 创建的参数都会使用这个 RNG 初始化。
    ///
    /// # NEAT 友好性
    /// 每个 Graph 有独立的 RNG 状态，多个 Graph 可以并行进化互不干扰。
    ///
    /// # 示例
    /// ```ignore
    /// let graph1 = Graph::new_with_seed(42);
    /// let graph2 = Graph::new_with_seed(42);
    /// // graph1 和 graph2 的参数初始化结果相同
    /// ```
    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            name: "default_graph".to_string(),
            nodes: HashMap::new(),
            forward_edges: HashMap::new(),
            backward_edges: HashMap::new(),
            last_forward_pass_id: 0,
            last_backward_pass_id: 0,
            next_id: 0,
            is_eval_mode: false,
            rng: Some(StdRng::seed_from_u64(seed)),
        }
    }

    /// 创建一个带名称和固定种子的计算图
    pub fn with_name_and_seed(name: &str, seed: u64) -> Self {
        Self {
            name: name.to_string(),
            nodes: HashMap::new(),
            forward_edges: HashMap::new(),
            backward_edges: HashMap::new(),
            last_forward_pass_id: 0,
            last_backward_pass_id: 0,
            next_id: 0,
            is_eval_mode: false,
            rng: Some(StdRng::seed_from_u64(seed)),
        }
    }

    pub fn with_name(name: &str) -> Self {
        Self {
            name: name.to_string(),
            nodes: HashMap::new(),
            forward_edges: HashMap::new(),
            backward_edges: HashMap::new(),
            last_forward_pass_id: 0,
            last_backward_pass_id: 0,
            next_id: 0,
            is_eval_mode: false,
            rng: None,
        }
    }

    /// 设置/重置图的随机种子
    ///
    /// 调用此方法会重置 RNG 状态，后续的参数创建将从新种子开始。
    ///
    /// # 示例
    /// ```ignore
    /// let mut graph = Graph::new();
    /// graph.set_seed(42);
    /// // 现在 graph 使用固定种子
    /// ```
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = Some(StdRng::seed_from_u64(seed));
    }

    /// 检查图是否有固定种子
    pub fn has_seed(&self) -> bool {
        self.rng.is_some()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn nodes(&self) -> Vec<NodeId> {
        self.nodes.keys().copied().collect()
    }

    pub fn has_node_value(&self, node_id: NodeId) -> Result<bool, GraphError> {
        self.nodes
            .get(&node_id)
            .map(super::nodes::node_handle::NodeHandle::has_value)
            .ok_or(GraphError::NodeNotFound(node_id))
    }

    // 前向传播：
    pub fn forward_node(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        // 1. 检查节点类型
        let node = self.get_node(node_id)?;
        match node.node_type() {
            NodeType::Input(_) | NodeType::Parameter(_) => {
                return Err(GraphError::InvalidOperation(format!(
                    "{node}是输入或参数节点，其值应通过set_value设置，而不是通过父节点前向传播计算"
                )));
            }
            _ => {}
        }

        // 2. 为图本次的前向传播设置新id
        let new_graph_forward_pass_id = self.last_forward_pass_id + 1;

        // 3. 通过内部方法执行完整的前向传播
        self.forward_node_internal(node_id, new_graph_forward_pass_id)?;

        // 4. 只有成功后才更新图的前向传播ID
        self.last_forward_pass_id = new_graph_forward_pass_id;
        Ok(())
    }

    // 前向传播的内部实现
    fn forward_node_internal(
        &mut self,
        node_id: NodeId,
        new_graph_forward_pass_id: u64,
    ) -> Result<(), GraphError> {
        // 1. 必要检查
        let node = self.get_node_mut(node_id)?;

        // 1.1 检查节点类型和状态
        match node.node_type() {
            // 1.1.1 输入和参数节点
            NodeType::Input(_) | NodeType::Parameter(_) => {
                if node.has_value() {
                    node.set_last_forward_pass_id(new_graph_forward_pass_id);
                    return Ok(());
                }
                return Err(GraphError::InvalidOperation(format!(
                    "{}不能直接前向传播（须通过set_value或初始化时设置`init`为true来增加前向传播次数）。问题节点的前向传播次数为{}，而图的前向传播次数为{}",
                    node,
                    node.last_forward_pass_id(),
                    new_graph_forward_pass_id
                )));
            }
            _ => {
                // 1.1.2 其他类型节点，若已在本代计算过则直接返回
                if node.last_forward_pass_id() == new_graph_forward_pass_id {
                    return Ok(());
                }
            }
        }

        // 2. 递归计算所有父节点
        let parents_ids = self.get_node_parents(node_id)?;
        for parent_id in &parents_ids {
            self.forward_node_internal(*parent_id, new_graph_forward_pass_id)?;
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
        node.set_last_forward_pass_id(new_graph_forward_pass_id);

        // 6. 返回
        Ok(())
    }

    /// 反向传播：计算结果节点对本节点的雅可比矩阵
    /// NOTE: 这里的逻辑参考了 MatrixSlow/matrixslow/core/node.py#L83 (Node.backward)
    pub fn backward_nodes(
        &mut self,
        target_nodes_ids: &[NodeId],
        result_node_id: NodeId,
    ) -> Result<(), GraphError> {
        // 1. 为图本次的反向传播设置新id
        self.last_backward_pass_id += 1;
        let graph_backward_pass_id = self.last_backward_pass_id;

        // 2. 对每个目标节点执行反向传播
        for &target_id in target_nodes_ids {
            self.backward_node_internal(target_id, result_node_id)
                .inspect_err(|_| {
                    // 如果出错则回滚backward_pass_id
                    self.last_backward_pass_id = graph_backward_pass_id - 1;
                })?;
        }

        Ok(())
    }

    // 反向传播的内部实现
    fn backward_node_internal(
        &mut self,
        target_node_id: NodeId,
        result_node_id: NodeId,
    ) -> Result<(), GraphError> {
        // 0. 首先检查目标节点是否为输入节点
        let target_node = self.get_node(target_node_id)?;
        if let NodeType::Input(_) = target_node.node_type() {
            return Err(GraphError::InvalidOperation(format!(
                "输入{target_node}不应该有雅可比矩阵"
            )));
        }

        let graph_backward_pass_id = self.last_backward_pass_id;

        // 1. 若已经在本次反向传播中计算过，则直接返回
        if target_node.last_backward_pass_id() == graph_backward_pass_id {
            return Ok(());
        }

        // 2. 若目标节点是结果节点（是自身），则自己对自己的雅可比矩阵为单位矩阵
        if target_node_id == result_node_id {
            let element_number = target_node
                .value()
                .ok_or_else(|| {
                    let node = self.get_node(target_node_id).unwrap();
                    GraphError::ComputationError(format!("反向传播：{node}没有值"))
                })?
                .size();
            let eye = Tensor::eyes(element_number);
            self.get_node_mut(target_node_id)?.set_jacobi(Some(&eye))?;
            // 更新节点的反向传播次数
            self.get_node_mut(target_node_id)?
                .set_last_backward_pass_id(graph_backward_pass_id);
            return Ok(());
        }

        // 3. 其他情况的雅可比矩阵计算
        // 3.1 若目标节点没有子节点且不是结果节点，则返回错误
        let children_ids = self.get_node_children(target_node_id)?;
        if children_ids.is_empty() {
            return Err(GraphError::InvalidOperation(format!(
                "无法对没有子节点的{}进行反向传播",
                self.get_node(target_node_id).unwrap()
            )));
        }

        // 3.2 初始化雅可比矩阵（如果还没有的话）
        // 注意：为了支持梯度累积，我们不会重置已存在的雅可比矩阵
        let result_node = self.get_node(result_node_id)?;
        {
            let target_node = self.get_node(target_node_id)?;
            if target_node.jacobi().is_none() {
                let (result_dim, node_dim) = (
                    result_node
                        .value()
                        .map(super::super::tensor::Tensor::size)
                        .ok_or_else(|| {
                            GraphError::ComputationError(format!(
                                "反向传播：结果{result_node}没有值"
                            ))
                        })?,
                    target_node
                        .value()
                        .map(super::super::tensor::Tensor::size)
                        .ok_or_else(|| {
                            GraphError::ComputationError(format!("反向传播：{target_node}没有值"))
                        })?,
                );
                let zeros = Tensor::zeros(&[result_dim, node_dim]);
                self.get_node_mut(target_node_id)?
                    .set_jacobi(Some(&zeros))?;
            }
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
            self.backward_node_internal(child_id, result_node_id)?;

            // 3.3.2 计算子节点对当前节点的梯度（雅可比矩阵）贡献
            let contribution = {
                let child = self.get_node(child_id)?;

                // 根据节点类型决定是否需要另一个父节点
                let assistant_parent = match child.node_type() {
                    NodeType::MatMul(_)
                    | NodeType::Multiply(_)
                    | NodeType::ScalarMultiply(_)
                    | NodeType::SoftmaxCrossEntropy(_) => {
                        // 找到另一个父节点
                        let parents = self.get_node_parents(child_id)?;
                        let other_parent_id = parents
                            .iter()
                            .find(|&&id| id != target_node_id)
                            .ok_or_else(|| {
                                GraphError::ComputationError(
                                    "双父节点类型节点缺少另一个父节点".to_string(),
                                )
                            })?;
                        Some(self.get_node(*other_parent_id)?)
                    }
                    _ => None,
                };

                let parent = self.get_node(target_node_id).unwrap();
                let local_jacobi = child
                    .calc_jacobi_to_a_parent(parent, assistant_parent)
                    .unwrap();
                child.jacobi().unwrap().mat_mul(&local_jacobi)
            };
            // 3.3.3 更新当前节点的梯度（雅可比矩阵）
            {
                let current = {
                    let node = self.get_node(target_node_id)?;
                    node.jacobi()
                        .ok_or_else(|| {
                            GraphError::ComputationError(format!("反向传播：{node}没有雅可比矩阵"))
                        })?
                        .clone()
                };
                let node = self.get_node_mut(target_node_id)?;
                node.set_jacobi(Some(&(current + contribution)))?;
            }
        }

        // 4. 更新节点的反向传播次数
        self.get_node_mut(target_node_id)?
            .set_last_backward_pass_id(graph_backward_pass_id);

        // 5. 返回
        Ok(())
    }

    /// 清除所有节点的雅可比矩阵
    /// 通常在优化器的每个训练步骤开始时调用
    pub fn clear_jacobi(&mut self) -> Result<(), GraphError> {
        for node in self.nodes.values_mut() {
            node.clear_jacobi()?;
        }
        Ok(())
    }

    // ========== Batch 模式（Gradient-based）==========

    /// Batch 前向传播
    ///
    /// 与单样本 `forward_node` 类似，但输入节点的 value 应包含 batch 维度。
    /// 例如：输入 shape 为 `[batch_size, 784]` 而非 `[1, 784]`
    ///
    /// # 注意
    /// 当前实现复用 `forward_node` 的逻辑，因为大多数节点的 element-wise
    /// 操作天然支持 batch。未来可能需要针对特定节点优化。
    pub fn forward_batch(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        // 当前阶段：复用单样本前向传播逻辑
        // 大多数节点（Add、Sigmoid、Tanh 等）的 element-wise 操作天然支持 batch
        // MatMul 需要特殊处理（在节点层实现）
        self.forward_node(node_id)
    }

    /// Batch 反向传播
    ///
    /// 从损失节点开始，计算所有可训练参数的梯度。
    /// 与单样本模式的 `backward_nodes` 不同，此方法：
    /// 1. 使用 gradient-based 而非 Jacobi-based 反向传播
    /// 2. 自动对 batch 维度求平均
    /// 3. 梯度存储在节点的 `grad` 字段而非 `jacobi` 字段
    ///
    /// # 参数
    /// - `loss_id`: 损失节点 ID，应为标量 `[1, 1]`
    ///
    /// # 示例
    /// ```ignore
    /// // 设置 batch 输入
    /// graph.set_node_value(x, Some(&batch_images))?;  // [64, 784]
    /// graph.set_node_value(y, Some(&batch_labels))?;  // [64, 10]
    ///
    /// // Batch 前向传播
    /// graph.forward_batch(loss)?;
    ///
    /// // Batch 反向传播
    /// graph.backward_batch(loss)?;
    ///
    /// // 获取梯度
    /// let grad_w = graph.get_node_grad_batch(w)?;
    /// ```
    pub fn backward_batch(&mut self, loss_id: NodeId) -> Result<(), GraphError> {
        // 1. 验证损失节点
        let loss_node = self.get_node(loss_id)?;
        let loss_value = loss_node.value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "损失节点 {} 没有值，请先执行 forward_batch",
                loss_node
            ))
        })?;

        // 损失应为标量
        if loss_value.size() != 1 {
            return Err(GraphError::InvalidOperation(format!(
                "Batch 反向传播要求损失为标量 [1, 1]，但得到 {:?}",
                loss_value.shape()
            )));
        }

        // 2. 清除所有节点的梯度
        self.clear_grad()?;

        // 3. 损失节点的梯度为 1.0
        let loss_grad = Tensor::ones(&[1, 1]);
        self.get_node_mut(loss_id)?.set_grad(Some(&loss_grad))?;

        // 4. 获取拓扑排序（从损失到输入）
        let topo_order = self.topological_sort_backward(loss_id)?;

        // 5. 按拓扑顺序反向传播梯度
        for node_id in topo_order {
            self.backward_batch_node(node_id, loss_id)?;
        }

        Ok(())
    }

    /// 对单个节点执行 batch 反向传播
    fn backward_batch_node(&mut self, node_id: NodeId, _loss_id: NodeId) -> Result<(), GraphError> {
        // 获取当前节点的梯度
        let upstream_grad = {
            let node = self.get_node(node_id)?;
            match node.grad() {
                Some(g) => g.clone(),
                None => return Ok(()), // 没有梯度，跳过
            }
        };

        // 获取父节点列表
        let parent_ids = self.get_node_parents(node_id)?;
        if parent_ids.is_empty() {
            return Ok(()); // 输入/参数节点，无需继续传播
        }

        // 对每个父节点计算梯度
        for parent_id in &parent_ids {
            // 跳过 Input 节点（Input 不需要梯度）
            {
                let parent = self.get_node(*parent_id)?;
                if let NodeType::Input(_) = parent.node_type() {
                    continue;
                }
            }

            // 找到辅助父节点（如果需要）
            let assistant_parent_id = parent_ids.iter().find(|&&id| id != *parent_id).copied();

            // 计算对该父节点的梯度
            let parent_grad = {
                let node = self.get_node(node_id)?;
                let parent = self.get_node(*parent_id)?;
                let assistant = assistant_parent_id
                    .map(|id| self.get_node(id))
                    .transpose()?;

                node.calc_grad_to_parent(parent, &upstream_grad, assistant)?
            };

            // 累加到父节点的梯度
            let parent_node = self.get_node_mut(*parent_id)?;
            if let Some(existing_grad) = parent_node.grad() {
                let new_grad = existing_grad + &parent_grad;
                parent_node.set_grad(Some(&new_grad))?;
            } else {
                parent_node.set_grad(Some(&parent_grad))?;
            }
        }

        Ok(())
    }

    /// 获取从 loss 到所有输入的反向拓扑排序
    /// 返回的顺序：loss 在最前，input 在最后（适合反向传播）
    fn topological_sort_backward(&self, loss_id: NodeId) -> Result<Vec<NodeId>, GraphError> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();

        fn dfs(
            graph: &Graph,
            node_id: NodeId,
            visited: &mut std::collections::HashSet<NodeId>,
            result: &mut Vec<NodeId>,
        ) -> Result<(), GraphError> {
            if visited.contains(&node_id) {
                return Ok(());
            }
            visited.insert(node_id);

            // 先添加当前节点（因为我们要从 loss 向 input 方向传播）
            result.push(node_id);

            // 再访问父节点（反向传播方向）
            let parents = graph.get_node_parents(node_id)?;
            for parent_id in parents {
                dfs(graph, parent_id, visited, result)?;
            }

            Ok(())
        }

        // 从 loss 节点开始 DFS
        dfs(self, loss_id, &mut visited, &mut result)?;

        Ok(result)
    }

    /// 清除所有节点的梯度（Batch 模式）
    pub fn clear_grad(&mut self) -> Result<(), GraphError> {
        for node in self.nodes.values_mut() {
            let _ = node.clear_grad(); // 忽略不支持 grad 的节点
        }
        Ok(())
    }

    /// 获取节点的梯度（Batch 模式）
    pub fn get_node_grad_batch(&self, node_id: NodeId) -> Result<Option<&Tensor>, GraphError> {
        let node = self.get_node(node_id)?;

        // 输入节点不应该有梯度
        if let NodeType::Input(_) = node.node_type() {
            return Err(GraphError::InvalidOperation(format!(
                "输入节点 {} 不应该有梯度",
                node
            )));
        }

        Ok(node.grad())
    }

    /// 当图拓扑发生变化时调用（添加/删除节点或连接）
    /// 这会清除所有反向传播相关的状态（Jacobi），但保留前向传播的值
    ///
    /// # NEAT 友好性
    /// 这个方法是为神经进化算法（如 NEAT）设计的，在变异操作后调用
    /// 确保后续的前向/反向传播能正确工作
    ///
    /// # 示例
    /// ```ignore
    /// // 1. 初始图已经训练过
    /// graph.forward_node(loss)?;
    /// graph.backward_nodes(&[w], loss)?;
    ///
    /// // 2. 动态添加新节点（NEAT 变异）
    /// let new_node = graph.new_parameter_node(&[1, 1], Some("new"))?;
    /// let new_add = graph.new_add_node(&[old_node, new_node], None)?;
    ///
    /// // 3. 通知图拓扑已变化
    /// graph.on_topology_changed();
    ///
    /// // 4. 继续训练
    /// graph.forward_node(new_loss)?;
    /// graph.backward_nodes(&[w, new_node], new_loss)?;
    /// ```
    pub fn on_topology_changed(&mut self) {
        // 清除所有节点的 Jacobi（反向传播相关状态）
        // 保留 value（前向传播结果）以便复用
        for node in self.nodes.values_mut() {
            let _ = node.clear_jacobi();
            // 重置节点的反向传播 pass_id，确保下次 backward 会重新计算
            node.set_last_backward_pass_id(0);
        }
        // 注意：不重置 graph 的 last_backward_pass_id，
        // 因为新的 backward 调用会自增 pass_id，从而与所有节点的 0 不匹配，触发重新计算
    }

    const fn generate_valid_node_id(&mut self) -> NodeId {
        // 生成唯一的节点ID
        self.next_id += 1;
        NodeId(self.next_id)
    }

    // 用于调试
    pub fn nodes_count(&self) -> usize {
        self.nodes.len()
    }

    /// 获取所有可训练的参数节点ID
    pub fn get_trainable_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter_map(|(&id, node)| match node.node_type() {
                NodeType::Parameter(_) => Some(id),
                _ => None,
            })
            .collect()
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
    pub fn get_node_jacobi(&self, node_id: NodeId) -> Result<Option<&Tensor>, GraphError> {
        let node = self.get_node(node_id)?;

        // 1. 输入节点不应该有雅可比矩阵
        if let NodeType::Input(_) = node.node_type() {
            return Err(GraphError::InvalidOperation(format!(
                "输入{node}不应该有雅可比矩阵"
            )));
        }

        // 2. 返回节点的雅可比矩阵
        Ok(node.jacobi())
    }

    /// 获取节点的梯度。梯度是雅可比矩阵的转置，并reshape成与节点值相同的形状
    pub fn get_node_grad(&self, node_id: NodeId) -> Result<Option<Tensor>, GraphError> {
        // 1. 获取节点的雅可比矩阵
        let jacobi = match self.get_node_jacobi(node_id)? {
            Some(j) => j,
            None => return Ok(None),
        };

        // 2. 获取节点的预期形状
        let expected_shape = self.get_node_value_expected_shape(node_id)?;

        // 3. 转换雅可比矩阵为梯度
        Ok(Some(jacobi.transpose().reshape(expected_shape)))
    }

    // Add new public methods for node information access
    pub fn get_node_name(&self, id: NodeId) -> Result<&str, GraphError> {
        self.get_node(id)
            .map(super::nodes::node_handle::NodeHandle::name)
    }

    pub fn get_node_parents(&self, id: NodeId) -> Result<Vec<NodeId>, GraphError> {
        self.get_node(id)?; // 确保节点存在
        Ok(self.backward_edges.get(&id).cloned().unwrap_or_default())
    }

    pub fn get_node_children(&self, id: NodeId) -> Result<Vec<NodeId>, GraphError> {
        self.get_node(id)?; // 确保节点存在
        Ok(self.forward_edges.get(&id).cloned().unwrap_or_default())
    }

    pub fn is_node_inited(&self, id: NodeId) -> Result<bool, GraphError> {
        self.get_node(id)
            .map(super::nodes::node_handle::NodeHandle::is_inited)
    }

    pub fn get_node_value_shape(&self, id: NodeId) -> Result<Option<&[usize]>, GraphError> {
        Ok(self
            .get_node(id)?
            .value()
            .map(super::super::tensor::Tensor::shape))
    }

    pub fn get_node_value_expected_shape(&self, id: NodeId) -> Result<&[usize], GraphError> {
        Ok(self.get_node(id)?.value_expected_shape())
    }

    pub fn get_node_value_size(&self, id: NodeId) -> Result<Option<usize>, GraphError> {
        Ok(self
            .get_node(id)?
            .value()
            .map(super::super::tensor::Tensor::size))
    }

    pub fn get_node_jacobi_shape(&self, id: NodeId) -> Result<Option<&[usize]>, GraphError> {
        Ok(self
            .get_node(id)?
            .jacobi()
            .map(super::super::tensor::Tensor::shape))
    }

    pub fn get_node_jacobi_size(&self, id: NodeId) -> Result<Option<usize>, GraphError> {
        Ok(self
            .get_node(id)?
            .jacobi()
            .map(super::super::tensor::Tensor::size))
    }
}

// 图模式相关
impl Graph {
    pub const fn set_train_mode(&mut self) {
        self.is_eval_mode = false;
    }

    pub const fn set_eval_mode(&mut self) {
        self.is_eval_mode = true;
    }

    pub const fn is_train_mode(&self) -> bool {
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
                .or_default()
                .push(node_id);
        }
        // 2.2 更新反向边：子节点 -> 父节点
        self.backward_edges
            .entry(node_id)
            .or_default()
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

    /// 创建参数节点
    ///
    /// 如果 Graph 有种子（通过 `new_with_seed` 或 `set_seed` 设置），
    /// 则使用 Graph 的 RNG 进行参数初始化（确定性）。
    /// 否则使用默认的随机初始化（非确定性）。
    pub fn new_parameter_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        // 如果 Graph 有 RNG，从中生成种子用于参数初始化
        let node = if let Some(ref mut rng) = self.rng {
            use rand::Rng;
            let seed: u64 = rng.r#gen();
            NodeHandle::new_parameter_seeded(shape, seed)?
        } else {
            NodeHandle::new_parameter(shape)?
        };
        self.add_node_to_list(node, name, "parameter", &[])
    }

    /// 使用固定种子创建参数节点（确保可重复性）
    ///
    /// 注意：此方法会覆盖 Graph 的 RNG 设置，直接使用指定的种子。
    /// 适用于需要精确控制单个参数初始化的场景（如单元测试）。
    pub fn new_parameter_node_seeded(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
        seed: u64,
    ) -> Result<NodeId, GraphError> {
        let node = NodeHandle::new_parameter_seeded(shape, seed)?;
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

    /// 创建 Conv2d（2D 卷积）节点
    ///
    /// # 设计
    /// - **PyTorch 风格**：单节点处理多通道，而非 MatrixSlow 的每通道独立节点
    /// - 支持 Jacobi 模式（单样本）和 Batch 模式
    ///
    /// # 参数
    /// - `input_id`: 输入节点 ID，形状 `[C_in, H, W]` 或 `[batch, C_in, H, W]`
    /// - `kernel_id`: 卷积核参数节点 ID，形状 `[C_out, C_in, kH, kW]`
    /// - `stride`: 步长 `(sH, sW)`
    /// - `padding`: 零填充 `(pH, pW)`
    /// - `name`: 可选的节点名称
    ///
    /// # 输出形状
    /// - 单样本: `[C_out, H', W']`
    /// - Batch: `[batch, C_out, H', W']`
    /// - 其中 `H' = (H + 2*pH - kH) / sH + 1`
    ///
    /// # 示例
    /// ```ignore
    /// // 创建卷积核参数: 32 个 3x3 卷积核，输入 1 通道
    /// let kernel = graph.new_parameter_node(&[32, 1, 3, 3], Some("conv1_kernel"))?;
    ///
    /// // 输入: [batch, 1, 28, 28]（如 MNIST 图像）
    /// let input = graph.new_input_node(&[batch_size, 1, 28, 28], Some("input"))?;
    ///
    /// // 创建卷积层: stride=1, padding=1（保持尺寸）
    /// let conv_out = graph.new_conv2d_node(input, kernel, (1, 1), (1, 1), Some("conv1"))?;
    /// // 输出形状: [batch, 32, 28, 28]
    /// ```
    pub fn new_conv2d_node(
        &mut self,
        input_id: NodeId,
        kernel_id: NodeId,
        stride: (usize, usize),
        padding: (usize, usize),
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle =
            NodeHandle::new_conv2d(&self.get_nodes(&[input_id, kernel_id])?, stride, padding)?;
        self.add_node_to_list(handle, name, "conv2d", &[input_id, kernel_id])
    }

    /// 创建 MaxPool2d（2D 最大池化）节点
    ///
    /// # 设计
    /// - 在每个池化窗口中取最大值
    /// - 记录最大值位置用于反向传播（稀疏梯度）
    ///
    /// # 参数
    /// - `input_id`: 输入节点 ID，形状 `[C, H, W]` 或 `[batch, C, H, W]`
    /// - `kernel_size`: 池化窗口大小 `(kH, kW)`
    /// - `stride`: 步长 `(sH, sW)`，`None` 时默认等于 `kernel_size`
    /// - `name`: 可选的节点名称
    ///
    /// # 输出形状
    /// - 单样本: `[C, H', W']`
    /// - Batch: `[batch, C, H', W']`
    /// - 其中 `H' = (H - kH) / sH + 1`
    ///
    /// # 示例
    /// ```ignore
    /// // 输入: [batch, 32, 28, 28]
    /// let pool = graph.new_max_pool2d_node(conv_out, (2, 2), None, Some("pool1"))?;
    /// // 输出: [batch, 32, 14, 14]（默认 stride = kernel_size）
    ///
    /// // 自定义 stride
    /// let pool2 = graph.new_max_pool2d_node(input, (3, 3), Some((2, 2)), Some("pool2"))?;
    /// ```
    pub fn new_max_pool2d_node(
        &mut self,
        input_id: NodeId,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle =
            NodeHandle::new_max_pool2d(&self.get_nodes(&[input_id])?, kernel_size, stride)?;
        self.add_node_to_list(handle, name, "max_pool2d", &[input_id])
    }

    /// 创建 AvgPool2d（2D 平均池化）节点
    ///
    /// # 设计
    /// - 计算每个池化窗口内所有值的平均
    /// - 反向传播时梯度均匀分配到窗口内所有位置
    ///
    /// # 参数
    /// - `input_id`: 输入节点 ID，形状 `[C, H, W]` 或 `[batch, C, H, W]`
    /// - `kernel_size`: 池化窗口大小 `(kH, kW)`
    /// - `stride`: 步长 `(sH, sW)`，`None` 时默认等于 `kernel_size`
    /// - `name`: 可选的节点名称
    ///
    /// # 输出形状
    /// - 单样本: `[C, H', W']`
    /// - Batch: `[batch, C, H', W']`
    /// - 其中 `H' = (H - kH) / sH + 1`
    ///
    /// # 示例
    /// ```ignore
    /// // 输入: [batch, 32, 28, 28]
    /// let pool = graph.new_avg_pool2d_node(conv_out, (2, 2), None, Some("pool1"))?;
    /// // 输出: [batch, 32, 14, 14]（默认 stride = kernel_size）
    /// ```
    pub fn new_avg_pool2d_node(
        &mut self,
        input_id: NodeId,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle =
            NodeHandle::new_avg_pool2d(&self.get_nodes(&[input_id])?, kernel_size, stride)?;
        self.add_node_to_list(handle, name, "avg_pool2d", &[input_id])
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

    /// 创建逐元素乘法节点
    /// 两个父节点必须形状相同
    pub fn new_multiply_node(
        &mut self,
        left_node_id: NodeId,
        right_node_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_multiply(&self.get_nodes(&[left_node_id, right_node_id])?)?;
        self.add_node_to_list(handle, name, "multiply", &[left_node_id, right_node_id])
    }

    /// 创建 Flatten 节点
    ///
    /// 将张量展平为 2D，常用于 CNN 与全连接层之间的转换。
    ///
    /// # 参数
    /// - `parent_id`: 父节点 ID
    /// - `keep_first_dim`: 是否保留首维度
    ///   - `true`: 保留首维度（batch），展平其余维度
    ///   - `false`: 完全展平为行向量 `[1, n]`
    /// - `name`: 可选的节点名称
    ///
    /// # 示例
    /// ```ignore
    /// // CNN 输出 [batch, features] 展平（2D 保持不变）
    /// let flat = graph.new_flatten_node(conv_out, true, Some("flatten"))?;
    ///
    /// // 完全展平为行向量
    /// let row_vec = graph.new_flatten_node(input, false, Some("row_vec"))?;
    /// ```
    pub fn new_flatten_node(
        &mut self,
        parent_id: NodeId,
        keep_first_dim: bool,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_flatten(&self.get_nodes(&[parent_id])?, keep_first_dim)?;
        self.add_node_to_list(handle, name, "flatten", &[parent_id])
    }

    /// 创建 Reshape 节点
    ///
    /// 改变张量形状而不改变数据，常用于 CNN 与全连接层之间的转换。
    ///
    /// # 参数
    /// - `parent_id`: 父节点 ID
    /// - `target_shape`: 目标形状（元素总数必须与输入相同）
    /// - `name`: 可选的节点名称
    ///
    /// # 示例
    /// ```ignore
    /// // 将 [batch, 32, 7, 7] reshape 为 [batch, 1568]
    /// let flat = graph.new_reshape_node(conv_out, &[batch_size, 1568], Some("flatten"))?;
    /// ```
    pub fn new_reshape_node(
        &mut self,
        parent_id: NodeId,
        target_shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_reshape(&self.get_nodes(&[parent_id])?, target_shape)?;
        self.add_node_to_list(handle, name, "reshape", &[parent_id])
    }

    /// 创建标量乘法节点
    /// scalar_node_id: 标量节点(1x1)的ID
    /// matrix_node_id: 矩阵节点的ID
    pub fn new_scalar_multiply_node(
        &mut self,
        scalar_node_id: NodeId,
        matrix_node_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle =
            NodeHandle::new_scalar_multiply(&self.get_nodes(&[scalar_node_id, matrix_node_id])?)?;
        self.add_node_to_list(
            handle,
            name,
            "scalar_multiply",
            &[scalar_node_id, matrix_node_id],
        )
    }

    pub fn new_step_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_step(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "step", &[parent_id])
    }

    pub fn new_tanh_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_tanh(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "tanh", &[parent_id])
    }

    pub fn new_sigmoid_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_sigmoid(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "sigmoid", &[parent_id])
    }

    /// 创建 SoftPlus 激活节点
    ///
    /// SoftPlus 是 ReLU 的平滑近似: f(x) = ln(1 + e^x)
    /// 导数为 sigmoid: f'(x) = 1 / (1 + e^(-x))
    ///
    /// 适用场景：
    /// - 需要正值输出（如方差/标准差预测）
    /// - 需要平滑梯度的优化
    /// - 概率模型（VAE）、连续动作空间 RL（SAC/PPO）
    pub fn new_softplus_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_softplus(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "softplus", &[parent_id])
    }

    /// 创建 LeakyReLU 激活节点
    ///
    /// # 参数
    /// - `parent_id`: 父节点 ID
    /// - `negative_slope`: 负半轴斜率（0.0 时等价于标准 ReLU，MatrixSlow 使用 0.1）
    /// - `name`: 节点名称（可选）
    pub fn new_leaky_relu_node(
        &mut self,
        parent_id: NodeId,
        negative_slope: f64,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_leaky_relu(&self.get_nodes(&[parent_id])?, negative_slope)?;
        self.add_node_to_list(handle, name, "leaky_relu", &[parent_id])
    }

    /// 创建标准 ReLU 激活节点（等价于 negative_slope=0 的 LeakyReLU）
    ///
    /// # 参数
    /// - `parent_id`: 父节点 ID
    /// - `name`: 节点名称（可选）
    pub fn new_relu_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        self.new_leaky_relu_node(parent_id, 0.0, name)
    }

    pub fn new_perception_loss_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_perception_loss(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "perception_loss", &[parent_id])
    }

    /// 创建 SoftmaxCrossEntropy 损失节点
    ///
    /// # 参数
    /// - `logits_id`: 预测值节点 ID（未经 softmax 的原始分数）
    /// - `labels_id`: 标签节点 ID（one-hot 编码）
    /// - `name`: 可选的节点名称
    ///
    /// # 返回
    /// 新创建的损失节点 ID，输出为标量 [1, 1]
    pub fn new_softmax_cross_entropy_node(
        &mut self,
        logits_id: NodeId,
        labels_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let parents = self.get_nodes(&[logits_id, labels_id])?;
        let handle = NodeHandle::new_softmax_cross_entropy(&parents)?;
        self.add_node_to_list(handle, name, "softmax_ce", &[logits_id, labels_id])
    }
}

/// 图错误类型
#[derive(Debug, PartialEq, Eq)]
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
