/*
 * @Author       : 老董
 * @Date         : 2024-01-31 17:57:13
 * @LastEditors  : 老董
 * @LastEditTime : 2025-01-15 16:41:45
 * @Description  : 神经网络模型的计算图
 */

use super::NodeId;
use super::nodes::raw_node::Reduction;
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

    /// 反向传播：计算结果节点对本节点的雅可比矩阵（扩展版本）
    ///
    /// # 参数
    /// - `target_nodes_ids`: 目标节点列表（通常是需要计算梯度的参数节点）
    /// - `result_node_id`: 结果节点（通常是 loss 节点）
    /// - `retain_graph`: 是否保留计算图供后续反向传播使用
    ///   - `true`: 保留中间值，允许再次 backward
    ///   - `false`: 释放中间值以节省内存
    ///
    /// # 用途
    /// - **多 Loss 共享计算路径**：多任务学习中多个 loss 需要分别 backward
    /// - **高阶导数**：计算梯度的梯度需要保留计算图
    /// - **Actor-Critic 共享 backbone**：两个 loss 共享前向计算
    ///
    /// # 示例
    /// ```ignore
    /// // 多任务学习
    /// graph.forward_node(shared_output)?;
    /// let cls_loss = compute_cls_loss();
    /// let reg_loss = compute_reg_loss();
    ///
    /// // 第一个 loss backward，保留图
    /// graph.backward_nodes_ex(&[cls_weights], cls_loss, true)?;
    ///
    /// // 第二个 loss backward，可以释放图
    /// graph.backward_nodes_ex(&[reg_weights], reg_loss, false)?;
    /// ```
    pub fn backward_nodes_ex(
        &mut self,
        target_nodes_ids: &[NodeId],
        result_node_id: NodeId,
        retain_graph: bool,
    ) -> Result<(), GraphError> {
        // 0. 警告：在 no_grad（eval）模式下调用 backward 通常是误用
        // 虽然静态图架构允许此操作，但大多数情况下无实际意义
        // 参考设计文档: .doc/design/gradient_flow_control_design.md
        if !self.is_train_mode() {
            eprintln!(
                "[only_torch 警告] 在 no_grad/eval 模式下调用 backward，这通常是误用。\
                如确需此行为，请忽略此警告。"
            );
        }

        // 1. 重置中间节点的 jacobi（PyTorch 语义：只有参数节点梯度累积）
        // 这确保每次 backward 时中间节点的梯度是重新计算的
        self.reset_intermediate_jacobi();

        // 2. 为图本次的反向传播设置新id
        self.last_backward_pass_id += 1;
        let graph_backward_pass_id = self.last_backward_pass_id;

        // 3. 对每个目标节点执行反向传播
        for &target_id in target_nodes_ids {
            self.backward_node_internal(target_id, result_node_id)
                .inspect_err(|_| {
                    // 如果出错则回滚backward_pass_id
                    self.last_backward_pass_id = graph_backward_pass_id - 1;
                })?;
        }

        // 4. 根据 retain_graph 决定是否释放中间值
        if !retain_graph {
            self.release_intermediate_results()?;
        }

        Ok(())
    }

    /// 反向传播：计算结果节点对本节点的雅可比矩阵
    ///
    /// 这是 `backward_nodes_ex` 的简化版本，默认 `retain_graph = false`。
    /// 这意味着 backward 后中间节点的值会被释放以节省内存。
    /// 如需多次 backward，请使用 `backward_nodes_ex(..., retain_graph=true)`。
    ///
    /// NOTE: 这里的逻辑参考了 MatrixSlow/matrixslow/core/node.py#L83 (Node.backward)
    pub fn backward_nodes(
        &mut self,
        target_nodes_ids: &[NodeId],
        result_node_id: NodeId,
    ) -> Result<(), GraphError> {
        // 默认不保留图（与 PyTorch 行为一致）
        self.backward_nodes_ex(target_nodes_ids, result_node_id, false)
    }

    /// 释放中间节点的值和梯度以节省内存
    ///
    /// 保留 Input 和 Parameter 节点的数据，只清除计算节点（如 Add、MatMul 等）的值和梯度。
    /// 这些值在下次 forward 时会重新计算，梯度在下次 backward 时会重新计算。
    ///
    /// # 用途
    /// - 在 `backward_nodes_ex(..., retain_graph=false)` 后自动调用
    /// - 可手动调用以释放内存
    ///
    /// # 设计理由
    /// 当 `retain_graph=false` 时，同时释放值和梯度以保持一致性：
    /// - 值被释放：需要重新 forward 才能再次 backward
    /// - 梯度也被释放：避免用户误以为中间节点的梯度是累积的（实际只是本次的）
    ///
    /// 这更接近 PyTorch 的语义：中间节点的梯度默认不保留（除非 `retain_grad()`）
    fn release_intermediate_results(&mut self) -> Result<(), GraphError> {
        for node in self.nodes.values_mut() {
            match node.node_type() {
                // 保留输入和参数节点的值和梯度
                NodeType::Input(_) | NodeType::Parameter(_) => {}
                // 清除其他节点的值和梯度
                _ => {
                    node.clear_value()?;
                    let _ = node.clear_jacobi();
                }
            }
        }
        Ok(())
    }

    /// 重置中间节点的 jacobi（保留参数节点的 jacobi 以支持梯度累积）
    ///
    /// PyTorch 语义：
    /// - 参数节点（叶节点）：梯度跨 backward 调用累积
    /// - 中间节点：每次 backward 重新计算，不累积
    ///
    /// 这确保了多任务学习等场景下梯度计算的正确性。
    fn reset_intermediate_jacobi(&mut self) {
        for node in self.nodes.values_mut() {
            match node.node_type() {
                // 保留参数节点的 jacobi（支持梯度累积）
                NodeType::Parameter(_) => {}
                // 清除其他节点的 jacobi（中间节点每次 backward 重新计算）
                _ => {
                    let _ = node.clear_jacobi();
                    // 重置 backward pass id 以便重新计算
                    node.set_last_backward_pass_id(0);
                }
            }
        }
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

        // 0.1 检查节点是否被 detach
        // 被 detach 的节点视为叶子节点，不向父节点传播梯度
        if target_node.is_detached() {
            return Ok(());
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
        // 3.1 若目标节点没有子节点且不是结果节点
        // 说明该节点不在从 target 到 result 的路径上，跳过（不设置 jacobi）
        let children_ids = self.get_node_children(target_node_id)?;
        if children_ids.is_empty() {
            // 这种情况发生在多输出网络中：某些输出节点不在当前 backward 的路径上
            // 例如：features -> [out1, out2]，backward on out1 时 out2 会走到这里
            return Ok(());
        }

        // 3.2 初始化雅可比矩阵（如果还没有的话）
        // 注意：为了支持梯度累积，我们不会重置已存在的雅可比矩阵
        let result_node = self.get_node(result_node_id)?;
        // 记录 jacobi 是否是本次 backward 新初始化的（用于 detach 语义）
        let jacobi_was_none = {
            let target_node = self.get_node(target_node_id)?;
            let was_none = target_node.jacobi().is_none();
            if was_none {
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
            was_none
        };

        // 3.3 计算所有子节点的梯度（雅可比矩阵）对当前节点的贡献
        // 跟踪是否有任何子节点贡献了梯度，以及是否有子节点因 detach 被跳过
        let mut has_contribution = false;
        let mut skipped_due_to_detach = false;
        for child_id in children_ids {
            // 若子节点从未前向传播过（forward_pass_id = 0），则跳过
            // 注意：不再检查 forward_pass_id 是否等于图的当前 id，以支持多次 forward 后的 backward
            let child = self.get_node(child_id)?;
            if child.last_forward_pass_id() == 0 {
                continue;
            }
            // 3.3.1 先计算结果节点对子节点的梯度（雅可比矩阵）
            self.backward_node_internal(child_id, result_node_id)?;

            // 3.3.2 检查子节点是否有 jacobi（可能因为 detach 而没有）
            let child = self.get_node(child_id)?;
            let child_jacobi = match child.jacobi() {
                Some(j) => j,
                None => {
                    // 子节点被 detach 或没有 jacobi，记录并跳过
                    skipped_due_to_detach = true;
                    continue;
                }
            };

            // 标记有贡献
            has_contribution = true;

            // 3.3.3 计算子节点对当前节点的梯度（雅可比矩阵）贡献
            let contribution = {
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
                child_jacobi.mat_mul(&local_jacobi)
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

        // 3.4 如果没有任何子节点贡献梯度，且是因为 detach 导致的（不是因为 forward_pass_id 不匹配），
        // 且 jacobi 是本次新初始化的，则将 jacobi 设为 None，符合 PyTorch 语义
        // 注意：只有明确因 detach 阻断且 jacobi 是本次初始化的才清除
        if !has_contribution && skipped_due_to_detach && jacobi_was_none {
            self.get_node_mut(target_node_id)?.clear_jacobi()?;
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
        // 0. 警告：在 no_grad（eval）模式下调用 backward 通常是误用
        if !self.is_train_mode() {
            eprintln!(
                "[only_torch 警告] 在 no_grad/eval 模式下调用 backward_batch，这通常是误用。\
                如确需此行为，请忽略此警告。"
            );
        }

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
        // 检查当前节点是否被 detach
        // 被 detach 的节点不向父节点传播梯度
        {
            let node = self.get_node(node_id)?;
            if node.is_detached() {
                return Ok(());
            }
        }

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

    /// 检查是否启用梯度计算（等价于 `is_train_mode()`）
    ///
    /// 在训练模式下返回 `true`，在评估模式（no_grad 上下文）中返回 `false`。
    ///
    /// # 示例
    /// ```ignore
    /// assert!(graph.is_grad_enabled());  // 默认训练模式
    /// graph.no_grad_scope(|g| {
    ///     assert!(!g.is_grad_enabled());  // no_grad 上下文
    ///     Ok(())
    /// })?;
    /// ```
    pub const fn is_grad_enabled(&self) -> bool {
        self.is_train_mode()
    }

    // ========== detach 机制 ==========

    /// 将节点标记为 detached，阻止梯度回流到其父节点
    ///
    /// 被 detach 的节点在反向传播时会被视为叶子节点，梯度不会继续向上传播。
    /// 这在 GAN、Actor-Critic 等需要精细控制梯度流向的场景中非常有用。
    ///
    /// # 用途
    /// - **GAN 训练**：训练判别器时 detach 生成器输出，防止 D 的 loss 更新 G
    /// - **Actor-Critic**：Critic 的 value 估计传给 Actor 时需要 detach
    /// - **Target Network**：目标网络的输出需要 detach
    ///
    /// # 示例
    /// ```ignore
    /// // GAN 训练 - 训练判别器
    /// let fake = graph.forward_node(generator_output)?;
    /// graph.detach_node(fake)?;  // 防止 D 的 loss 更新 G
    /// let d_fake = graph.forward_node(discriminator_on_fake)?;
    /// graph.backward_nodes(&[d_weights], d_loss)?;
    ///
    /// // 训练生成器时恢复梯度流
    /// graph.attach_node(fake)?;
    /// graph.backward_nodes(&[g_weights], g_loss)?;
    /// ```
    pub fn detach_node(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        self.get_node_mut(node_id)?.set_detached(true);
        Ok(())
    }

    /// 取消节点的 detach 状态，恢复梯度流
    ///
    /// # 示例
    /// ```ignore
    /// graph.attach_node(fake)?;  // 恢复梯度流
    /// graph.backward_nodes(&[g_weights], g_loss)?;
    /// ```
    pub fn attach_node(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        self.get_node_mut(node_id)?.set_detached(false);
        Ok(())
    }

    /// 检查节点是否被 detach
    ///
    /// # 返回
    /// - `true`: 节点被 detach，反向传播时不向父节点传播梯度
    /// - `false`: 正常状态，梯度会向父节点传播
    pub fn is_node_detached(&self, node_id: NodeId) -> Result<bool, GraphError> {
        Ok(self.get_node(node_id)?.is_detached())
    }

    /// 在 no_grad 上下文中执行闭包
    ///
    /// 在此上下文中，图处于评估模式，前向传播不会为反向传播缓存中间值。
    /// 闭包执行完毕后，图会自动恢复到之前的模式。
    ///
    /// # 用途
    /// - **推理/评估**：模型评估时不需要计算梯度
    /// - **性能优化**：跳过梯度追踪相关的开销
    /// - **内存节省**：不存储用于反向传播的中间值
    ///
    /// # 参数
    /// - `f`: 在 no_grad 上下文中执行的闭包
    ///
    /// # 返回
    /// 闭包的返回值
    ///
    /// # 示例
    /// ```ignore
    /// // 训练阶段
    /// graph.set_train_mode();
    /// graph.forward_node(loss)?;
    /// graph.backward_nodes(&[w], loss)?;
    ///
    /// // 验证阶段（no_grad）
    /// let val_loss = graph.no_grad_scope(|g| {
    ///     g.set_node_value(x, Some(&val_data))?;
    ///     g.forward_node(output)?;
    ///     let loss_val = g.get_node_value(loss)?.unwrap().data()[0];
    ///     Ok(loss_val)
    /// })?;
    /// ```
    ///
    /// # 嵌套调用
    /// 支持嵌套调用，每次调用都会正确恢复到调用前的状态：
    /// ```ignore
    /// graph.set_train_mode();
    /// graph.no_grad_scope(|g| {
    ///     assert!(!g.is_grad_enabled());
    ///     g.no_grad_scope(|g2| {
    ///         assert!(!g2.is_grad_enabled());
    ///         Ok(())
    ///     })
    /// })?;
    /// assert!(graph.is_grad_enabled());  // 恢复到训练模式
    /// ```
    pub fn no_grad_scope<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        // 保存当前模式
        let was_train = self.is_train_mode();

        // 切换到评估模式（禁用梯度）
        self.set_eval_mode();

        // 执行闭包
        let result = f(self);

        // 恢复之前的模式
        if was_train {
            self.set_train_mode();
        }

        result
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

    /// 创建 MSELoss（均方误差损失）节点
    ///
    /// 使用默认的 Mean reduction 模式。
    ///
    /// # 参数
    /// - `input_id`: 预测值节点 ID
    /// - `target_id`: 目标值节点 ID
    /// - `name`: 可选的节点名称
    ///
    /// # 返回
    /// 新创建的损失节点 ID，输出为标量 [1, 1]
    ///
    /// # 公式
    /// `MSE = mean((input - target)^2)`
    pub fn new_mse_loss_node(
        &mut self,
        input_id: NodeId,
        target_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let parents = self.get_nodes(&[input_id, target_id])?;
        let handle = NodeHandle::new_mse_loss(&parents)?;
        self.add_node_to_list(handle, name, "mse_loss", &[input_id, target_id])
    }

    /// 创建 MSELoss 节点（指定 reduction 模式）
    ///
    /// # 参数
    /// - `input_id`: 预测值节点 ID
    /// - `target_id`: 目标值节点 ID
    /// - `reduction`: Reduction 模式（Mean 或 Sum）
    /// - `name`: 可选的节点名称
    ///
    /// # 返回
    /// 新创建的损失节点 ID，输出为标量 [1, 1]
    pub fn new_mse_loss_node_with_reduction(
        &mut self,
        input_id: NodeId,
        target_id: NodeId,
        reduction: Reduction,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let parents = self.get_nodes(&[input_id, target_id])?;
        let handle = NodeHandle::new_mse_loss_with_reduction(&parents, reduction)?;
        self.add_node_to_list(handle, name, "mse_loss", &[input_id, target_id])
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
