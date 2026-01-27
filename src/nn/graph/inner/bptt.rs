/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : GraphInner BPTT（通过时间反向传播）
 */

use super::super::error::GraphError;
use super::GraphInner;
use crate::nn::nodes::NodeType;
use crate::nn::NodeId;
use crate::tensor::Tensor;

impl GraphInner {
    // ========== BPTT API（Phase 2） ==========

    /// 获取当前存储的时间步历史长度
    pub fn history_len(&self) -> usize {
        self.step_history.len()
    }

    /// 获取时间步历史长度（别名，兼容旧 API）
    pub fn step_history_len(&self) -> usize {
        self.step_history.len()
    }

    /// 清除 BPTT 历史（不重置循环状态）
    ///
    /// 与 `reset()` 不同，此方法只清除历史记录，不影响当前的循环状态和时间步。
    /// 用于 TBPTT 截断后开始新的截断窗口。
    pub fn clear_history(&mut self) {
        self.step_history.clear();
    }

    /// 通过时间反向传播（BPTT）
    ///
    /// 遍历所有存储的时间步，从最后一步向前反向传播，累加梯度到参数节点。
    ///
    /// # 参数
    /// - `target_nodes`: 需要计算梯度的目标节点（通常是参数节点）
    /// - `loss_node`: 损失节点（每个时间步都会从这个节点开始反向传播）
    ///
    /// # 工作原理
    /// ```text
    /// 对于序列 [t=0, t=1, t=2]：
    ///   1. 恢复 t=2 的快照，backward(loss) → 梯度累加到参数
    ///   2. 恢复 t=1 的快照，backward(loss) → 梯度继续累加
    ///   3. 恢复 t=0 的快照，backward(loss) → 梯度继续累加
    /// 最终参数的梯度 = Σ(各时间步的梯度贡献)
    /// ```
    ///
    /// # 示例
    /// ```ignore
    /// // 前向传播整个序列
    /// for input in sequence {
    ///     graph.set_node_value(x, Some(&input))?;
    ///     graph.step(output)?;
    /// }
    ///
    /// // 反向传播整个序列
    /// graph.backward_through_time(&[w, b], loss)?;
    ///
    /// // 更新参数
    /// optimizer.step(&mut graph)?;
    /// graph.zero_grad();
    /// graph.reset();
    /// ```
    pub fn backward_through_time(
        &mut self,
        target_nodes: &[NodeId],
        loss_node: NodeId,
    ) -> Result<(), GraphError> {
        self.backward_through_time_truncated(target_nodes, loss_node, None)
    }

    /// 截断的通过时间反向传播（TBPTT）
    ///
    /// 与 `backward_through_time` 相同，但只反向传播最近的 `truncation_steps` 个时间步。
    /// 用于处理长序列时限制内存使用和梯度消失/爆炸问题。
    ///
    /// # 参数
    /// - `target_nodes`: 需要计算梯度的目标节点
    /// - `loss_node`: 损失节点
    /// - `truncation_steps`: 截断长度，None 表示不截断（等同于 `backward_through_time`）
    ///
    /// # TBPTT 策略
    /// ```text
    /// 序列长度 = 10，truncation = 3
    ///
    /// 方式 1（本实现）：只反向传播最近 3 步
    ///   [t=7, t=8, t=9] → backward
    ///
    /// 方式 2（高级）：分段处理（需要用户自己实现）
    ///   [t=0,1,2] → backward → step
    ///   [t=3,4,5] → backward → step
    ///   [t=6,7,8,9] → backward → step
    /// ```
    pub fn backward_through_time_truncated(
        &mut self,
        target_nodes: &[NodeId],
        loss_node: NodeId,
        truncation_steps: Option<usize>,
    ) -> Result<(), GraphError> {
        if self.step_history.is_empty() {
            return Err(GraphError::InvalidOperation(
                "BPTT 失败：没有时间步历史。请确保在训练模式下调用 step()。".to_string(),
            ));
        }

        // 确定要反向传播的时间步范围
        let total_steps = self.step_history.len();
        let steps_to_backprop = truncation_steps.unwrap_or(total_steps).min(total_steps);
        let start_idx = total_steps - steps_to_backprop;

        // 收集循环边信息: to_node (State, 如 h_prev) -> from_node (如 hidden)
        let recurrent_edges_vec: Vec<(NodeId, NodeId)> = self
            .recurrent_edges
            .iter()
            .map(|(&to, &from)| (to, from))
            .collect();

        // 收集所有 State 节点（循环边的目标节点）
        let state_nodes: Vec<NodeId> = recurrent_edges_vec.iter().map(|(to, _)| *to).collect();

        // 合并目标节点：params + State 节点
        let mut all_targets: Vec<NodeId> = target_nodes.to_vec();
        for state_id in &state_nodes {
            if !all_targets.contains(state_id) {
                all_targets.push(*state_id);
            }
        }

        // 存储来自"未来"时间步的梯度
        // key: source_node (如 hidden), value: dL/d(source[t]) 从 t+1 传来
        let mut incoming_grads: std::collections::HashMap<NodeId, Tensor> =
            std::collections::HashMap::new();

        // 从最后一个时间步向前反向传播
        let is_first_step = |t: usize| t == total_steps - 1;

        #[cfg(test)]
        let debug = self.bptt_debug;
        #[cfg(not(test))]
        let debug = false;

        // 清除参数的 grad（确保干净的累加起点）
        for &param in target_nodes {
            self.get_node_mut(param)?.clear_grad()?;
        }

        for t in (start_idx..total_steps).rev() {
            // 恢复该时间步的快照
            let snapshot = self.step_history[t].clone();
            self.restore_snapshot(&snapshot);

            if debug {
                println!("\n=== BPTT t={t} ===");
                // 打印当前 incoming_grads
                for (node_id, grad) in &incoming_grads {
                    let name = self
                        .get_node(*node_id)
                        .map(|n| n.name().to_string())
                        .unwrap_or_default();
                    println!(
                        "  incoming_grads[{}({})]: {:?}",
                        name,
                        node_id.0,
                        grad.data_as_slice()
                    );
                }
            }

            // 收集传递到上一时间步的梯度（VJP 模式）
            let mut next_incoming_grads: std::collections::HashMap<NodeId, Tensor> =
                std::collections::HashMap::new();

            if is_first_step(t) {
                // === 最后一个时间步：从 loss 反向传播（纯 VJP）===
                // 使用 backward_from_loss_vjp 进行反向传播
                // 这会：1) 累加参数的 grad，2) 返回 State 节点收到的 grad
                let state_grads =
                    self.backward_from_loss_vjp(target_nodes, &state_nodes, loss_node)?;

                if debug {
                    println!("  [t={t}] After backward_from_loss_vjp:");
                    for &param in target_nodes {
                        let name = self
                            .get_node(param)
                            .map(|n| n.name().to_string())
                            .unwrap_or_default();
                        let grad = self.get_node_grad_ref(param).ok().flatten();
                        println!(
                            "    {} grad: {:?}",
                            name,
                            grad.map(|g| g.data_as_slice().to_vec())
                        );
                    }
                }

                // 收集 State grad 用于跨时间传递
                for &(to_node, from_node) in &recurrent_edges_vec {
                    if let Some(state_grad) = state_grads.get(&to_node) {
                        if debug {
                            let to_name = self
                                .get_node(to_node)
                                .map(|n| n.name().to_string())
                                .unwrap_or_default();
                            let from_name = self
                                .get_node(from_node)
                                .map(|n| n.name().to_string())
                                .unwrap_or_default();
                            println!(
                                "  [t={}] State grad {}({}) -> {}({}): {:?}",
                                t,
                                to_name,
                                to_node.0,
                                from_name,
                                from_node.0,
                                state_grad.data_as_slice()
                            );
                        }
                        next_incoming_grads.insert(from_node, state_grad.clone());
                    }
                }
            } else {
                // === 中间时间步：只从 incoming_grad 传播（纯 VJP）===
                if !incoming_grads.is_empty() {
                    for &(to_node, from_node) in &recurrent_edges_vec {
                        if let Some(incoming_grad) = incoming_grads.get(&from_node) {
                            if debug {
                                let from_name = self
                                    .get_node(from_node)
                                    .map(|n| n.name().to_string())
                                    .unwrap_or_default();
                                println!(
                                    "  [t={}] Processing from {} with incoming {:?}",
                                    t,
                                    from_name,
                                    incoming_grad.data_as_slice()
                                );
                            }

                            // 1) 传播参数梯度
                            self.bptt_backward_from_node_vjp(
                                from_node,
                                incoming_grad,
                                target_nodes,
                            )?;

                            if debug {
                                println!("  [t={t}] After param grad propagation:");
                                for &param in target_nodes {
                                    let name = self
                                        .get_node(param)
                                        .map(|n| n.name().to_string())
                                        .unwrap_or_default();
                                    let grad = self.get_node_grad_ref(param).ok().flatten();
                                    println!(
                                        "    {} grad: {:?}",
                                        name,
                                        grad.map(|g| g.data_as_slice().to_vec())
                                    );
                                }
                            }

                            // 2) 传播 State 梯度（用于跨时间传递）
                            let state_grads = self.bptt_propagate_to_state_vjp(
                                from_node,
                                incoming_grad,
                                target_nodes,
                                false, // 参数梯度已由上面处理，不要重复累加
                            )?;

                            if let Some(state_grad) = state_grads.get(&to_node) {
                                if debug {
                                    let to_name = self
                                        .get_node(to_node)
                                        .map(|n| n.name().to_string())
                                        .unwrap_or_default();
                                    println!(
                                        "  [t={}] State {} received grad: {:?}",
                                        t,
                                        to_name,
                                        state_grad.data_as_slice()
                                    );
                                }
                                next_incoming_grads.insert(from_node, state_grad.clone());
                            }
                        }
                    }
                }
            }

            if debug {
                println!("  [t={t}] next_incoming_grads:");
                for (node_id, grad) in &next_incoming_grads {
                    let name = self
                        .get_node(*node_id)
                        .map(|n| n.name().to_string())
                        .unwrap_or_default();
                    println!("    {}({}): {:?}", name, node_id.0, grad.data_as_slice());
                }
            }

            incoming_grads = next_incoming_grads;
        }

        Ok(())
    }

    /// BPTT 辅助方法：从源节点传播梯度到 State 节点（VJP/grad 模式）
    ///
    /// 使用 VJP (Vector-Jacobian Product) 而非完整 Jacobian 矩阵，
    /// 避免 O(N²) 内存开销，支持大 batch/hidden 尺寸的 RNN 训练。
    ///
    /// # 与 Jacobian 版本 (`bptt_propagate_to_state`) 的区别
    /// - Jacobian 模式：构造 N×N 对角矩阵，然后矩阵乘法
    /// - VJP 模式：直接调用 `calc_grad_to_parent`，做 O(N) 元素乘法
    ///
    /// # 参数
    /// - `source_node`: 开始传播的节点（如 hidden）
    /// - `initial_grad`: 该节点的上游梯度，形状与节点值相同
    /// - `target_params`: 需要累加 grad 的参数节点
    /// - `accumulate_params`: 是否累加 grad 到参数
    ///
    /// # 返回
    /// `HashMap<NodeId, Tensor>`: State 节点 ID -> 该节点收到的 grad
    fn bptt_propagate_to_state_vjp(
        &mut self,
        source_node: NodeId,
        initial_grad: &Tensor,
        target_params: &[NodeId],
        accumulate_params: bool,
    ) -> Result<std::collections::HashMap<NodeId, Tensor>, GraphError> {
        use std::collections::{HashMap, HashSet, VecDeque};

        let mut state_grads: HashMap<NodeId, Tensor> = HashMap::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // 起点：source_node，其 grad = initial_grad
        queue.push_back((source_node, initial_grad.clone()));
        visited.insert(source_node);

        while let Some((node_id, upstream_grad)) = queue.pop_front() {
            // 获取该节点的父节点
            let parent_ids = self.get_node_parents(node_id)?;
            if parent_ids.is_empty() {
                // 叶子节点（可能是 Parameter）
                if target_params.contains(&node_id) && accumulate_params {
                    // 累加到 Parameter 的 grad
                    // 注意：使用 get_node_grad_ref 读取 grad 字段（VJP 模式）
                    let existing_grad = self.get_node_grad_ref(node_id)?;
                    let new_grad = match existing_grad {
                        Some(existing) if existing.shape() == upstream_grad.shape() => {
                            existing + &upstream_grad
                        }
                        _ => upstream_grad.clone(),
                    };
                    self.get_node_mut(node_id)?.set_grad(Some(&new_grad))?;
                }
                continue;
            }

            // 计算所有父节点的 grad（只读阶段）
            let mut contributions: Vec<(NodeId, Tensor, bool, bool)> = Vec::new();

            {
                let node = self.get_node(node_id)?;
                for &parent_id in &parent_ids {
                    let parent = self.get_node(parent_id)?;

                    // 检查父节点类型
                    let is_input = matches!(parent.node_type(), NodeType::Input(_));
                    let is_state = matches!(parent.node_type(), NodeType::State(_));
                    let is_param = target_params.contains(&parent_id);

                    // 跳过 Input 节点
                    if is_input {
                        continue;
                    }

                    // 使用 VJP 模式计算梯度（关键：使用 calc_grad_to_parent 而非 calc_jacobi_to_a_parent）
                    let assistant_parent = parent_ids.iter().find(|&&id| id != parent_id).copied();
                    let assistant = assistant_parent.map(|id| self.get_node(id)).transpose()?;

                    let local_grad = node.calc_grad_to_parent(parent, &upstream_grad, assistant)?;

                    contributions.push((parent_id, local_grad, is_param, is_state));
                }
            }

            // 处理各类贡献（可变阶段）
            for (parent_id, local_grad, is_param, is_state) in contributions {
                if is_state {
                    // State 节点：收集 grad
                    state_grads
                        .entry(parent_id)
                        .and_modify(|existing| {
                            if existing.shape() == local_grad.shape() {
                                *existing = &*existing + &local_grad;
                            }
                        })
                        .or_insert_with(|| local_grad.clone());
                } else if is_param {
                    // Parameter 节点：根据参数决定是否累加 grad
                    if accumulate_params {
                        // 注意：使用 get_node_grad_ref 读取 grad 字段（VJP 模式）
                        let existing_grad = self.get_node_grad_ref(parent_id)?;
                        let new_grad = match existing_grad {
                            Some(existing) if existing.shape() == local_grad.shape() => {
                                existing + &local_grad
                            }
                            _ => local_grad.clone(),
                        };
                        self.get_node_mut(parent_id)?.set_grad(Some(&new_grad))?;
                    }
                    visited.insert(parent_id);
                } else {
                    // 中间节点：继续向上传播
                    if !visited.contains(&parent_id) {
                        visited.insert(parent_id);
                        queue.push_back((parent_id, local_grad));
                    }
                }
            }
        }

        Ok(state_grads)
    }

    /// BPTT 辅助方法：将 incoming grad 传播到参数（VJP 模式）
    ///
    /// 使用 VJP 而非 Jacobian 模式。与 `bptt_backward_from_node` 类似，但：
    /// - 使用 `calc_grad_to_parent` 而非 `calc_jacobi_to_a_parent`
    /// - 输入是值格式（[batch, hidden]）而非 jacobi 格式（[1, N]）
    /// - 累加到 `grad` 而非 `jacobi`
    fn bptt_backward_from_node_vjp(
        &mut self,
        source_node: NodeId,
        initial_grad: &Tensor,
        target_nodes: &[NodeId],
    ) -> Result<(), GraphError> {
        use std::collections::{HashSet, VecDeque};

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back((source_node, initial_grad.clone()));
        visited.insert(source_node);

        while let Some((node_id, upstream_grad)) = queue.pop_front() {
            let parent_ids = self.get_node_parents(node_id)?;
            if parent_ids.is_empty() {
                continue;
            }

            // 第一阶段：计算所有父节点的贡献（只读）
            let mut contributions: Vec<(NodeId, Tensor, bool)> = Vec::new();

            {
                let node = self.get_node(node_id)?;
                for &parent_id in &parent_ids {
                    // 跳过 Input 和 State 节点
                    let parent = self.get_node(parent_id)?;
                    match parent.node_type() {
                        NodeType::Input(_) => continue,
                        NodeType::State(_) => continue,
                        _ => {}
                    }

                    // 使用 VJP 计算梯度
                    let assistant_parent = parent_ids.iter().find(|&&id| id != parent_id).copied();
                    let assistant = assistant_parent.map(|id| self.get_node(id)).transpose()?;

                    let local_grad = node.calc_grad_to_parent(parent, &upstream_grad, assistant)?;

                    let should_update = target_nodes.contains(&parent_id);
                    contributions.push((parent_id, local_grad, should_update));
                }
            }

            // 第二阶段：更新 grad 和队列（可变）
            for (parent_id, local_grad, should_update) in contributions {
                if should_update {
                    // 目标节点（Parameter）：累加到 grad
                    // 注意：使用 get_node_grad_ref 读取 grad 字段（VJP 模式）
                    let existing_grad = self.get_node_grad_ref(parent_id)?;
                    let new_grad = match existing_grad {
                        Some(existing) if existing.shape() == local_grad.shape() => {
                            existing + &local_grad
                        }
                        _ => local_grad.clone(),
                    };
                    self.get_node_mut(parent_id)?.set_grad(Some(&new_grad))?;
                    visited.insert(parent_id);
                } else {
                    // 非目标节点：继续向上传播
                    if !visited.contains(&parent_id) {
                        visited.insert(parent_id);
                        queue.push_back((parent_id, local_grad));
                    }
                }
            }
        }

        Ok(())
    }

    /// 从 loss 反向传播到目标节点（VJP 模式）
    ///
    /// 使用 VJP 模式计算梯度：
    /// - 梯度存储在 `grad` 字段
    /// - 使用 `calc_grad_to_parent` 计算梯度
    /// - 支持 batch 形状
    ///
    /// # 参数
    /// - `target_params`: 参数节点（累加 grad）
    /// - `state_nodes`: State 节点（收集 grad 用于跨时间传递）
    /// - `loss_node`: loss 节点（反向传播起点）
    ///
    /// # 返回
    /// `HashMap<NodeId, Tensor>`: State 节点收到的 grad
    fn backward_from_loss_vjp(
        &mut self,
        target_params: &[NodeId],
        state_nodes: &[NodeId],
        loss_node: NodeId,
    ) -> Result<std::collections::HashMap<NodeId, Tensor>, GraphError> {
        use std::collections::{HashMap, HashSet, VecDeque};

        #[cfg(test)]
        let debug = self.bptt_debug;
        #[cfg(not(test))]
        let debug = false;

        let mut state_grads: HashMap<NodeId, Tensor> = HashMap::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // 起点：loss 节点，其 grad = 1（标量 loss 的初始梯度）
        let loss_value = self
            .get_node_value(loss_node)?
            .ok_or_else(|| GraphError::ComputationError("Loss node has no value".to_string()))?;
        let initial_grad = Tensor::ones(loss_value.shape());

        if debug {
            println!(
                "  backward_from_loss_vjp: loss={}, target_params={:?}, state_nodes={:?}",
                loss_node.0,
                target_params.iter().map(|n| n.0).collect::<Vec<_>>(),
                state_nodes.iter().map(|n| n.0).collect::<Vec<_>>()
            );
        }

        queue.push_back((loss_node, initial_grad));
        visited.insert(loss_node);

        while let Some((node_id, upstream_grad)) = queue.pop_front() {
            let parent_ids = self.get_node_parents(node_id)?;
            if parent_ids.is_empty() {
                continue;
            }

            if debug {
                let node_name = self
                    .get_node(node_id)
                    .map(|n| n.name().to_string())
                    .unwrap_or_default();
                println!(
                    "    Processing node {}({}), upstream_grad={:?}, parents={:?}",
                    node_name,
                    node_id.0,
                    upstream_grad.data_as_slice(),
                    parent_ids.iter().map(|n| n.0).collect::<Vec<_>>()
                );
            }

            // 计算所有父节点的 grad
            let mut contributions: Vec<(NodeId, Tensor, bool, bool)> = Vec::new();

            {
                let node = self.get_node(node_id)?;
                for &parent_id in &parent_ids {
                    let parent = self.get_node(parent_id)?;

                    // 检查父节点类型
                    let is_input = matches!(parent.node_type(), NodeType::Input(_));
                    let is_state = state_nodes.contains(&parent_id);
                    let is_param = target_params.contains(&parent_id);

                    if debug {
                        println!(
                            "      parent {}({}): is_input={}, is_state={}, is_param={}",
                            parent.name(),
                            parent_id.0,
                            is_input,
                            is_state,
                            is_param
                        );
                    }

                    // 跳过 Input 节点
                    if is_input {
                        continue;
                    }

                    // 使用 VJP 计算梯度
                    let assistant_parent = parent_ids.iter().find(|&&id| id != parent_id).copied();
                    let assistant = assistant_parent.map(|id| self.get_node(id)).transpose()?;

                    let local_grad = node.calc_grad_to_parent(parent, &upstream_grad, assistant)?;

                    if debug {
                        println!("        -> local_grad={:?}", local_grad.data_as_slice());
                    }

                    contributions.push((parent_id, local_grad, is_param, is_state));
                }
            }

            // 处理各类贡献
            for (parent_id, local_grad, is_param, is_state) in contributions {
                if is_state {
                    // State 节点：收集 grad（不继续向上，State 是当前时间步的叶子）
                    if debug {
                        println!(
                            "      -> State {}({}): collecting grad",
                            self.get_node(parent_id)
                                .map(|n| n.name().to_string())
                                .unwrap_or_default(),
                            parent_id.0
                        );
                    }
                    state_grads
                        .entry(parent_id)
                        .and_modify(|existing| {
                            if existing.shape() == local_grad.shape() {
                                *existing = &*existing + &local_grad;
                            }
                        })
                        .or_insert_with(|| local_grad.clone());
                } else if is_param {
                    // Parameter 节点：累加到 grad
                    if debug {
                        println!(
                            "      -> Param {}({}): setting grad",
                            self.get_node(parent_id)
                                .map(|n| n.name().to_string())
                                .unwrap_or_default(),
                            parent_id.0
                        );
                    }
                    // 注意：使用 get_node_grad_ref 读取 grad 字段（VJP 模式）
                    let existing_grad = self.get_node_grad_ref(parent_id)?;
                    let new_grad = match existing_grad {
                        Some(existing) if existing.shape() == local_grad.shape() => {
                            existing + &local_grad
                        }
                        _ => local_grad.clone(),
                    };
                    self.get_node_mut(parent_id)?.set_grad(Some(&new_grad))?;
                    visited.insert(parent_id);
                } else {
                    // 中间节点：继续向上传播
                    if debug {
                        println!(
                            "      -> Intermediate {}({}): {}",
                            self.get_node(parent_id)
                                .map(|n| n.name().to_string())
                                .unwrap_or_default(),
                            parent_id.0,
                            if visited.contains(&parent_id) {
                                "already visited"
                            } else {
                                "adding to queue"
                            }
                        );
                    }
                    if !visited.contains(&parent_id) {
                        visited.insert(parent_id);
                        queue.push_back((parent_id, local_grad));
                    }
                }
            }
        }

        Ok(state_grads)
    }

    /// 获取状态节点的梯度
    pub fn get_state_grad(&self, state_id: NodeId) -> Result<Option<&Tensor>, GraphError> {
        let node = self.get_node(state_id)?;
        Ok(node.grad())
    }
}
