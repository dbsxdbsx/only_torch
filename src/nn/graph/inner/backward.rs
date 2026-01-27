/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : GraphInner VJP 反向传播
 */

use super::super::error::GraphError;
use super::GraphInner;
use crate::nn::nodes::NodeType;
use crate::nn::NodeId;
use crate::tensor::Tensor;
use std::collections::HashSet;

impl GraphInner {
    // ========== VJP 反向传播核心 ==========

    /// 反向传播
    pub fn backward(&mut self, loss: NodeId) -> Result<f32, GraphError> {
        self.backward_ex(loss, false)
    }

    /// 反向传播（扩展版本）
    pub fn backward_ex(&mut self, loss: NodeId, retain_graph: bool) -> Result<f32, GraphError> {
        let loss_node = self.get_node(loss)?;
        let loss_value = loss_node.value().ok_or_else(|| {
            GraphError::ComputationError(format!("损失{loss_node}没有值，请先执行 forward"))
        })?;

        let loss_scalar = loss_value.get_data_number().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "无法从损失节点获取标量值，形状: {:?}",
                loss_value.shape()
            ))
        })?;

        let needs_bptt = !self.step_history.is_empty() && !self.recurrent_edges.is_empty();

        if needs_bptt {
            let param_ids = self.get_trainable_nodes();
            self.backward_through_time(&param_ids, loss)?;
        } else {
            self.backward_vjp_core(loss)?;
        }

        if !retain_graph {
            self.release_intermediate_results()?;
        }

        Ok(loss_scalar)
    }

    /// VJP 反向传播核心实现
    pub(in crate::nn::graph) fn backward_vjp_core(&mut self, loss_id: NodeId) -> Result<(), GraphError> {
        if !self.is_train_mode() {
            eprintln!(
                "[only_torch 警告] 在 no_grad/eval 模式下调用 backward，这通常是误用。"
            );
        }

        self.reset_intermediate_grad();

        let loss_node = self.get_node(loss_id)?;
        let loss_value = loss_node.value().ok_or_else(|| {
            GraphError::ComputationError(format!("损失{loss_node}没有值，请先执行 forward"))
        })?;

        if loss_value.size() != 1 {
            return Err(GraphError::InvalidOperation(format!(
                "反向传播要求损失为标量 [1, 1]，但得到 {:?}",
                loss_value.shape()
            )));
        }

        let loss_grad = Tensor::ones(&[1, 1]);
        self.get_node_mut(loss_id)?.set_grad(Some(&loss_grad))?;

        let topo_order = self.topological_sort_backward(loss_id)?;

        for node_id in &topo_order {
            self.propagate_grad_to_parents(*node_id, loss_id, None)?;
        }

        let routed_targets = self.process_gradient_routing()?;
        for target_id in routed_targets {
            self.backward_from_node(target_id)?;
        }

        self.last_backward_pass_id += 1;
        let new_pass_id = self.last_backward_pass_id;

        for node_id in topo_order {
            if let Ok(node) = self.get_node_mut(node_id) {
                if node.grad().is_some() {
                    node.set_last_backward_pass_id(new_pass_id);
                }
            }
        }

        Ok(())
    }

    /// 处理梯度路由
    fn process_gradient_routing(&mut self) -> Result<Vec<NodeId>, GraphError> {
        use crate::nn::nodes::raw_node::InputVariant;

        let mut routing_info: Vec<(NodeId, Tensor)> = Vec::new();

        for node in self.nodes.values() {
            if let NodeType::Input(
                InputVariant::Smart(smart) | InputVariant::RecurrentOutput(smart),
            ) = node.node_type()
            {
                if let Some(target_id) = smart.gradient_target() {
                    if !smart.is_detached() {
                        if let Some(grad) = node.grad() {
                            routing_info.push((target_id, grad.clone()));
                        }
                    }
                }
            }
        }

        let mut routed_targets = Vec::new();
        for (target_id, grad) in routing_info {
            if let Ok(target_node) = self.get_node_mut(target_id) {
                if let Some(existing_grad) = target_node.grad() {
                    let new_grad = existing_grad + &grad;
                    target_node.set_grad(Some(&new_grad))?;
                } else {
                    target_node.set_grad(Some(&grad))?;
                }
                routed_targets.push(target_id);
            }
        }

        Ok(routed_targets)
    }

    /// 从指定节点继续反向传播
    fn backward_from_node(&mut self, start_id: NodeId) -> Result<(), GraphError> {
        let topo_order = self.topological_sort_backward(start_id)?;
        for node_id in &topo_order {
            self.propagate_grad_to_parents(*node_id, start_id, None)?;
        }
        Ok(())
    }

    /// 将梯度从当前节点传播到其父节点
    fn propagate_grad_to_parents(
        &mut self,
        node_id: NodeId,
        _loss_id: NodeId,
        target_params: Option<&HashSet<NodeId>>,
    ) -> Result<(), GraphError> {
        use crate::nn::nodes::raw_node::InputVariant;

        {
            let node = self.get_node(node_id)?;
            if node.is_detached() {
                return Ok(());
            }
        }

        let parent_ids = self.get_node_parents(node_id)?;
        if parent_ids.is_empty() {
            return Ok(());
        }

        let parent_grads: Vec<(NodeId, Tensor)> = {
            let node = self.get_node(node_id)?;
            let upstream_grad = match node.grad() {
                Some(g) => g,
                None => return Ok(()),
            };

            let mut grads = Vec::with_capacity(parent_ids.len());
            for parent_id in &parent_ids {
                let parent = self.get_node(*parent_id)?;

                if let NodeType::Input(variant) = parent.node_type() {
                    match variant {
                        InputVariant::Data(_) | InputVariant::Target(_) => continue,
                        InputVariant::Smart(_) | InputVariant::RecurrentOutput(_) => {}
                    }
                }

                if let Some(targets) = target_params {
                    if let NodeType::Parameter(_) = parent.node_type() {
                        if !targets.contains(parent_id) {
                            continue;
                        }
                    }
                }

                let assistant_parent_id = parent_ids.iter().find(|&&id| id != *parent_id).copied();
                let assistant = assistant_parent_id
                    .map(|id| self.get_node(id))
                    .transpose()?;

                let parent_grad = node.calc_grad_to_parent(parent, upstream_grad, assistant)?;
                grads.push((*parent_id, parent_grad));
            }
            grads
        };

        for (parent_id, parent_grad) in parent_grads {
            let parent_node = self.get_node_mut(parent_id)?;

            if parent_node.is_detached() {
                continue;
            }

            if let Some(existing_grad) = parent_node.grad() {
                let new_grad = existing_grad + &parent_grad;
                parent_node.set_grad(Some(&new_grad))?;
            } else {
                parent_node.set_grad(Some(&parent_grad))?;
            }
        }

        Ok(())
    }

    /// 拓扑排序（反向）
    fn topological_sort_backward(&self, loss_id: NodeId) -> Result<Vec<NodeId>, GraphError> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();

        fn dfs(
            graph: &GraphInner,
            node_id: NodeId,
            visited: &mut HashSet<NodeId>,
            result: &mut Vec<NodeId>,
        ) -> Result<(), GraphError> {
            if visited.contains(&node_id) {
                return Ok(());
            }
            visited.insert(node_id);
            result.push(node_id);

            let parents = graph.get_node_parents(node_id)?;
            for parent_id in parents {
                dfs(graph, parent_id, visited, result)?;
            }

            Ok(())
        }

        dfs(self, loss_id, &mut visited, &mut result)?;
        Ok(result)
    }

    /// 清除所有节点的梯度
    pub fn clear_grad(&mut self) -> Result<(), GraphError> {
        for node in self.nodes.values_mut() {
            let _ = node.clear_grad();
        }
        Ok(())
    }

    /// 清除单个节点的梯度
    pub fn clear_node_grad(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        let node = self.get_node_mut(node_id)?;
        let _ = node.clear_grad();
        Ok(())
    }

    /// 清零梯度（PyTorch 风格）
    pub fn zero_grad(&mut self) -> Result<(), GraphError> {
        self.clear_grad()
    }

    /// 拓扑变化通知
    pub fn on_topology_changed(&mut self) {
        for node in self.nodes.values_mut() {
            let _ = node.clear_grad();
            node.set_last_backward_pass_id(0);
        }
    }
}
