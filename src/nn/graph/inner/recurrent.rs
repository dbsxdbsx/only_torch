/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : GraphInner 循环机制（connect_recurrent, step, reset）
 */

use super::super::error::GraphError;
use super::super::types::StepSnapshot;
use super::GraphInner;
use crate::nn::NodeId;
use std::collections::HashMap;

impl GraphInner {
    // ========== 循环/记忆机制 API ==========

    /// 声明循环连接
    pub fn connect_recurrent(
        &mut self,
        from_node: NodeId,
        to_node: NodeId,
    ) -> Result<(), GraphError> {
        self.get_node(from_node)?;
        self.get_node(to_node)?;

        if self.recurrent_edges.contains_key(&to_node) {
            return Err(GraphError::InvalidOperation(format!(
                "节点 {} 已经有循环连接源，不能重复声明",
                self.get_node_name(to_node)?
            )));
        }

        self.recurrent_edges.insert(to_node, from_node);
        Ok(())
    }

    /// 获取节点的循环连接源
    pub fn get_recurrent_source(&self, to_node: NodeId) -> Option<NodeId> {
        self.recurrent_edges.get(&to_node).copied()
    }

    /// 检查图中是否有循环连接
    pub fn has_recurrent_edges(&self) -> bool {
        !self.recurrent_edges.is_empty()
    }

    /// 获取当前时间步
    pub const fn current_time_step(&self) -> u64 {
        self.time_step
    }

    /// 执行一个时间步的前向传播
    pub fn step(&mut self, output_node: NodeId) -> Result<(), GraphError> {
        for (&to_node, &from_node) in &self.recurrent_edges.clone() {
            let prev_value = self.prev_values.get(&from_node).cloned();
            if let Some(value) = prev_value {
                self.set_node_value(to_node, Some(&value))?;
            }
        }

        self.forward(output_node)?;

        for &from_node in self.recurrent_edges.values() {
            if let Some(value) = self.get_node_value(from_node)? {
                self.prev_values.insert(from_node, value.clone());
            }
        }

        if self.is_train_mode() {
            let snapshot = self.capture_snapshot();
            self.step_history.push(snapshot);
        }

        self.time_step += 1;
        Ok(())
    }

    /// 捕获当前所有节点的快照
    pub(in crate::nn::graph) fn capture_snapshot(&self) -> HashMap<NodeId, StepSnapshot> {
        self.nodes
            .iter()
            .map(|(&id, node)| {
                (
                    id,
                    StepSnapshot {
                        value: node.value().cloned(),
                    },
                )
            })
            .collect()
    }

    /// 恢复节点值到指定快照
    pub(in crate::nn::graph) fn restore_snapshot(&mut self, snapshot: &HashMap<NodeId, StepSnapshot>) {
        for (&node_id, snap) in snapshot {
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.set_value_unchecked(snap.value.as_ref());
            }
        }
    }

    /// 重置循环状态
    ///
    /// 清除历史记录并将所有循环目标节点重置为零张量。
    /// 只有全部节点重置成功后才会重置时间步，确保状态一致性。
    pub fn reset(&mut self) -> Result<(), GraphError> {
        self.prev_values.clear();
        self.step_history.clear();

        // 先收集需要处理的节点 ID，避免借用冲突
        let to_nodes: Vec<_> = self.recurrent_edges.keys().copied().collect();

        for to_node in to_nodes {
            let shape = self.get_node(to_node)?.value_expected_shape().to_vec();
            let zeros = crate::tensor::Tensor::zeros(&shape);
            self.get_node_mut(to_node)?.set_value(Some(&zeros))?;
        }

        // 全部成功后才重置时间步，确保状态一致性
        self.time_step = 0;
        Ok(())
    }
}
