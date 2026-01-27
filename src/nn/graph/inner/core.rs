/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : GraphInner 核心操作 + 前向传播
 */

use super::super::error::GraphError;
use super::super::types::{GroupKind, LayerGroup, RecurrentLayerMeta, RecurrentUnrollInfo};
use super::GraphInner;
use crate::nn::nodes::{NodeHandle, NodeType};
use crate::nn::NodeId;
use crate::tensor::Tensor;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;

impl GraphInner {
    // ========== 创建 ==========

    pub fn new() -> Self {
        Self::with_name("default_graph")
    }

    /// 创建一个带固定种子的计算图（确保可重复性）
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
            layer_groups: Vec::new(),
            recurrent_layer_metas: Vec::new(),
            recurrent_edges: HashMap::new(),
            prev_values: HashMap::new(),
            time_step: 0,
            step_history: Vec::new(),
            #[cfg(test)]
            bptt_debug: false,
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
            layer_groups: Vec::new(),
            recurrent_layer_metas: Vec::new(),
            recurrent_edges: HashMap::new(),
            prev_values: HashMap::new(),
            time_step: 0,
            step_history: Vec::new(),
            #[cfg(test)]
            bptt_debug: false,
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
            layer_groups: Vec::new(),
            recurrent_layer_metas: Vec::new(),
            recurrent_edges: HashMap::new(),
            prev_values: HashMap::new(),
            time_step: 0,
            step_history: Vec::new(),
            #[cfg(test)]
            bptt_debug: false,
        }
    }

    // ========== 基础访问器 ==========

    #[cfg(test)]
    pub(in crate::nn) fn last_forward_pass_id(&self) -> u64 {
        self.last_forward_pass_id
    }

    #[allow(dead_code)]
    pub(in crate::nn) const fn last_backward_pass_id(&self) -> u64 {
        self.last_backward_pass_id
    }

    /// 设置/重置图的随机种子
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = Some(StdRng::seed_from_u64(seed));
    }

    /// 检查图是否有固定种子
    pub const fn has_seed(&self) -> bool {
        self.rng.is_some()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn nodes(&self) -> Vec<NodeId> {
        self.nodes.keys().copied().collect()
    }

    pub fn nodes_count(&self) -> usize {
        self.nodes.len()
    }

    pub(in crate::nn) fn get_node(&self, id: NodeId) -> Result<&NodeHandle, GraphError> {
        self.nodes
            .get(&id)
            .ok_or(GraphError::NodeNotFound(id))
    }

    pub(in crate::nn) fn get_node_mut(&mut self, id: NodeId) -> Result<&mut NodeHandle, GraphError> {
        self.nodes
            .get_mut(&id)
            .ok_or(GraphError::NodeNotFound(id))
    }

    pub(in crate::nn) fn get_nodes(&self, ids: &[NodeId]) -> Result<Vec<&NodeHandle>, GraphError> {
        ids.iter().map(|&id| self.get_node(id)).collect()
    }

    pub fn get_node_parents(&self, id: NodeId) -> Result<Vec<NodeId>, GraphError> {
        // 先检查节点是否存在
        let _ = self.get_node(id)?;
        Ok(self
            .backward_edges
            .get(&id)
            .cloned()
            .unwrap_or_default())
    }

    pub fn get_node_children(&self, id: NodeId) -> Result<Vec<NodeId>, GraphError> {
        // 先检查节点是否存在
        let _ = self.get_node(id)?;
        Ok(self
            .forward_edges
            .get(&id)
            .cloned()
            .unwrap_or_default())
    }

    pub fn get_node_name(&self, id: NodeId) -> Result<&str, GraphError> {
        Ok(self.get_node(id)?.name())
    }

    pub fn has_node_value(&self, node_id: NodeId) -> Result<bool, GraphError> {
        self.nodes
            .get(&node_id)
            .map(NodeHandle::has_value)
            .ok_or(GraphError::NodeNotFound(node_id))
    }

    pub fn get_node_value(&self, id: NodeId) -> Result<Option<&Tensor>, GraphError> {
        Ok(self.get_node(id)?.value())
    }

    pub fn set_node_value(&mut self, id: NodeId, value: Option<&Tensor>) -> Result<(), GraphError> {
        self.get_node_mut(id)?.set_value(value)
    }

    pub fn get_node_grad(&self, id: NodeId) -> Result<Option<Tensor>, GraphError> {
        let node = self.get_node(id)?;
        // 输入节点不应该有梯度
        if let NodeType::Input(_) = node.node_type() {
            return Err(GraphError::InvalidOperation(format!(
                "输入{node}不应该有梯度"
            )));
        }
        Ok(node.grad().cloned())
    }

    pub fn get_node_grad_ref(&self, node_id: NodeId) -> Result<Option<&Tensor>, GraphError> {
        let node = self.get_node(node_id)?;
        if let NodeType::Input(_) = node.node_type() {
            return Err(GraphError::InvalidOperation(format!(
                "输入{node}不应该有梯度"
            )));
        }
        Ok(node.grad())
    }

    /// 获取所有可训练的参数节点
    pub fn get_trainable_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter_map(|(&id, node)| {
                if let NodeType::Parameter(_) = node.node_type() {
                    Some(id)
                } else {
                    None
                }
            })
            .collect()
    }

    // ========== ID/名称生成 ==========

    pub(in crate::nn::graph) fn generate_valid_node_id(&mut self) -> NodeId {
        // 生成唯一的节点ID（先递增再返回，所以第一个节点 ID 是 1）
        self.next_id += 1;
        NodeId(self.next_id)
    }

    pub(in crate::nn::graph) fn check_duplicate_node_name(&self, name: &str) -> Result<(), GraphError> {
        if self.nodes.values().any(|node| node.name() == name) {
            return Err(GraphError::DuplicateNodeName(format!(
                "节点{}在图{}中重复",
                name,
                self.name()
            )));
        }
        Ok(())
    }

    pub(in crate::nn::graph) fn generate_valid_new_node_name(
        &self,
        base_name: &str,
        node_type: &str,
    ) -> Result<String, GraphError> {
        if !base_name.is_empty() {
            self.check_duplicate_node_name(base_name)?;
            return Ok(base_name.to_string());
        }

        let mut counter = 1;
        loop {
            let name = format!("{node_type}_{counter}");
            if self.check_duplicate_node_name(&name).is_ok() {
                return Ok(name);
            }
            counter += 1;
        }
    }

    // ========== 层分组相关 ==========

    /// 获取所有层分组信息
    pub fn layer_groups(&self) -> &[LayerGroup] {
        &self.layer_groups
    }

    /// 注册一个层分组
    pub fn register_layer_group(
        &mut self,
        name: &str,
        layer_type: &str,
        description: &str,
        node_ids: Vec<NodeId>,
    ) {
        if self.layer_groups.iter().any(|g| g.name == name) {
            return;
        }
        self.layer_groups.push(LayerGroup {
            name: name.to_string(),
            layer_type: layer_type.to_string(),
            description: description.to_string(),
            node_ids,
            kind: GroupKind::Layer,
            recurrent_steps: None,
            min_steps: None,
            max_steps: None,
            hidden_node_ids: vec![],
            folded_nodes: vec![],
            output_proxy: None,
        });
    }

    /// 注册循环层元信息
    pub fn register_recurrent_layer_meta(
        &mut self,
        name: &str,
        layer_type: &str,
        description: &str,
        param_node_ids: Vec<NodeId>,
        nodes_per_step: usize,
    ) {
        if self.recurrent_layer_metas.iter().any(|m| m.name == name) {
            return;
        }
        self.recurrent_layer_metas.push(RecurrentLayerMeta {
            name: name.to_string(),
            layer_type: layer_type.to_string(),
            description: description.to_string(),
            param_node_ids,
            nodes_per_step,
            unroll_infos: Vec::new(),
        });
    }

    /// 追加循环层的展开信息
    pub fn update_recurrent_layer_unroll_info(
        &mut self,
        name: &str,
        unroll_info: RecurrentUnrollInfo,
    ) {
        if let Some(meta) = self.recurrent_layer_metas.iter_mut().find(|m| m.name == name) {
            meta.unroll_infos.push(unroll_info);
        }
    }

    /// 获取循环层元信息列表
    pub fn recurrent_layer_metas(&self) -> &[RecurrentLayerMeta] {
        &self.recurrent_layer_metas
    }

    // ========== 前向传播 ==========

    pub fn forward(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        let node = self.get_node(node_id)?;
        match node.node_type() {
            NodeType::Input(_) | NodeType::Parameter(_) | NodeType::State(_) => {
                if node.has_value() {
                    return Ok(());
                }
                return Err(GraphError::InvalidOperation(format!(
                    "{node}是输入/参数/状态类型，其值应通过 set_value 设置，而非通过父节点前向传播计算"
                )));
            }
            _ => {}
        }

        let new_graph_forward_pass_id = self.last_forward_pass_id + 1;
        self.forward_node_internal(node_id, new_graph_forward_pass_id)?;
        self.last_forward_pass_id = new_graph_forward_pass_id;
        Ok(())
    }

    fn forward_node_internal(
        &mut self,
        node_id: NodeId,
        new_graph_forward_pass_id: u64,
    ) -> Result<(), GraphError> {
        let node = self.get_node_mut(node_id)?;

        match node.node_type() {
            NodeType::Input(_) | NodeType::Parameter(_) | NodeType::State(_) => {
                if node.has_value() {
                    node.set_last_forward_pass_id(new_graph_forward_pass_id);
                    return Ok(());
                }
                return Err(GraphError::InvalidOperation(format!(
                    "{}不能直接前向传播",
                    node
                )));
            }
            _ => {
                if node.last_forward_pass_id() == new_graph_forward_pass_id {
                    return Ok(());
                }
            }
        }

        let parents_ids = self.get_node_parents(node_id)?;
        for parent_id in &parents_ids {
            self.forward_node_internal(*parent_id, new_graph_forward_pass_id)?;
        }

        let parent_nodes = parents_ids
            .iter()
            .map(|id| self.get_node(*id).unwrap().clone())
            .collect::<Vec<NodeHandle>>();

        let node = self.get_node_mut(node_id)?;
        node.calc_value_by_parents(&parent_nodes)?;
        node.set_last_forward_pass_id(new_graph_forward_pass_id);

        Ok(())
    }

    /// 释放中间节点的值和梯度
    pub(in crate::nn::graph) fn release_intermediate_results(&mut self) -> Result<(), GraphError> {
        for node in self.nodes.values_mut() {
            match node.node_type() {
                NodeType::Input(_) | NodeType::Parameter(_) | NodeType::State(_) => {}
                _ => {
                    node.clear_value()?;
                    let _ = node.clear_grad();
                }
            }
        }
        Ok(())
    }

    /// 重置中间节点的 grad
    pub(in crate::nn::graph) fn reset_intermediate_grad(&mut self) {
        for node in self.nodes.values_mut() {
            match node.node_type() {
                NodeType::Parameter(_) => {}
                _ => {
                    let _ = node.clear_grad();
                    node.set_last_backward_pass_id(0);
                }
            }
        }
    }
}
