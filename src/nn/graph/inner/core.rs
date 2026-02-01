/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @LastEditTime : 2026-02-02
 * @Description  : GraphInner 核心操作
 *
 * 方案 C 清理：已移除旧的 forward/get_node 系列方法
 * 新架构使用 NodeInner 进行前向/反向传播，不依赖 nodes HashMap
 */

use super::super::error::GraphError;
use super::super::types::{GroupKind, LayerGroup, RecurrentLayerMeta, RecurrentUnrollInfo};
use super::GraphInner;
use crate::nn::NodeId;
use rand::SeedableRng;
use rand::rngs::StdRng;
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
            parameters: HashMap::new(), // 方案 C 新增
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
            parameters: HashMap::new(), // 方案 C 新增
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
            parameters: HashMap::new(), // 方案 C 新增
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

    /// 生成下一个随机种子（用于 Dropout 等需要独立 rng 的节点）
    ///
    /// 如果 graph 有 rng，则从 rng 生成；否则使用 `thread_rng`
    pub(in crate::nn::graph) fn next_seed(&mut self) -> u64 {
        use rand::Rng;
        if let Some(ref mut rng) = self.rng {
            rng.r#gen()
        } else {
            rand::thread_rng().r#gen()
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    // ========== 方案 C：参数注册表 API ==========

    /// 注册参数到参数注册表
    ///
    /// # 参数
    /// - `name`: 参数名称（如 "linear1.weight"）
    /// - `node`: 参数节点的弱引用
    ///
    /// # 返回
    /// - 如果名称已存在且指向有效节点，返回错误
    /// - 如果名称已存在但节点已失效，则替换
    pub fn register_parameter(
        &mut self,
        name: String,
        node: std::rc::Weak<crate::nn::nodes::NodeInner>,
    ) -> Result<(), GraphError> {
        // 检查是否已存在同名参数
        if let Some(existing) = self.parameters.get(&name) {
            if existing.upgrade().is_some() {
                return Err(GraphError::InvalidOperation(format!(
                    "参数 '{}' 已存在且仍有效",
                    name
                )));
            }
            // 已失效，允许替换
        }
        self.parameters.insert(name, node);
        Ok(())
    }

    /// 获取指定名称的参数
    ///
    /// 如果参数存在且仍有效，返回其强引用
    pub fn get_parameter(&self, name: &str) -> Option<std::rc::Rc<crate::nn::nodes::NodeInner>> {
        self.parameters.get(name).and_then(|weak| weak.upgrade())
    }

    /// 获取所有有效的参数（过滤掉已失效的弱引用）
    ///
    /// 返回 (名称, 节点) 对的列表
    pub fn get_all_parameters(&self) -> Vec<(String, std::rc::Rc<crate::nn::nodes::NodeInner>)> {
        self.parameters
            .iter()
            .filter_map(|(name, weak)| weak.upgrade().map(|rc| (name.clone(), rc)))
            .collect()
    }

    /// 获取参数注册表中的参数数量（包括可能已失效的）
    pub fn registered_parameters_count(&self) -> usize {
        self.parameters.len()
    }

    /// 获取有效参数的数量（过滤掉已失效的）
    pub fn valid_parameters_count(&self) -> usize {
        self.parameters
            .values()
            .filter(|w| w.upgrade().is_some())
            .count()
    }

    /// 清理已失效的参数引用
    ///
    /// 返回清理掉的数量
    pub fn cleanup_dead_parameters(&mut self) -> usize {
        let before = self.parameters.len();
        self.parameters.retain(|_, weak| weak.upgrade().is_some());
        before - self.parameters.len()
    }

    /// 清零参数梯度（方案 C 新路径）
    ///
    /// 遍历 parameters 注册表，清除每个有效参数节点的梯度。
    /// 这是新路径的 `zero_grad` 实现，与旧路径的 `clear_grad()` 不同：
    /// - 旧路径：遍历 `nodes` HashMap 清除所有节点
    /// - 新路径：只遍历 `parameters` 注册表清除参数节点
    ///
    /// # 注意
    /// 中间节点的梯度会在 `backward_via_node_inner` 完成后被释放（如果 retain_graph=false），
    /// 所以只清除参数梯度是正确的行为。
    pub fn zero_grad_via_parameters(&self) -> Result<(), GraphError> {
        for weak in self.parameters.values() {
            if let Some(node) = weak.upgrade() {
                node.clear_grad()?;
            }
        }
        Ok(())
    }

    /// 检查参数是否已注册
    pub fn has_parameter(&self, name: &str) -> bool {
        self.parameters
            .get(name)
            .map(|w| w.upgrade().is_some())
            .unwrap_or(false)
    }

    // ========== ID/名称生成 ==========

    pub(in crate::nn::graph) const fn generate_valid_node_id(&mut self) -> NodeId {
        // 生成唯一的节点ID（先递增再返回，所以第一个节点 ID 是 1）
        self.next_id += 1;
        NodeId(self.next_id)
    }

    // ========== 层分组相关 ==========

    /// 获取所有层分组信息
    pub fn layer_groups(&self) -> &[LayerGroup] {
        &self.layer_groups
    }

    /// 注册一个层分组
    ///
    /// 如果同名分组已存在，会将新的节点 ID 追加到该分组中（避免重复）。
    /// 这支持共享层（如 Siamese 网络中的共享编码器）的正确可视化：
    /// 多次 forward 调用产生的操作节点都会归入同一个 Layer cluster。
    pub fn register_layer_group(
        &mut self,
        name: &str,
        layer_type: &str,
        description: &str,
        node_ids: Vec<NodeId>,
    ) {
        // 查找已存在的同名分组
        if let Some(group) = self.layer_groups.iter_mut().find(|g| g.name == name) {
            // 扩展节点列表（避免重复）
            for id in node_ids {
                if !group.node_ids.contains(&id) {
                    group.node_ids.push(id);
                }
            }
            return;
        }

        // 新建分组
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
        if let Some(meta) = self
            .recurrent_layer_metas
            .iter_mut()
            .find(|m| m.name == name)
        {
            meta.unroll_infos.push(unroll_info);
        }
    }

    /// 获取循环层元信息列表
    pub fn recurrent_layer_metas(&self) -> &[RecurrentLayerMeta] {
        &self.recurrent_layer_metas
    }

    // ========== 方案 C：新前向传播 API ==========

    /// 通过 NodeInner 进行前向传播（方案 C）
    ///
    /// 与旧 `forward()` 方法不同，此方法直接使用 `Rc<NodeInner>` 的递归前向传播，
    /// 不依赖 GraphInner 中存储的节点。
    ///
    /// # 参数
    /// - `node`: 目标节点（通常是 loss 节点）
    ///
    /// # 返回
    /// - `Ok(())`: 前向传播成功
    /// - `Err(GraphError)`: 前向传播失败
    pub fn forward_via_node_inner(
        &mut self,
        node: &std::rc::Rc<crate::nn::nodes::NodeInner>,
    ) -> Result<(), GraphError> {
        let pass_id = self.last_forward_pass_id + 1;
        let is_training = !self.is_eval_mode;
        node.forward_recursive(pass_id, is_training)?;
        self.last_forward_pass_id = pass_id;
        Ok(())
    }

    /// 通过 NodeInner 进行反向传播（方案 C）
    ///
    /// 与旧 `backward()` 方法不同，此方法直接使用 `Rc<NodeInner>` 的拓扑逆序反向传播，
    /// 不依赖 GraphInner 中存储的节点。
    ///
    /// # 参数
    /// - `node`: loss 节点
    /// - `retain_graph`: 是否保留计算图（默认 false，释放中间结果）
    ///
    /// # 返回
    /// - `Ok(loss_scalar)`: 反向传播成功，返回 loss 标量值
    /// - `Err(GraphError)`: 反向传播失败
    ///
    /// # 注意
    /// - 当前不支持 BPTT（循环网络），BPTT 功能将在后续版本评估
    /// - 如果检测到 step_history 非空，会发出警告
    pub fn backward_via_node_inner(
        &mut self,
        node: &std::rc::Rc<crate::nn::nodes::NodeInner>,
        retain_graph: bool,
    ) -> Result<f32, GraphError> {
        use crate::tensor::Tensor;

        // 1. 检查训练模式
        if !self.is_train_mode() {
            eprintln!("[only_torch 警告] 在 no_grad/eval 模式下调用 backward，这通常是误用。");
        }

        // 2. 检查 BPTT（当前不支持）
        if !self.step_history.is_empty() && !self.recurrent_edges.is_empty() {
            eprintln!(
                "[only_torch 警告] 检测到循环网络结构，但 backward_via_node_inner 当前不支持 BPTT。\
                 请使用旧路径 backward() 或等待后续版本。"
            );
        }

        // 3. 获取 loss 值并验证
        let loss_value = node.value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "损失节点 {}[{}] 没有值，请先执行 forward",
                node.type_name(),
                node.id()
            ))
        })?;

        if loss_value.size() != 1 {
            return Err(GraphError::InvalidOperation(format!(
                "反向传播要求损失为标量（size=1），但得到 shape={:?}",
                loss_value.shape()
            )));
        }

        let loss_scalar = loss_value.get_data_number().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "无法从损失节点获取标量值，形状: {:?}",
                loss_value.shape()
            ))
        })?;

        // 4. 设置 loss 梯度为 1
        let loss_grad = Tensor::ones(&[1, 1]);
        node.set_grad(Some(&loss_grad))?;

        // 5. 执行反向传播
        let pass_id = self.last_backward_pass_id + 1;
        node.backward_propagate(pass_id)?;
        self.last_backward_pass_id = pass_id;

        // 6. 中间结果释放说明
        // 方案 C 架构下，中间节点由 Rc<NodeInner> 管理，
        // 当 Var 离开作用域时自动释放，无需显式清理。
        // retain_graph 参数暂时保留以保持 API 兼容性，
        // 未来可能用于控制是否清除中间节点的梯度。
        let _ = retain_graph; // 标记参数已使用，避免 warning

        Ok(loss_scalar)
    }
}
