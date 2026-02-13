/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @LastEditTime : 2026-02-02
 * @Description  : GraphInner 核心操作
 *
 * 已移除旧的 forward/get_node 系列方法
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
            last_forward_pass_id: 0,
            last_backward_pass_id: 0,
            next_id: 0,
            is_eval_mode: false,
            rng: Some(StdRng::seed_from_u64(seed)),
            layer_groups: Vec::new(),
            recurrent_layer_metas: Vec::new(),
            parameters: HashMap::new(),
            node_type_counts: HashMap::new(),
            counts_reset_pass_id: 0,
            node_group_context: None,
            next_node_group_id: 0,
            visualization_snapshot: None,
        }
    }

    /// 创建一个带名称和固定种子的计算图
    pub fn with_name_and_seed(name: &str, seed: u64) -> Self {
        Self {
            name: name.to_string(),
            last_forward_pass_id: 0,
            last_backward_pass_id: 0,
            next_id: 0,
            is_eval_mode: false,
            rng: Some(StdRng::seed_from_u64(seed)),
            layer_groups: Vec::new(),
            recurrent_layer_metas: Vec::new(),
            parameters: HashMap::new(),
            node_type_counts: HashMap::new(),
            counts_reset_pass_id: 0,
            node_group_context: None,
            next_node_group_id: 0,
            visualization_snapshot: None,
        }
    }

    pub fn with_name(name: &str) -> Self {
        Self {
            name: name.to_string(),
            last_forward_pass_id: 0,
            last_backward_pass_id: 0,
            next_id: 0,
            is_eval_mode: false,
            rng: None,
            layer_groups: Vec::new(),
            recurrent_layer_metas: Vec::new(),
            parameters: HashMap::new(),
            node_type_counts: HashMap::new(),
            counts_reset_pass_id: 0,
            node_group_context: None,
            next_node_group_id: 0,
            visualization_snapshot: None,
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

    /// 获取下一个节点分组实例 ID（全局递增）
    pub fn next_node_group_instance_id(&mut self) -> usize {
        let id = self.next_node_group_id;
        self.next_node_group_id += 1;
        id
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
    #[allow(dead_code)]
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

    // ========== 参数注册表 API ==========

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

    /// 清零参数梯度
    ///
    /// 遍历 parameters 注册表，清除每个有效参数节点的梯度。
    ///
    /// # 注意
    /// 中间节点的梯度会在每次 `backward_via_node_inner` 开始时自动清除，
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

    // ========== 前向传播 API ==========

    /// 通过 NodeInner 进行前向传播
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

    /// 通过 NodeInner 进行反向传播
    ///
    /// 直接使用 `Rc<NodeInner>` 的拓扑逆序反向传播，不依赖 GraphInner 中存储的节点。
    /// 动态图架构下，中间结果由 Rc 引用计数管理，天然支持多次 backward，
    /// 无需显式的 `retain_graph` 参数。
    ///
    /// # 参数
    /// - `node`: loss 节点
    ///
    /// # 返回
    /// - `Ok(loss_scalar)`: 反向传播成功，返回 loss 标量值
    /// - `Err(GraphError)`: 反向传播失败
    pub fn backward_via_node_inner(
        &mut self,
        node: &std::rc::Rc<crate::nn::nodes::NodeInner>,
    ) -> Result<f32, GraphError> {
        use crate::tensor::Tensor;

        // 1. 检查训练模式
        if !self.is_train_mode() {
            eprintln!("[only_torch 警告] 在 no_grad/eval 模式下调用 backward，这通常是误用。");
        }

        // 2. 获取 loss 值并验证
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

        // 4. 清除中间节点梯度（避免多次 backward 时梯度在中间节点累积）
        //
        // 设计说明：
        // - 参数节点的梯度由用户通过 zero_grad() 控制（支持梯度累积语义）
        // - 中间节点（运算节点）的梯度必须每次 backward 重新计算
        // - 这与 PyTorch 行为一致：PyTorch 每次 forward 重建图，中间梯度自然不会累积
        //   本框架复用计算图，因此需要手动清除
        {
            let topo_order = node.backward_topo_order();
            for n in &topo_order {
                if !n.is_leaf() {
                    let _ = n.clear_grad();
                }
            }
        }

        // 5. 设置 loss 梯度为 1
        let loss_grad = Tensor::ones(&[1, 1]);
        node.set_grad(Some(&loss_grad))?;

        // 6. 执行反向传播
        let pass_id = self.last_backward_pass_id + 1;
        node.backward_propagate(pass_id)?;
        self.last_backward_pass_id = pass_id;

        Ok(loss_scalar)
    }
}
