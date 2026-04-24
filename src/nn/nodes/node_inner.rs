/*
 * @Author       : 老董
 * @Date         : 2026-02-01
 * @Description  : NodeInner - 核心节点结构
 *
 * 这是动态图生命周期管理的核心组件：
 * - 被 Rc 包装，通过引用计数控制生命周期
 * - 持有 parents 强引用，保证反向传播时上游存活
 * - 当 Var 离开作用域且无其他引用时，节点自动释放
 */

use super::NodeType;
use super::raw_node::{GradResult, TraitNode};
use crate::nn::NodeId;
use crate::nn::graph::GraphError;
use crate::nn::graph::NodeGroupTag;
use crate::tensor::Tensor;
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::rc::Rc;

/// 节点内部结构
///
/// 与旧的 `NodeHandle` 不同，`NodeInner` 直接被 `Rc` 包装：
/// - `Var` 持有 `Rc<NodeInner>`，控制节点生命周期
/// - `parents` 持有 `Rc<NodeInner>`，保证反向传播时上游存活
/// - 当所有引用消失时，节点自动释放（级联释放）
///
/// # 内部可变性
/// - `raw_node` 使用 `RefCell`（value/grad 是非 Copy 类型）
/// - `last_forward_pass_id` 等使用 `Cell`（Copy 类型）
pub struct NodeInner {
    // === 节点标识（用于可视化/调试）===
    id: NodeId,
    name: Option<String>,

    // === 核心内容（使用 RefCell，因为 value/grad 是非 Copy 类型）===
    raw_node: RefCell<NodeType>,

    // === 内部可变字段（使用 Cell，适用于 Copy 类型）===
    last_forward_pass_id: Cell<u64>,
    last_backward_pass_id: Cell<u64>,
    is_detached: Cell<bool>,

    // === 父节点引用（强引用，保证反向传播时存活）===
    /// 父节点列表，顺序与 `raw_node.calc_value_by_parents` 的参数顺序一致
    parents: Vec<Rc<NodeInner>>,

    // === 节点分组标签（用于可视化 cluster）===
    /// 如果该节点是在某个分组上下文中创建的，则记录所属分组。
    /// 使用 RefCell 支持后补标签（如 Layer 的 Parameter 节点在 guard 之前创建）。
    node_group_tag: RefCell<Option<NodeGroupTag>>,

    // === ONNX 来源追溯（provenance）===
    /// 如果节点由 ONNX 导入而来，记录原 ONNX 模型的节点名链路
    /// （详见 `NodeDescriptor::origin_onnx_nodes`）。
    /// 演化、单元测试、Layer API 等非 ONNX 路径下默认空 `Vec`。
    /// 使用 `RefCell` 支持后补（rebuild_node 创建完 NodeInner 后再注入）。
    origin_onnx_nodes: RefCell<Vec<String>>,
}

impl NodeInner {
    /// 创建新的 NodeInner
    ///
    /// # 参数
    /// - `id`: 节点 ID（用于可视化/调试）
    /// - `name`: 节点名称
    /// - `raw_node`: 节点类型（包含 value/grad）
    /// - `parents`: 父节点列表（强引用）
    #[allow(private_interfaces)]
    pub fn new(
        id: NodeId,
        name: Option<String>,
        raw_node: NodeType,
        parents: Vec<Rc<NodeInner>>,
    ) -> Self {
        Self {
            id,
            name,
            raw_node: RefCell::new(raw_node),
            last_forward_pass_id: Cell::new(0),
            last_backward_pass_id: Cell::new(0),
            is_detached: Cell::new(false),
            parents,
            node_group_tag: RefCell::new(None),
            origin_onnx_nodes: RefCell::new(Vec::new()),
        }
    }

    /// 创建叶子节点（无父节点）
    #[allow(private_interfaces)]
    pub fn new_leaf(id: NodeId, name: Option<String>, raw_node: NodeType) -> Self {
        Self::new(id, name, raw_node, vec![])
    }

    // ==================== 基本访问器 ====================

    /// 获取节点 ID
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// 获取节点名称
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// 获取父节点列表
    pub fn parents(&self) -> &[Rc<NodeInner>] {
        &self.parents
    }

    /// 是否是叶子节点（无父节点）
    pub fn is_leaf(&self) -> bool {
        self.parents.is_empty()
    }

    // ==================== 前向/反向传播标记 ====================

    /// 获取最后一次前向传播 ID
    pub fn last_forward_pass_id(&self) -> u64 {
        self.last_forward_pass_id.get()
    }

    /// 设置最后一次前向传播 ID
    pub fn set_last_forward_pass_id(&self, id: u64) {
        self.last_forward_pass_id.set(id);
    }

    /// 获取最后一次反向传播 ID
    pub fn last_backward_pass_id(&self) -> u64 {
        self.last_backward_pass_id.get()
    }

    /// 设置最后一次反向传播 ID
    pub fn set_last_backward_pass_id(&self, id: u64) {
        self.last_backward_pass_id.set(id);
    }

    // ==================== Detach 状态 ====================

    /// 是否被 detach（梯度截断）
    pub fn is_detached(&self) -> bool {
        self.is_detached.get()
    }

    /// 设置 detach 状态
    pub fn set_detached(&self, detached: bool) {
        self.is_detached.set(detached);
    }

    // ==================== 节点分组标签 ====================

    /// 获取节点分组标签的克隆（通过 RefCell 借用）
    pub fn node_group_tag(&self) -> Option<NodeGroupTag> {
        self.node_group_tag.borrow().clone()
    }

    /// 设置节点分组标签（支持通过 &self 后补标签，如 Layer 的 Parameter 节点）
    pub fn set_node_group_tag(&self, tag: Option<NodeGroupTag>) {
        *self.node_group_tag.borrow_mut() = tag;
    }

    // ==================== ONNX 来源追溯（provenance）====================

    /// 获取 ONNX 来源节点名列表的克隆
    ///
    /// 演化、Layer 等非 ONNX 路径下返回空 `Vec`。
    pub fn origin_onnx_nodes(&self) -> Vec<String> {
        self.origin_onnx_nodes.borrow().clone()
    }

    /// 设置 ONNX 来源节点名（由 `descriptor_rebuild::rebuild_node` 在创建完
    /// NodeInner 后从 NodeDescriptor 注入）。
    pub fn set_origin_onnx_nodes(&self, names: Vec<String>) {
        *self.origin_onnx_nodes.borrow_mut() = names;
    }

    // ==================== 值和梯度访问 ====================

    /// 通过闭包访问节点的值（避免 clone，推荐用于只读场景）
    ///
    /// # 示例
    /// ```ignore
    /// let has_value = node.with_value(|v| v.is_some());
    /// let shape = node.with_value(|v| v.map(|t| t.shape().to_vec()));
    /// ```
    pub fn with_value<F, R>(&self, f: F) -> R
    where
        F: FnOnce(Option<&Tensor>) -> R,
    {
        f(self.raw_node.borrow().value())
    }

    /// 通过闭包访问节点的梯度（避免 clone，推荐用于只读场景）
    pub fn with_grad<F, R>(&self, f: F) -> R
    where
        F: FnOnce(Option<&Tensor>) -> R,
    {
        f(self.raw_node.borrow().grad())
    }

    /// 获取节点的值（clone 版本，用于需要 owned 值的场景）
    ///
    /// 注意：此方法会 clone Tensor，如果只需要读取，请使用 `with_value()`
    pub fn value(&self) -> Option<Tensor> {
        self.raw_node.borrow().value().cloned()
    }

    /// 设置节点的值
    pub fn set_value(&self, value: Option<&Tensor>) -> Result<(), GraphError> {
        self.raw_node.borrow_mut().set_value(value)
    }

    /// 设置节点的值（move 语义，零拷贝）
    ///
    /// 优化器更新参数时使用，避免 `set_value(Some(&val))` 的 clone 开销。
    pub fn set_value_owned(&self, value: Tensor) -> Result<(), GraphError> {
        self.raw_node.borrow_mut().set_value_owned(value)
    }

    /// 获取节点的梯度（clone 版本，用于需要 owned 值的场景）
    ///
    /// 注意：此方法会 clone Tensor，如果只需要读取，请使用 `with_grad()`
    pub fn grad(&self) -> Option<Tensor> {
        self.raw_node.borrow().grad().cloned()
    }

    /// 设置节点的梯度
    pub fn set_grad(&self, grad: Option<&Tensor>) -> Result<(), GraphError> {
        self.raw_node.borrow_mut().set_grad(grad)
    }

    /// 累加梯度（如果已有梯度则相加，否则直接设置）
    pub fn accumulate_grad(&self, grad: &Tensor) -> Result<(), GraphError> {
        let mut raw = self.raw_node.borrow_mut();
        // 对于不支持设置梯度的节点（如 BasicInput），静默跳过
        // 这是预期行为：输入数据不需要梯度
        let result = if raw.grad().is_some() {
            // 已有梯度：原地累加（避免创建临时 Tensor）
            raw.accumulate_grad_inplace(grad)
        } else {
            raw.set_grad(Some(grad))
        };
        match result {
            Ok(()) => Ok(()),
            Err(GraphError::InvalidOperation(msg)) if msg.contains("不支持设置梯度") => {
                // 静默跳过不支持梯度的节点
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// 累加取反梯度：已有梯度时 -= delta（零分配），首次时 set_grad(-delta)
    ///
    /// 用于 `GradResult::Negated`，避免先分配 `-upstream_grad` 再累加。
    pub fn accumulate_grad_negated(&self, delta: &Tensor) -> Result<(), GraphError> {
        let mut raw = self.raw_node.borrow_mut();
        let result = if let Some(existing) = raw.grad_mut() {
            // 快速路径：原地 -= 避免临时 Tensor
            *existing -= delta;
            Ok(())
        } else {
            // 首次设置：需要分配一次（不可避免）
            let negated = -delta;
            raw.set_grad(Some(&negated))
        };
        match result {
            Ok(()) => Ok(()),
            Err(GraphError::InvalidOperation(msg)) if msg.contains("不支持设置梯度") => {
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// 清除梯度
    pub fn clear_grad(&self) -> Result<(), GraphError> {
        self.raw_node.borrow_mut().clear_grad()
    }

    // ==================== 形状查询 ====================

    /// 获取节点的期望输出形状（静态形状）
    pub fn value_expected_shape(&self) -> Vec<usize> {
        self.raw_node.borrow().value_expected_shape().to_vec()
    }

    /// 获取节点的动态期望形状（支持动态维度）
    pub fn dynamic_expected_shape(&self) -> crate::nn::shape::DynamicShape {
        self.raw_node.borrow().dynamic_expected_shape()
    }

    // ==================== 前向传播 ====================

    /// 从父节点计算当前节点的值
    ///
    /// 收集父节点的值，调用 `raw_node.calc_value_by_parents()`
    ///
    /// # 性能优化
    /// 直接借用父节点的 raw_node，避免通过 `value()` 方法 clone Tensor
    fn calc_value_from_parents(&self, is_training: bool) -> Result<(), GraphError> {
        // 1. 先收集错误信息所需的元数据（在借用 raw_node 之前）
        let self_type_name = self.type_name();
        let parent_info: Vec<(String, NodeId)> = self
            .parents
            .iter()
            .map(|p| (p.type_name(), p.id()))
            .collect();

        // 2. 借用所有父节点的 raw_node（保持 borrow 存活直到计算完成）
        let parent_borrows: Vec<std::cell::Ref<NodeType>> =
            self.parents.iter().map(|p| p.raw_node.borrow()).collect();

        // 3. 从 borrow 中提取值引用（零 clone！）
        let parent_values: Result<Vec<&Tensor>, GraphError> = parent_borrows
            .iter()
            .enumerate()
            .map(|(i, b)| {
                b.value().ok_or_else(|| {
                    let (p_type, p_id) = &parent_info[i];
                    GraphError::ComputationError(format!(
                        "{}[{}] 的父节点 {}[{}] 没有值",
                        self_type_name, self.id, p_type, p_id
                    ))
                })
            })
            .collect();
        let parent_values = parent_values?;

        // 4. 设置训练模式并计算
        // 注意：self.raw_node 与 parent.raw_node 是不同的 RefCell，可以同时借用
        let mut raw = self.raw_node.borrow_mut();
        raw.set_training_mode(is_training);
        raw.calc_value_by_parents(&parent_values)
    }

    /// 递归前向传播
    ///
    /// # 参数
    /// - `pass_id`: 当前前向传播的 ID（用于避免重复计算）
    /// - `is_training`: 是否训练模式
    ///
    /// # 逻辑
    /// 1. 如果 `last_forward_pass_id == pass_id`，跳过（已计算）
    /// 2. 递归确保所有父节点已计算
    /// 3. 叶子节点检查值是否已设置
    /// 4. 非叶子节点调用 `calc_value_from_parents`
    /// 5. 更新 `last_forward_pass_id`
    pub fn forward_recursive(&self, pass_id: u64, is_training: bool) -> Result<(), GraphError> {
        // 1. 检查是否已计算
        if self.last_forward_pass_id.get() == pass_id {
            return Ok(());
        }

        // 2. 递归计算父节点
        for parent in &self.parents {
            parent.forward_recursive(pass_id, is_training)?;
        }

        // 3. 计算当前节点
        if self.is_leaf() {
            // 叶子节点：检查值是否已设置（使用 with_value 避免 clone）
            let has_value = self.with_value(|v| v.is_some());
            if !has_value {
                return Err(GraphError::ComputationError(format!(
                    "叶子节点 {}[{}] 没有值，请先设置",
                    self.type_name(),
                    self.id
                )));
            }
        } else {
            // 非叶子节点：从父节点计算
            self.calc_value_from_parents(is_training)?;
        }

        // 4. 更新 pass_id
        self.last_forward_pass_id.set(pass_id);
        Ok(())
    }

    // ==================== 反向传播 ====================

    /// 计算对指定父节点的梯度
    ///
    /// 封装对 `raw_node.calc_grad_to_parent()` 的调用
    ///
    /// # 参数
    /// - `target_index`: 目标父节点的索引
    /// - `upstream_grad`: 上游梯度（从子节点传来）
    ///
    /// # 返回
    /// `GradResult` 枚举，调用方根据变体选择零拷贝或分配策略
    ///
    /// # 性能优化
    /// 直接借用父节点的 raw_node，避免 clone Tensor
    pub(in crate::nn) fn calc_grad_to_parent_index(
        &self,
        target_index: usize,
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        // 1. 借用所有父节点的 raw_node
        let parent_borrows: Vec<std::cell::Ref<NodeType>> =
            self.parents.iter().map(|p| p.raw_node.borrow()).collect();

        // 2. 从 borrow 中提取值引用
        let parent_values: Vec<&Tensor> = parent_borrows.iter().filter_map(|b| b.value()).collect();

        // 3. 调用 raw_node 的梯度计算
        self.raw_node
            .borrow()
            .calc_grad_to_parent(target_index, &parent_values, upstream_grad)
    }

    /// 向父节点传播梯度
    ///
    /// 遍历所有父节点，计算梯度并累加到父节点
    ///
    /// # 参数
    /// - `upstream_grad`: 上游梯度（当前节点收到的梯度）
    ///
    /// # 逻辑
    /// 1. 如果当前节点是 detached，跳过（不向上游传播梯度）
    /// 2. 如果是叶子节点，跳过（无父节点）
    /// 3. 遍历父节点，计算梯度并累加
    ///
    /// # detach 语义
    /// - detached 节点可以**收到**梯度（作为其他节点的父节点）
    /// - 但 detached 节点**不会向上游**传播梯度（在此方法开头检查）
    pub fn propagate_grad_to_parents(&self, upstream_grad: &Tensor) -> Result<(), GraphError> {
        // 1. 检查 detach 状态：detached 节点不向上游传播梯度
        if self.is_detached() {
            return Ok(());
        }

        // 2. 叶子节点无需传播
        if self.is_leaf() {
            return Ok(());
        }

        // 3. 遍历父节点，计算并累加梯度
        // 注意：不检查父节点的 detach 状态，因为：
        // - 父节点可以收到梯度
        // - 当到达父节点执行 propagate_grad_to_parents 时，它会检查自己的 detach 状态
        for (i, parent) in self.parents.iter().enumerate() {
            // 计算对该父节点的梯度
            let result = match self.calc_grad_to_parent_index(i, upstream_grad) {
                Ok(r) => r,
                Err(GraphError::InvalidOperation(msg))
                    if msg.contains("不应该")
                        || msg.contains("不需要")
                        || msg.contains("不支持") =>
                {
                    // 某些节点设计上不需要计算对特定父节点的梯度
                    // 例如：labels 不需要梯度、某些常量节点等
                    continue;
                }
                Err(e) => return Err(e),
            };

            // 根据 GradResult 类型选择累加策略
            match result {
                GradResult::PassThrough => {
                    // 零拷贝：直接用 upstream_grad 累加
                    parent.accumulate_grad(upstream_grad)?;
                }
                GradResult::Negated => {
                    // 零分配累加：已有梯度时 -= upstream_grad
                    parent.accumulate_grad_negated(upstream_grad)?;
                }
                GradResult::Computed(grad) => {
                    parent.accumulate_grad(&grad)?;
                }
            }
        }

        Ok(())
    }

    /// 收集反向传播的拓扑顺序
    ///
    /// 从当前节点（通常是 loss）开始，DFS 遍历所有祖先节点，
    /// 返回拓扑逆序的节点列表（先子节点后父节点）。
    ///
    /// # 实现
    /// 使用**后序 DFS + 反转**确保正确的反向传播顺序：
    /// - 后序 DFS：先递归访问所有父节点（输入），再压入当前节点
    ///   → 产生正向拓扑序（输入在前，输出在后）
    /// - 反转：得到反向拓扑序（loss 在前，输入在后）
    ///
    /// 这保证了当处理某个节点时，所有消费该节点输出的下游节点
    /// 都已将梯度传播回来，避免中间节点因梯度不完整而传播错误梯度。
    ///
    /// # 参数
    /// - `self`: 起始节点的 Rc 引用（需要 &Rc<Self> 以便 clone）
    ///
    /// # 返回
    /// 拓扑逆序的节点列表，用于反向传播遍历
    pub fn backward_topo_order(self: &Rc<Self>) -> Vec<Rc<NodeInner>> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();

        fn dfs(
            node: &Rc<NodeInner>,
            visited: &mut HashSet<NodeId>,
            result: &mut Vec<Rc<NodeInner>>,
        ) {
            if visited.contains(&node.id()) {
                return;
            }
            visited.insert(node.id());

            // 后序：先递归访问所有父节点
            for parent in node.parents() {
                dfs(parent, visited, result);
            }

            // 再压入当前节点（保证当前节点排在所有父节点之后）
            result.push(node.clone());
        }

        dfs(self, &mut visited, &mut result);

        // 反转：从正向拓扑序 → 反向拓扑序（loss 在前，输入在后）
        result.reverse();
        result
    }

    /// 执行完整的反向传播
    ///
    /// 从当前节点（loss）开始，按拓扑逆序遍历所有节点，
    /// 将梯度传播到各父节点。
    ///
    /// # 参数
    /// - `pass_id`: 反向传播 ID（用于避免重复处理）
    ///
    /// # 前置条件
    /// - 当前节点（loss）的梯度已设置（通常为 1.0）
    ///
    /// # 逻辑
    /// 1. 收集拓扑顺序
    /// 2. 遍历每个节点：
    ///    - 跳过已处理的节点（pass_id 检查）
    ///    - 获取节点梯度，调用 `propagate_grad_to_parents`
    ///    - 更新 `last_backward_pass_id`
    pub fn backward_propagate(self: &Rc<Self>, pass_id: u64) -> Result<(), GraphError> {
        let topo_order = self.backward_topo_order();

        for node in &topo_order {
            // 跳过已处理的节点
            if node.last_backward_pass_id() == pass_id {
                continue;
            }

            // 获取当前节点的梯度
            let grad = match node.grad() {
                Some(g) => g,
                None => continue, // 没有梯度的节点跳过
            };

            // 向父节点传播梯度
            node.propagate_grad_to_parents(&grad)?;

            // 更新 pass_id
            node.set_last_backward_pass_id(pass_id);
        }

        Ok(())
    }

    // ==================== 节点类型信息 ====================

    /// 获取节点类型的只读引用
    #[allow(private_interfaces, private_bounds)]
    pub fn with_raw_node<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&NodeType) -> R,
    {
        f(&self.raw_node.borrow())
    }

    /// 获取节点类型的可变引用
    #[allow(private_interfaces, private_bounds)]
    pub fn with_raw_node_mut<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut NodeType) -> R,
    {
        f(&mut self.raw_node.borrow_mut())
    }

    /// 判断节点是否是参数节点
    pub fn is_parameter(&self) -> bool {
        matches!(&*self.raw_node.borrow(), NodeType::Parameter(_))
    }

    /// 判断节点是否是输入节点
    pub fn is_input(&self) -> bool {
        matches!(&*self.raw_node.borrow(), NodeType::Input(_))
    }

    /// 获取节点的期望形状
    pub fn shape(&self) -> Vec<usize> {
        self.raw_node.borrow().value_expected_shape().to_vec()
    }

    /// 获取节点的动态形状
    pub fn dynamic_shape(&self) -> crate::nn::shape::DynamicShape {
        self.raw_node.borrow().dynamic_expected_shape()
    }

    /// 获取节点类型的显示名称
    pub fn type_name(&self) -> String {
        self.raw_node.borrow().get_type_name().to_string()
    }
}

impl std::fmt::Debug for NodeInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeInner")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("type", &self.type_name())
            .field("parents_count", &self.parents.len())
            .field("is_detached", &self.is_detached.get())
            .finish()
    }
}
