/*
 * @Author       : 老董
 * @Date         : 2026-02-01
 * @Description  : NodeInner - 方案 C 的核心节点结构
 *
 * 这是动态图生命周期管理的核心组件：
 * - 被 Rc 包装，通过引用计数控制生命周期
 * - 持有 parents 强引用，保证反向传播时上游存活
 * - 当 Var 离开作用域且无其他引用时，节点自动释放
 */

use super::NodeType;
use super::raw_node::TraitNode;
use crate::nn::NodeId;
use crate::nn::graph::GraphError;
use crate::tensor::Tensor;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

/// 节点内部结构 - 方案 C 的核心
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
}

impl NodeInner {
    /// 创建新的 NodeInner
    ///
    /// # 参数
    /// - `id`: 节点 ID（用于可视化/调试）
    /// - `name`: 节点名称
    /// - `raw_node`: 节点类型（包含 value/grad）
    /// - `parents`: 父节点列表（强引用）
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
        }
    }

    /// 创建叶子节点（无父节点）
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
        if let Some(existing) = raw.grad() {
            let new_grad = existing + grad;
            raw.set_grad(Some(&new_grad))
        } else {
            raw.set_grad(Some(grad))
        }
    }

    /// 清除梯度
    pub fn clear_grad(&self) -> Result<(), GraphError> {
        self.raw_node.borrow_mut().clear_grad()
    }

    // ==================== 前向传播（方案 C）====================

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
        let parent_borrows: Vec<std::cell::Ref<NodeType>> = self
            .parents
            .iter()
            .map(|p| p.raw_node.borrow())
            .collect();

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

    // ==================== 反向传播（方案 C）====================

    /// 计算对指定父节点的梯度
    ///
    /// 封装对 `raw_node.calc_grad_to_parent()` 的调用
    ///
    /// # 参数
    /// - `target_index`: 目标父节点的索引
    /// - `upstream_grad`: 上游梯度（从子节点传来）
    ///
    /// # 性能优化
    /// 直接借用父节点的 raw_node，避免 clone Tensor
    fn calc_grad_to_parent_index(
        &self,
        target_index: usize,
        upstream_grad: &Tensor,
    ) -> Result<Tensor, GraphError> {
        // 1. 借用所有父节点的 raw_node
        let parent_borrows: Vec<std::cell::Ref<NodeType>> = self
            .parents
            .iter()
            .map(|p| p.raw_node.borrow())
            .collect();

        // 2. 从 borrow 中提取值引用
        let parent_values: Vec<&Tensor> = parent_borrows
            .iter()
            .filter_map(|b| b.value())
            .collect();

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
    /// 1. 如果当前节点是 detached，跳过（不传播梯度）
    /// 2. 如果是叶子节点，跳过（无父节点）
    /// 3. 遍历父节点，计算梯度并累加
    /// 4. 跳过 detached 的父节点
    pub fn propagate_grad_to_parents(&self, upstream_grad: &Tensor) -> Result<(), GraphError> {
        // 1. 检查 detach 状态
        if self.is_detached() {
            return Ok(());
        }

        // 2. 叶子节点无需传播
        if self.is_leaf() {
            return Ok(());
        }

        // 3. 遍历父节点，计算并累加梯度
        for (i, parent) in self.parents.iter().enumerate() {
            // 跳过 detached 的父节点
            if parent.is_detached() {
                continue;
            }

            // 计算对该父节点的梯度
            let grad = self.calc_grad_to_parent_index(i, upstream_grad)?;

            // 累加到父节点
            parent.accumulate_grad(&grad)?;
        }

        Ok(())
    }

    // ==================== 节点类型信息 ====================

    /// 获取节点类型的只读引用
    pub fn with_raw_node<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&NodeType) -> R,
    {
        f(&self.raw_node.borrow())
    }

    /// 获取节点类型的可变引用
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
