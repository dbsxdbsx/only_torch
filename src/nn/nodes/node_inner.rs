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

use super::raw_node::TraitNode;
use super::NodeType;
use crate::nn::graph::GraphError;
use crate::nn::NodeId;
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

    /// 获取节点的值
    pub fn value(&self) -> Option<Tensor> {
        self.raw_node.borrow().value().cloned()
    }

    /// 设置节点的值
    pub fn set_value(&self, value: Option<&Tensor>) -> Result<(), GraphError> {
        self.raw_node.borrow_mut().set_value(value)
    }

    /// 获取节点的梯度
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
