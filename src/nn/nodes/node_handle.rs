use super::super::graph::GraphError;
use super::raw_node::{
    Abs, Add, Amax, Amin, AvgPool2d, BCE, Conv2d, Divide, Dropout, Flatten, Gather, Huber,
    Identity, InputVariant, LeakyReLU, Ln, LogSoftmax, MAE, MSE, MatMul, MaxPool2d, Maximum,
    Mean, Minimum, Multiply, Parameter, Reduction, Reshape, Select, Sigmoid, Sign, SoftPlus,
    Softmax, SoftmaxCrossEntropy, Stack, State, Step, Subtract, Sum, Tanh, ZerosLike,
};
use super::{NodeType, TraitNode};
use crate::tensor::Tensor;

/// 为Graph所管理的节点提供一个句柄，所有内置数据类型必须为2维张量
#[derive(Clone)]
pub(in crate::nn) struct NodeHandle {
    raw_node: NodeType,
    /// 节点最后一次计算的前向传播次数
    last_forward_pass_id: u64,
    /// 节点最后一次计算的反向传播次数
    last_backward_pass_id: u64,
    /// 是否被 detach（梯度截断标记）
    /// 若为 true，反向传播时不会向该节点的父节点传播梯度
    is_detached: bool,
}

impl std::fmt::Display for NodeHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.raw_node.display_node())
    }
}

impl NodeHandle {
    pub(in crate::nn) fn new<T: Into<NodeType>>(raw_node: T) -> Result<Self, GraphError> {
        let raw_node = raw_node.into();
        Ok(Self {
            raw_node,
            last_forward_pass_id: 0,
            last_backward_pass_id: 0,
            is_detached: false,
        })
    }

    pub(in crate::nn) const fn node_type(&self) -> &NodeType {
        &self.raw_node
    }

    /// 检查新张量形状是否与节点的动态形状兼容
    ///
    /// 使用 `dynamic_expected_shape()` 的 `is_compatible_with_tensor` 方法，
    /// 自动处理动态维度（None）和固定维度的兼容性检查。
    fn check_shape_consistency(&self, new_tensor: &Tensor) -> Result<(), GraphError> {
        let dyn_shape = self.raw_node.dynamic_expected_shape();
        let actual = new_tensor.shape();

        // 使用 DynamicShape 的内置兼容性检查
        // - 维度数必须相同
        // - 动态维度（None）与任何值兼容
        // - 固定维度必须完全匹配
        if !dyn_shape.is_compatible_with_tensor(actual) {
            let expected_fixed = self.raw_node.value_expected_shape();
            return Err(GraphError::ShapeMismatch {
                expected: expected_fixed.to_vec(),
                got: actual.to_vec(),
                message: format!(
                    "新张量的形状 {:?} 与节点 '{}' 的动态形状 {} 不兼容。",
                    actual,
                    self.name(),
                    dyn_shape,
                ),
            });
        }

        Ok(())
    }

    pub(in crate::nn) fn bind_id_and_name(&mut self, id: NodeId, name: &str) {
        self.raw_node.set_id(id);
        self.raw_node.set_name(name);
    }

    pub(in crate::nn) fn id(&self) -> NodeId {
        self.raw_node.id()
    }

    pub(in crate::nn) fn name(&self) -> &str {
        self.raw_node.name()
    }

    pub(in crate::nn) fn is_inited(&self) -> bool {
        self.raw_node.is_inited()
    }

    pub(in crate::nn) fn value(&self) -> Option<&Tensor> {
        self.raw_node.value()
    }

    pub(in crate::nn) fn set_value(&mut self, value: Option<&Tensor>) -> Result<(), GraphError> {
        if let Some(tensor) = value {
            self.check_shape_consistency(tensor)?;
        }
        self.raw_node.set_value(value)
    }

    /// 强制设置节点的值（绕过类型检查）
    /// 仅供内部使用（如 BPTT 快照恢复）
    pub(in crate::nn) fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.raw_node.set_value_unchecked(value);
    }

    pub(in crate::nn) fn clear_value(&mut self) -> Result<(), GraphError> {
        self.raw_node.clear_value()
    }

    /// 设置训练模式（仅 Dropout 等节点需要）
    pub(in crate::nn) fn set_training_mode(&mut self, is_training: bool) {
        self.raw_node.set_training_mode(is_training);
    }

    /// 获取节点的梯度
    pub(in crate::nn) fn grad(&self) -> Option<&Tensor> {
        self.raw_node.grad()
    }

    /// 设置节点的梯度
    pub(in crate::nn) fn set_grad(&mut self, grad: Option<&Tensor>) -> Result<(), GraphError> {
        self.raw_node.set_grad(grad)
    }

    /// 清除节点的梯度
    pub(in crate::nn) fn clear_grad(&mut self) -> Result<(), GraphError> {
        self.raw_node.clear_grad()
    }

    // ========== 通用方法 ==========

    pub(in crate::nn) fn value_expected_shape(&self) -> &[usize] {
        self.raw_node.value_expected_shape()
    }

    pub(in crate::nn) fn dynamic_expected_shape(&self) -> crate::nn::shape::DynamicShape {
        self.raw_node.dynamic_expected_shape()
    }

    pub(in crate::nn) fn supports_dynamic_batch(&self) -> bool {
        self.raw_node.supports_dynamic_batch()
    }

    pub(in crate::nn) fn has_value(&self) -> bool {
        self.raw_node.value().is_some()
    }

    pub(in crate::nn) const fn last_forward_pass_id(&self) -> u64 {
        self.last_forward_pass_id
    }

    pub(in crate::nn) const fn set_last_forward_pass_id(&mut self, forward_pass_id: u64) {
        self.last_forward_pass_id = forward_pass_id;
    }

    // 用于测试中验证 backward pass 状态
    #[allow(dead_code)]
    pub(in crate::nn) const fn last_backward_pass_id(&self) -> u64 {
        self.last_backward_pass_id
    }

    pub(in crate::nn) const fn set_last_backward_pass_id(&mut self, backward_pass_id: u64) {
        self.last_backward_pass_id = backward_pass_id;
    }

    /// 检查节点是否被 detach
    ///
    /// 若返回 true，反向传播时不会向该节点的父节点传播梯度
    /// 检查节点是否处于 detached 状态
    ///
    /// 对于 SmartInput/RecurrentOutput 节点，使用其内部动态标志；
    /// 对于其他节点，使用 `NodeHandle` 的静态标志。
    pub(in crate::nn) fn is_detached(&self) -> bool {
        // SmartInput/RecurrentOutput 有自己的动态 detached 标志
        if let NodeType::Input(InputVariant::Smart(smart) | InputVariant::RecurrentOutput(smart)) =
            &self.raw_node
        {
            return smart.is_detached();
        }
        self.is_detached
    }

    /// 设置节点的 detach 状态
    ///
    /// - `true`: 截断该节点的梯度流，反向传播时不向父节点传播
    /// - `false`: 正常传播梯度（默认状态）
    ///
    /// 注意：对于 `SmartInput` 节点，应使用 `set_router_detached()` 方法。
    pub(in crate::nn) const fn set_detached(&mut self, detached: bool) {
        self.is_detached = detached;
    }
}

/// 节点ID
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct NodeId(pub(in crate::nn) u64);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
