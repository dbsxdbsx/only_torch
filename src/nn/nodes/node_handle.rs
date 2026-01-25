use super::super::graph::GraphError;
use super::raw_node::{
    Add, AvgPool2d, Conv2d, Divide, Flatten, Identity, InputVariant, LeakyReLU, MSELoss, MatMul,
    MaxPool2d, Multiply, Parameter, Reduction, Reshape, Select, Sigmoid, Sign, SoftPlus, Softmax,
    SoftmaxCrossEntropy, State, Step, Subtract, Tanh, ZerosLike,
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

    /// 创建基本输入节点（Data 变体）
    pub(in crate::nn) fn new_basic_input(shape: &[usize]) -> Result<Self, GraphError> {
        let input = InputVariant::new_data(shape)?;
        Ok(Self {
            raw_node: NodeType::Input(input),
            last_forward_pass_id: 0,
            last_backward_pass_id: 0,
            is_detached: false,
        })
    }

    /// 创建目标输入节点（Target 变体，用于 Loss 的目标值）
    pub(in crate::nn) fn new_target_input(shape: &[usize]) -> Self {
        let input = InputVariant::new_target(shape);
        Self {
            raw_node: NodeType::Input(input),
            last_forward_pass_id: 0,
            last_backward_pass_id: 0,
            is_detached: false,
        }
    }

    pub(in crate::nn) fn new_parameter(shape: &[usize]) -> Result<Self, GraphError> {
        let parameter = Parameter::new(shape)?;
        Ok(Self {
            raw_node: NodeType::Parameter(parameter),
            last_forward_pass_id: 0,
            last_backward_pass_id: 0,
            is_detached: false,
        })
    }

    pub(in crate::nn) fn new_parameter_seeded(
        shape: &[usize],
        seed: u64,
    ) -> Result<Self, GraphError> {
        let parameter = Parameter::new_seeded(shape, seed)?;
        Ok(Self {
            raw_node: NodeType::Parameter(parameter),
            last_forward_pass_id: 0,
            last_backward_pass_id: 0,
            is_detached: false,
        })
    }

    /// 创建 State 节点（用于 RNN 时间状态）
    pub(in crate::nn) fn new_state(shape: &[usize]) -> Result<Self, GraphError> {
        let state = State::new(shape)?;
        Ok(Self {
            raw_node: NodeType::State(state),
            last_forward_pass_id: 0,
            last_backward_pass_id: 0,
            is_detached: false,
        })
    }

    /// 创建 `ZerosLike` 节点（动态零张量，用于 RNN 初始隐藏状态）
    pub(in crate::nn) fn new_zeros_like(feature_shape: &[usize]) -> Self {
        let zeros_like = ZerosLike::new(feature_shape);
        Self {
            raw_node: NodeType::ZerosLike(zeros_like),
            last_forward_pass_id: 0,
            last_backward_pass_id: 0,
            is_detached: false,
        }
    }

    pub(in crate::nn) fn new_add(parents: &[&Self]) -> Result<Self, GraphError> {
        Self::new(Add::new(parents)?)
    }

    /// 创建 Conv2d 节点
    ///
    /// # 参数
    /// - `parents`: [输入节点, 卷积核节点]
    /// - `stride`: 步长 (sH, sW)
    /// - `padding`: 填充 (pH, pW)
    pub(in crate::nn) fn new_conv2d(
        parents: &[&Self],
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self, GraphError> {
        Self::new(Conv2d::new(parents, stride, padding)?)
    }

    /// 创建 `MaxPool2d` 节点
    ///
    /// # 参数
    /// - `parents`: [输入节点]
    /// - `kernel_size`: 池化窗口大小 (kH, kW)
    /// - `stride`: 步长 (sH, sW)，None 时默认等于 `kernel_size`
    pub(in crate::nn) fn new_max_pool2d(
        parents: &[&Self],
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
    ) -> Result<Self, GraphError> {
        Self::new(MaxPool2d::new(parents, kernel_size, stride)?)
    }

    /// 创建 `AvgPool2d` 节点
    ///
    /// # 参数
    /// - `parents`: [输入节点]
    /// - `kernel_size`: 池化窗口大小 (kH, kW)
    /// - `stride`: 步长 (sH, sW)，None 时默认等于 `kernel_size`
    pub(in crate::nn) fn new_avg_pool2d(
        parents: &[&Self],
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
    ) -> Result<Self, GraphError> {
        Self::new(AvgPool2d::new(parents, kernel_size, stride)?)
    }

    pub(in crate::nn) fn new_mat_mul(parents: &[&Self]) -> Result<Self, GraphError> {
        Self::new(MatMul::new(parents)?)
    }

    pub(in crate::nn) fn new_multiply(parents: &[&Self]) -> Result<Self, GraphError> {
        Self::new(Multiply::new(parents)?)
    }

    pub(in crate::nn) fn new_divide(parents: &[&Self]) -> Result<Self, GraphError> {
        Self::new(Divide::new(parents)?)
    }

    pub(in crate::nn) fn new_subtract(parents: &[&Self]) -> Result<Self, GraphError> {
        Self::new(Subtract::new(parents)?)
    }

    pub(in crate::nn) fn new_flatten(
        parents: &[&Self],
        keep_first_dim: bool,
    ) -> Result<Self, GraphError> {
        Self::new(Flatten::new(parents, keep_first_dim)?)
    }

    pub(in crate::nn) fn new_reshape(
        parents: &[&Self],
        target_shape: &[usize],
    ) -> Result<Self, GraphError> {
        Self::new(Reshape::new(parents, target_shape)?)
    }

    pub(in crate::nn) fn new_sign(parents: &[&Self]) -> Result<Self, GraphError> {
        Self::new(Sign::new(parents)?)
    }

    pub(in crate::nn) fn new_step(parents: &[&Self]) -> Result<Self, GraphError> {
        Self::new(Step::new(parents)?)
    }

    pub(in crate::nn) fn new_tanh(parents: &[&Self]) -> Result<Self, GraphError> {
        Self::new(Tanh::new(parents)?)
    }

    pub(in crate::nn) fn new_sigmoid(parents: &[&Self]) -> Result<Self, GraphError> {
        Self::new(Sigmoid::new(parents)?)
    }

    pub(in crate::nn) fn new_softmax(parents: &[&Self]) -> Result<Self, GraphError> {
        Self::new(Softmax::new(parents)?)
    }

    pub(in crate::nn) fn new_identity(parents: &[&Self]) -> Result<Self, GraphError> {
        Self::new(Identity::new(parents)?)
    }

    /// 创建 SmartInput 节点（智能输入）
    ///
    /// SmartInput 用于 ModelState 的智能缓存机制，支持：
    /// - 动态设置 detached 状态
    /// - 梯度路由到外部目标节点
    /// - 动态 batch
    pub(in crate::nn) fn new_smart_input(shape: &[usize]) -> Result<Self, GraphError> {
        Ok(Self {
            raw_node: NodeType::Input(InputVariant::new_smart(shape)),
            last_forward_pass_id: 0,
            last_backward_pass_id: 0,
            is_detached: false,
        })
    }

    /// 创建 RecurrentOutput 节点（循环层输出桥接）
    ///
    /// RecurrentOutput 用于 RNN/LSTM/GRU 层的输出桥接，功能与 SmartInput 相同：
    /// - 固定的 node_id，使下游层可以复用
    /// - 支持梯度路由到实际的 RNN 输出节点
    /// - 支持动态 batch
    pub(in crate::nn) fn new_recurrent_output(shape: &[usize]) -> Result<Self, GraphError> {
        Ok(Self {
            raw_node: NodeType::Input(InputVariant::new_recurrent_output(shape)),
            last_forward_pass_id: 0,
            last_backward_pass_id: 0,
            is_detached: false,
        })
    }

    /// 设置 SmartInput/RecurrentOutput 的 detached 状态
    ///
    /// # 参数
    /// - `detached`: 是否阻止梯度传播
    /// - `mark_ever_detached`: 是否标记 was_ever_detached（用于可视化显示虚线边框）
    ///
    /// # 返回
    /// 如果节点不是 SmartInput/RecurrentOutput，返回错误
    pub(in crate::nn) fn set_router_detached(
        &mut self,
        detached: bool,
        mark_ever_detached: bool,
    ) -> Result<(), GraphError> {
        if let NodeType::Input(InputVariant::Smart(smart) | InputVariant::RecurrentOutput(smart)) =
            &self.raw_node
        {
            smart.set_detached(detached, mark_ever_detached);
            Ok(())
        } else {
            Err(GraphError::InvalidOperation(format!(
                "{} 不是 SmartInput/RecurrentOutput 节点",
                self.raw_node.display_node()
            )))
        }
    }

    /// 设置 SmartInput/RecurrentOutput 的梯度路由目标
    ///
    /// # 返回
    /// 如果节点不是 SmartInput/RecurrentOutput，返回错误
    pub(in crate::nn) fn set_gradient_target(
        &mut self,
        target: Option<NodeId>,
    ) -> Result<(), GraphError> {
        if let NodeType::Input(InputVariant::Smart(smart) | InputVariant::RecurrentOutput(smart)) =
            &self.raw_node
        {
            smart.set_gradient_target(target);
            Ok(())
        } else {
            Err(GraphError::InvalidOperation(format!(
                "{} 不是 SmartInput/RecurrentOutput 节点",
                self.raw_node.display_node()
            )))
        }
    }

    /// 获取 SmartInput/RecurrentOutput 的梯度路由目标
    pub(in crate::nn) fn gradient_target(&self) -> Option<NodeId> {
        if let NodeType::Input(InputVariant::Smart(smart) | InputVariant::RecurrentOutput(smart)) =
            &self.raw_node
        {
            smart.gradient_target()
        } else {
            None
        }
    }

    pub(in crate::nn) fn new_leaky_relu(
        parents: &[&Self],
        negative_slope: f64,
    ) -> Result<Self, GraphError> {
        Self::new(LeakyReLU::new(parents, negative_slope)?)
    }

    pub(in crate::nn) fn new_softplus(parents: &[&Self]) -> Result<Self, GraphError> {
        Self::new(SoftPlus::new(parents)?)
    }

    /// 创建 Select 节点（从张量中选择指定轴和索引的切片）
    ///
    /// # 参数
    /// - `parents`: [输入节点]
    /// - `axis`: 选择的轴
    /// - `index`: 选择的索引
    pub(in crate::nn) fn new_select(
        parents: &[&Self],
        axis: usize,
        index: usize,
    ) -> Result<Self, GraphError> {
        Self::new(Select::new(parents, axis, index)?)
    }

    pub(in crate::nn) fn new_softmax_cross_entropy(parents: &[&Self]) -> Result<Self, GraphError> {
        Self::new(SoftmaxCrossEntropy::new(parents)?)
    }

    /// 创建 `MSELoss` 节点（默认使用 Mean reduction）
    pub(in crate::nn) fn new_mse_loss(parents: &[&Self]) -> Result<Self, GraphError> {
        Self::new(MSELoss::new_mean(parents)?)
    }

    /// 创建 `MSELoss` 节点（指定 reduction 模式）
    pub(in crate::nn) fn new_mse_loss_with_reduction(
        parents: &[&Self],
        reduction: Reduction,
    ) -> Result<Self, GraphError> {
        Self::new(MSELoss::new(parents, reduction)?)
    }

    pub(in crate::nn) fn calc_value_by_parents(
        &mut self,
        parents: &[Self],
    ) -> Result<(), GraphError> {
        self.raw_node.calc_value_by_parents(parents)
    }

    // ========== 梯度（VJP 模式）==========

    /// 计算本节点对父节点的梯度（Batch 模式）
    pub(in crate::nn) fn calc_grad_to_parent(
        &self,
        parent: &Self,
        upstream_grad: &Tensor,
        assistant_parent: Option<&Self>,
    ) -> Result<Tensor, GraphError> {
        self.raw_node
            .calc_grad_to_parent(parent, upstream_grad, assistant_parent)
    }

    /// 获取节点的梯度（Batch 模式）
    pub(in crate::nn) fn grad(&self) -> Option<&Tensor> {
        self.raw_node.grad()
    }

    /// 设置节点的梯度（Batch 模式）
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
    /// 注意：对于 SmartInput 节点，应使用 `set_router_detached()` 方法。
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
