use super::super::graph::GraphError;
use super::raw_node::{
    Add, AvgPool2d, Conv2d, Flatten, Input, LeakyReLU, MSELoss, MatMul, MaxPool2d, Multiply,
    Parameter, PerceptionLoss, Reduction, Reshape, ScalarMultiply, Sigmoid, SoftPlus,
    SoftmaxCrossEntropy, Step, Tanh,
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

    fn check_shape_consistency(&self, new_tensor: &Tensor) -> Result<(), GraphError> {
        if let Some(current_value) = self.raw_node.value()
            && new_tensor.shape() != current_value.shape()
        {
            return Err(GraphError::ShapeMismatch {
                expected: current_value.shape().to_vec(),
                got: new_tensor.shape().to_vec(),
                message: format!(
                    "新张量的形状 {:?} 与节点 '{}' 现有张量的形状 {:?} 不匹配。",
                    new_tensor.shape(),
                    self.name(),
                    current_value.shape(),
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

    pub(in crate::nn) fn jacobi(&self) -> Option<&Tensor> {
        self.raw_node.jacobi()
    }

    pub(in crate::nn) fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError> {
        self.raw_node.set_jacobi(jacobi)
    }

    pub(in crate::nn) fn clear_jacobi(&mut self) -> Result<(), GraphError> {
        self.raw_node.clear_jacobi()
    }

    pub(in crate::nn) fn clear_value(&mut self) -> Result<(), GraphError> {
        self.raw_node.clear_value()
    }

    pub(in crate::nn) fn new_input(shape: &[usize]) -> Result<Self, GraphError> {
        let input = Input::new(shape)?;
        Ok(Self {
            raw_node: NodeType::Input(input),
            last_forward_pass_id: 0,
            last_backward_pass_id: 0,
            is_detached: false,
        })
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

    /// 创建 MaxPool2d 节点
    ///
    /// # 参数
    /// - `parents`: [输入节点]
    /// - `kernel_size`: 池化窗口大小 (kH, kW)
    /// - `stride`: 步长 (sH, sW)，None 时默认等于 kernel_size
    pub(in crate::nn) fn new_max_pool2d(
        parents: &[&Self],
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
    ) -> Result<Self, GraphError> {
        Self::new(MaxPool2d::new(parents, kernel_size, stride)?)
    }

    /// 创建 AvgPool2d 节点
    ///
    /// # 参数
    /// - `parents`: [输入节点]
    /// - `kernel_size`: 池化窗口大小 (kH, kW)
    /// - `stride`: 步长 (sH, sW)，None 时默认等于 kernel_size
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

    pub(in crate::nn) fn new_scalar_multiply(parents: &[&Self]) -> Result<Self, GraphError> {
        Self::new(ScalarMultiply::new(parents)?)
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

    pub(in crate::nn) fn new_leaky_relu(
        parents: &[&Self],
        negative_slope: f64,
    ) -> Result<Self, GraphError> {
        Self::new(LeakyReLU::new(parents, negative_slope)?)
    }

    pub(in crate::nn) fn new_softplus(parents: &[&Self]) -> Result<Self, GraphError> {
        Self::new(SoftPlus::new(parents)?)
    }

    pub(in crate::nn) fn new_perception_loss(parents: &[&Self]) -> Result<Self, GraphError> {
        Self::new(PerceptionLoss::new(parents)?)
    }

    pub(in crate::nn) fn new_softmax_cross_entropy(parents: &[&Self]) -> Result<Self, GraphError> {
        Self::new(SoftmaxCrossEntropy::new(parents)?)
    }

    /// 创建 MSELoss 节点（默认使用 Mean reduction）
    pub(in crate::nn) fn new_mse_loss(parents: &[&Self]) -> Result<Self, GraphError> {
        Self::new(MSELoss::new_mean(parents)?)
    }

    /// 创建 MSELoss 节点（指定 reduction 模式）
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

    // ========== 单样本模式（Jacobi-based）==========

    /// 计算本节点对父节点的雅可比矩阵
    pub(in crate::nn) fn calc_jacobi_to_a_parent(
        &self,
        parent: &Self,
        assistant_parent: Option<&Self>,
    ) -> Result<Tensor, GraphError> {
        self.raw_node
            .calc_jacobi_to_a_parent(parent, assistant_parent)
    }

    // ========== Batch 模式（Gradient-based）==========

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

    pub(in crate::nn) fn has_value(&self) -> bool {
        self.raw_node.value().is_some()
    }

    pub(in crate::nn) const fn last_forward_pass_id(&self) -> u64 {
        self.last_forward_pass_id
    }

    pub(in crate::nn) const fn set_last_forward_pass_id(&mut self, forward_pass_id: u64) {
        self.last_forward_pass_id = forward_pass_id;
    }

    pub(in crate::nn) const fn last_backward_pass_id(&self) -> u64 {
        self.last_backward_pass_id
    }

    pub(in crate::nn) const fn set_last_backward_pass_id(&mut self, backward_pass_id: u64) {
        self.last_backward_pass_id = backward_pass_id;
    }

    /// 检查节点是否被 detach
    ///
    /// 若返回 true，反向传播时不会向该节点的父节点传播梯度
    pub(in crate::nn) const fn is_detached(&self) -> bool {
        self.is_detached
    }

    /// 设置节点的 detach 状态
    ///
    /// - `true`: 截断该节点的梯度流，反向传播时不向父节点传播
    /// - `false`: 正常传播梯度（默认状态）
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
