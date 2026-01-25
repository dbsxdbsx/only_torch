/*
 * Input 模块：统一的输入节点类型
 *
 * 所有"接收外部数据"的节点都归类为 Input，通过 InputVariant 区分：
 * - Data: 用户手动创建的通用输入
 * - Target: Loss 的目标值（真实标签）
 * - Smart: 模型入口（ModelState 使用），支持动态 batch、梯度路由等
 */

mod basic;
mod smart;

pub(crate) use basic::BasicInput;
pub(crate) use smart::SmartInput;

use super::TraitNode;
use crate::nn::nodes::NodeHandle;
use crate::nn::shape::DynamicShape;
use crate::nn::{GraphError, NodeId};
use crate::tensor::Tensor;

/// 输入节点的变体
///
/// 所有"接收外部数据"的节点都归类为 Input，通过变体区分用途：
/// - `Data`: 用户手动创建的通用输入
/// - `Target`: Loss 的目标值（真实标签），支持动态 batch
/// - `Smart`: 模型入口（ModelState 使用），支持动态 batch、梯度路由等
/// - `RecurrentOutput`: 循环层输出桥接（RNN/LSTM/GRU 内部使用）
#[derive(Clone)]
pub(crate) enum InputVariant {
    /// 普通数据输入（用户手动创建）
    Data(BasicInput),
    /// Loss 目标值（Criterion 内部创建，支持动态 batch）
    Target(SmartInput),
    /// 智能输入（ModelState 使用，支持动态 batch、梯度路由等）
    Smart(SmartInput),
    /// 循环层输出桥接（RNN/LSTM/GRU 内部使用，固定 node_id 用于下游复用）
    RecurrentOutput(SmartInput),
}

impl InputVariant {
    /// 创建 Data 变体
    pub(crate) fn new_data(shape: &[usize]) -> Result<Self, GraphError> {
        Ok(Self::Data(BasicInput::new(shape)?))
    }

    /// 创建 Target 变体（支持动态 batch）
    pub(crate) fn new_target(shape: &[usize]) -> Self {
        Self::Target(SmartInput::new(shape))
    }

    /// 创建 Smart 变体
    pub(crate) fn new_smart(shape: &[usize]) -> Self {
        Self::Smart(SmartInput::new(shape))
    }

    /// 创建 RecurrentOutput 变体（循环层输出桥接）
    pub(crate) fn new_recurrent_output(shape: &[usize]) -> Self {
        Self::RecurrentOutput(SmartInput::new(shape))
    }

    /// 获取内部的 SmartInput（对 Target、Smart、RecurrentOutput 变体有效）
    #[allow(dead_code)]
    pub(crate) fn as_smart(&self) -> Option<&SmartInput> {
        match self {
            Self::Target(inner) | Self::Smart(inner) | Self::RecurrentOutput(inner) => Some(inner),
            Self::Data(_) => None,
        }
    }

    /// 获取变体类型名称（用于可视化）
    #[allow(dead_code)]
    pub(crate) fn variant_name(&self) -> &'static str {
        match self {
            Self::Data(_) => "Data",
            Self::Target(_) => "Target",
            Self::Smart(_) => "Smart",
            Self::RecurrentOutput(_) => "RecurrentOutput",
        }
    }
}

// 为 InputVariant 实现 TraitNode，委托给内部变体
impl TraitNode for InputVariant {
    fn id(&self) -> NodeId {
        match self {
            Self::Data(inner) => inner.id(),
            Self::Target(inner) | Self::Smart(inner) | Self::RecurrentOutput(inner) => inner.id(),
        }
    }

    fn set_id(&mut self, id: NodeId) {
        match self {
            Self::Data(inner) => inner.set_id(id),
            Self::Target(inner) | Self::Smart(inner) | Self::RecurrentOutput(inner) => {
                inner.set_id(id)
            }
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::Data(inner) => inner.name(),
            Self::Target(inner) | Self::Smart(inner) | Self::RecurrentOutput(inner) => {
                inner.name()
            }
        }
    }

    fn set_name(&mut self, name: &str) {
        match self {
            Self::Data(inner) => inner.set_name(name),
            Self::Target(inner) | Self::Smart(inner) | Self::RecurrentOutput(inner) => {
                inner.set_name(name)
            }
        }
    }

    fn get_type_name(&self) -> &'static str {
        match self {
            Self::Data(_) => "Input",
            Self::Target(_) => "TargetInput",
            Self::Smart(_) => "SmartInput",
            Self::RecurrentOutput(_) => "RecurrentOutput",
        }
    }

    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
        match self {
            Self::Data(inner) => inner.calc_value_by_parents(parents),
            Self::Target(inner) | Self::Smart(inner) | Self::RecurrentOutput(inner) => {
                inner.calc_value_by_parents(parents)
            }
        }
    }

    fn value(&self) -> Option<&Tensor> {
        match self {
            Self::Data(inner) => inner.value(),
            Self::Target(inner) | Self::Smart(inner) | Self::RecurrentOutput(inner) => {
                inner.value()
            }
        }
    }

    fn set_value(&mut self, value: Option<&Tensor>) -> Result<(), GraphError> {
        match self {
            Self::Data(inner) => inner.set_value(value),
            Self::Target(inner) | Self::Smart(inner) | Self::RecurrentOutput(inner) => {
                inner.set_value(value)
            }
        }
    }

    fn clear_value(&mut self) -> Result<(), GraphError> {
        match self {
            Self::Data(inner) => inner.clear_value(),
            Self::Target(inner) | Self::Smart(inner) | Self::RecurrentOutput(inner) => {
                inner.clear_value()
            }
        }
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        match self {
            Self::Data(inner) => inner.set_value_unchecked(value),
            Self::Target(inner) | Self::Smart(inner) | Self::RecurrentOutput(inner) => {
                inner.set_value_unchecked(value)
            }
        }
    }

    fn value_expected_shape(&self) -> &[usize] {
        match self {
            Self::Data(inner) => inner.value_expected_shape(),
            Self::Target(inner) | Self::Smart(inner) | Self::RecurrentOutput(inner) => {
                inner.value_expected_shape()
            }
        }
    }

    fn dynamic_expected_shape(&self) -> DynamicShape {
        match self {
            Self::Data(inner) => inner.dynamic_expected_shape(),
            Self::Target(inner) | Self::Smart(inner) | Self::RecurrentOutput(inner) => {
                inner.dynamic_expected_shape()
            }
        }
    }

    fn supports_dynamic_batch(&self) -> bool {
        match self {
            Self::Data(inner) => inner.supports_dynamic_batch(),
            Self::Target(inner) | Self::Smart(inner) | Self::RecurrentOutput(inner) => {
                inner.supports_dynamic_batch()
            }
        }
    }

    fn calc_grad_to_parent(
        &self,
        target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        match self {
            Self::Data(_) => Err(GraphError::InvalidOperation(
                "BasicInput 没有父节点，不应计算父节点梯度".to_string(),
            )),
            Self::Target(inner) | Self::Smart(inner) | Self::RecurrentOutput(inner) => {
                inner.calc_grad_to_parent(target_parent, upstream_grad, assistant_parent)
            }
        }
    }

    fn grad(&self) -> Option<&Tensor> {
        match self {
            Self::Data(_) => None,
            Self::Target(inner) | Self::Smart(inner) | Self::RecurrentOutput(inner) => {
                inner.grad()
            }
        }
    }

    fn set_grad(&mut self, grad: Option<&Tensor>) -> Result<(), GraphError> {
        match self {
            Self::Data(_) => Err(GraphError::InvalidOperation(
                "BasicInput 不支持设置梯度".to_string(),
            )),
            Self::Target(inner) | Self::Smart(inner) | Self::RecurrentOutput(inner) => {
                inner.set_grad(grad)
            }
        }
    }
}
