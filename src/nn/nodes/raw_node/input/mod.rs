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
/// - `Target`: Loss 的目标值（真实标签）
/// - `Smart`: 模型入口（ModelState 使用），支持额外功能
#[derive(Clone)]
pub(crate) enum InputVariant {
    /// 普通数据输入（用户手动创建）
    Data(BasicInput),
    /// Loss 目标值（Criterion 内部创建）
    Target(BasicInput),
    /// 智能输入（ModelState 使用，支持动态 batch、梯度路由等）
    Smart(SmartInput),
}

impl InputVariant {
    /// 创建 Data 变体
    pub(crate) fn new_data(shape: &[usize]) -> Result<Self, GraphError> {
        Ok(Self::Data(BasicInput::new(shape)?))
    }

    /// 创建 Target 变体
    pub(crate) fn new_target(shape: &[usize]) -> Result<Self, GraphError> {
        Ok(Self::Target(BasicInput::new(shape)?))
    }

    /// 创建 Smart 变体
    pub(crate) fn new_smart(shape: &[usize]) -> Self {
        Self::Smart(SmartInput::new(shape))
    }

    /// 获取内部的 SmartInput（仅对 Smart 变体有效）
    #[allow(dead_code)]
    pub(crate) fn as_smart(&self) -> Option<&SmartInput> {
        match self {
            Self::Smart(inner) => Some(inner),
            _ => None,
        }
    }

    /// 获取变体类型名称（用于可视化）
    #[allow(dead_code)]
    pub(crate) fn variant_name(&self) -> &'static str {
        match self {
            Self::Data(_) => "Data",
            Self::Target(_) => "Target",
            Self::Smart(_) => "Smart",
        }
    }
}

// 为 InputVariant 实现 TraitNode，委托给内部变体
impl TraitNode for InputVariant {
    fn id(&self) -> NodeId {
        match self {
            Self::Data(inner) => inner.id(),
            Self::Target(inner) => inner.id(),
            Self::Smart(inner) => inner.id(),
        }
    }

    fn set_id(&mut self, id: NodeId) {
        match self {
            Self::Data(inner) => inner.set_id(id),
            Self::Target(inner) => inner.set_id(id),
            Self::Smart(inner) => inner.set_id(id),
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::Data(inner) => inner.name(),
            Self::Target(inner) => inner.name(),
            Self::Smart(inner) => inner.name(),
        }
    }

    fn set_name(&mut self, name: &str) {
        match self {
            Self::Data(inner) => inner.set_name(name),
            Self::Target(inner) => inner.set_name(name),
            Self::Smart(inner) => inner.set_name(name),
        }
    }

    fn get_type_name(&self) -> &'static str {
        match self {
            Self::Data(_) => "Input",
            Self::Target(_) => "Target",
            Self::Smart(_) => "SmartInput",
        }
    }

    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
        match self {
            Self::Data(inner) => inner.calc_value_by_parents(parents),
            Self::Target(inner) => inner.calc_value_by_parents(parents),
            Self::Smart(inner) => inner.calc_value_by_parents(parents),
        }
    }

    fn value(&self) -> Option<&Tensor> {
        match self {
            Self::Data(inner) => inner.value(),
            Self::Target(inner) => inner.value(),
            Self::Smart(inner) => inner.value(),
        }
    }

    fn set_value(&mut self, value: Option<&Tensor>) -> Result<(), GraphError> {
        match self {
            Self::Data(inner) => inner.set_value(value),
            Self::Target(inner) => inner.set_value(value),
            Self::Smart(inner) => inner.set_value(value),
        }
    }

    fn clear_value(&mut self) -> Result<(), GraphError> {
        match self {
            Self::Data(inner) => inner.clear_value(),
            Self::Target(inner) => inner.clear_value(),
            Self::Smart(inner) => inner.clear_value(),
        }
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        match self {
            Self::Data(inner) => inner.set_value_unchecked(value),
            Self::Target(inner) => inner.set_value_unchecked(value),
            Self::Smart(inner) => inner.set_value_unchecked(value),
        }
    }

    fn value_expected_shape(&self) -> &[usize] {
        match self {
            Self::Data(inner) => inner.value_expected_shape(),
            Self::Target(inner) => inner.value_expected_shape(),
            Self::Smart(inner) => inner.value_expected_shape(),
        }
    }

    fn dynamic_expected_shape(&self) -> DynamicShape {
        match self {
            Self::Data(inner) => inner.dynamic_expected_shape(),
            Self::Target(inner) => inner.dynamic_expected_shape(),
            Self::Smart(inner) => inner.dynamic_expected_shape(),
        }
    }

    fn supports_dynamic_batch(&self) -> bool {
        match self {
            Self::Data(inner) => inner.supports_dynamic_batch(),
            Self::Target(inner) => inner.supports_dynamic_batch(),
            Self::Smart(inner) => inner.supports_dynamic_batch(),
        }
    }

    fn calc_grad_to_parent(
        &self,
        target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        match self {
            Self::Data(_) | Self::Target(_) => Err(GraphError::InvalidOperation(
                "BasicInput 没有父节点，不应计算父节点梯度".to_string(),
            )),
            Self::Smart(inner) => {
                inner.calc_grad_to_parent(target_parent, upstream_grad, assistant_parent)
            }
        }
    }

    fn grad(&self) -> Option<&Tensor> {
        match self {
            Self::Data(_) | Self::Target(_) => None,
            Self::Smart(inner) => inner.grad(),
        }
    }

    fn set_grad(&mut self, grad: Option<&Tensor>) -> Result<(), GraphError> {
        match self {
            Self::Data(_) | Self::Target(_) => Err(GraphError::InvalidOperation(
                "BasicInput 不支持设置梯度".to_string(),
            )),
            Self::Smart(inner) => inner.set_grad(grad),
        }
    }
}
