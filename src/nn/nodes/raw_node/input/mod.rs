/*
 * Input 模块：统一的输入节点类型
 *
 * 所有"接收外部数据"的节点都归类为 Input，通过 InputVariant 区分：
 * - Data: 用户数据输入（可视化蓝色）
 * - Target: Loss 的目标值/真实标签（可视化橙色）
 *
 * 两者底层共用 BasicInput 结构体，区别仅在于语义和可视化样式。
 */

mod basic;

pub(crate) use basic::BasicInput;

use super::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::nn::{GraphError, NodeId};
use crate::tensor::Tensor;

/// 输入节点的变体
///
/// 所有"接收外部数据"的节点都归类为 Input，通过变体区分用途：
/// - `Data`: 用户数据输入
/// - `Target`: Loss 的目标值（真实标签）
///
/// 底层均使用 `BasicInput`，区别仅在于语义和可视化样式。
#[derive(Clone)]
pub(crate) enum InputVariant {
    /// 用户数据输入
    Data(BasicInput),
    /// Loss 目标值（真实标签）
    Target(BasicInput),
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
}

impl InputVariant {
    /// 获取存储的 Tensor 的数据源 ID（用于同源数据追踪）
    pub(crate) fn value_source_id(&self) -> Option<u64> {
        match self {
            Self::Data(inner) | Self::Target(inner) => {
                inner.value().map(|t| t.source_id())
            }
        }
    }
}

// 为 InputVariant 实现 TraitNode，统一委托给内部 BasicInput
impl TraitNode for InputVariant {
    fn id(&self) -> NodeId {
        match self {
            Self::Data(inner) | Self::Target(inner) => inner.id(),
        }
    }

    fn set_id(&mut self, id: NodeId) {
        match self {
            Self::Data(inner) | Self::Target(inner) => inner.set_id(id),
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::Data(inner) | Self::Target(inner) => inner.name(),
        }
    }

    fn set_name(&mut self, name: &str) {
        match self {
            Self::Data(inner) | Self::Target(inner) => inner.set_name(name),
        }
    }

    fn get_type_name(&self) -> &'static str {
        match self {
            Self::Data(_) => "Input",
            Self::Target(_) => "TargetInput",
        }
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        match self {
            Self::Data(inner) | Self::Target(inner) => inner.calc_value_by_parents(parent_values),
        }
    }

    fn value(&self) -> Option<&Tensor> {
        match self {
            Self::Data(inner) | Self::Target(inner) => inner.value(),
        }
    }

    fn set_value(&mut self, value: Option<&Tensor>) -> Result<(), GraphError> {
        match self {
            Self::Data(inner) | Self::Target(inner) => inner.set_value(value),
        }
    }

    fn clear_value(&mut self) -> Result<(), GraphError> {
        match self {
            Self::Data(inner) | Self::Target(inner) => inner.clear_value(),
        }
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        match self {
            Self::Data(inner) | Self::Target(inner) => inner.set_value_unchecked(value),
        }
    }

    fn value_expected_shape(&self) -> &[usize] {
        match self {
            Self::Data(inner) | Self::Target(inner) => inner.value_expected_shape(),
        }
    }

    fn dynamic_expected_shape(&self) -> DynamicShape {
        match self {
            Self::Data(inner) | Self::Target(inner) => inner.dynamic_expected_shape(),
        }
    }

    fn supports_dynamic_batch(&self) -> bool {
        match self {
            Self::Data(inner) | Self::Target(inner) => inner.supports_dynamic_batch(),
        }
    }

    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        _upstream_grad: &Tensor,
    ) -> Result<Tensor, GraphError> {
        Err(GraphError::InvalidOperation(
            "Input 节点没有父节点，不应计算父节点梯度".to_string(),
        ))
    }

    // Input 节点不支持梯度（输入数据不参与梯度更新）
    fn grad(&self) -> Option<&Tensor> {
        None
    }

    fn set_grad(&mut self, _grad: Option<&Tensor>) -> Result<(), GraphError> {
        // 返回特定错误消息，accumulate_grad 会匹配并静默跳过
        Err(GraphError::InvalidOperation(
            "BasicInput 不支持设置梯度".to_string(),
        ))
    }
}
