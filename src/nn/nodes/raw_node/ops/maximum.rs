/*
 * @Author       : 老董
 * @Date         : 2026-01-31
 * @Description  : Maximum 节点 - 逐元素取两个张量的最大值
 *
 * 用于强化学习（PPO/TD3 等）需要可微分 max 操作的场景
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::shape::DynamicShape;
use crate::tensor::{Tensor, broadcast_shape};

/// Maximum 节点：逐元素取两个张量的最大值
///
/// 前向：`output[i] = max(a[i], b[i])`
/// 反向：
/// - 对 a: `grad_a[i] = upstream[i] * (a[i] >= b[i] ? 1 : 0)`
/// - 对 b: `grad_b[i] = upstream[i] * (a[i] < b[i] ? 1 : 0)`
/// - 当 a[i] == b[i] 时，梯度各 0.5
///
/// # 主要用途
/// - PPO: `min(ratio * adv, clipped_ratio * adv)` 中的 min（Maximum 的对偶）
/// - TD3: `min(Q1, Q2)` 计算 actor loss
///
/// # 父节点顺序
/// - parents[0]: 第一个输入张量
/// - parents[1]: 第二个输入张量
#[derive(Clone)]
pub(crate) struct Maximum {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 固定形状（广播后的形状）
    fixed_shape: Vec<usize>,
    /// 动态形状
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    supports_dynamic: bool,
}

impl Maximum {
    /// 从父节点形状信息创建 Maximum 节点（核心实现）
    pub(in crate::nn) fn new_from_shapes(
        a_shape: &[usize],
        b_shape: &[usize],
        a_dynamic_shape: &DynamicShape,
        b_dynamic_shape: &DynamicShape,
    ) -> Result<Self, GraphError> {
        // 计算广播后的形状
        let fixed_shape = broadcast_shape(a_shape, b_shape).ok_or_else(|| {
            GraphError::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: b_shape.to_vec(),
                message: "Maximum 节点的父节点形状无法广播".to_string(),
            }
        })?;

        // 动态形状
        let dynamic_shape = a_dynamic_shape.broadcast_with(b_dynamic_shape);

        // 是否支持动态 batch
        let supports_dynamic = a_dynamic_shape.dims().first() == Some(&None)
            || b_dynamic_shape.dims().first() == Some(&None);

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape,
            dynamic_shape,
            supports_dynamic,
        })
    }

    /// 从 NodeHandle 创建（过渡期 API，委托给 new_from_shapes）
    pub(crate) fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        if parents.len() != 2 {
            return Err(GraphError::InvalidOperation(
                "Maximum 节点需要 2 个父节点".to_string(),
            ));
        }

        Self::new_from_shapes(
            &parents[0].value_expected_shape(),
            &parents[1].value_expected_shape(),
            &parents[0].dynamic_expected_shape(),
            &parents[1].dynamic_expected_shape(),
        )
    }

    #[deprecated(note = "保留旧 API 签名，委托给 new_from_shapes")]
    #[allow(dead_code)]
    fn _new_legacy(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        // 1. 验证父节点数量（需要 2 个）
        if parents.len() != 2 {
            return Err(GraphError::InvalidOperation(
                "Maximum 节点需要 2 个父节点".to_string(),
            ));
        }

        let a_shape = parents[0].value_expected_shape();
        let b_shape = parents[1].value_expected_shape();

        // 2. 计算广播后的形状
        let fixed_shape = broadcast_shape(a_shape, b_shape).ok_or_else(|| {
            GraphError::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: b_shape.to_vec(),
                message: "Maximum 节点的父节点形状无法广播".to_string(),
            }
        })?;

        // 3. 动态形状
        let a_dyn = parents[0].dynamic_expected_shape();
        let b_dyn = parents[1].dynamic_expected_shape();
        let dynamic_shape = a_dyn.broadcast_with(&b_dyn);

        // 4. 是否支持动态 batch
        let supports_dynamic =
            parents[0].supports_dynamic_batch() || parents[1].supports_dynamic_batch();

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape,
            dynamic_shape,
            supports_dynamic,
        })
    }
}

impl TraitNode for Maximum {
    fn id(&self) -> NodeId {
        self.id.unwrap()
    }

    fn set_id(&mut self, id: NodeId) {
        self.id = Some(id);
    }

    fn name(&self) -> &str {
        self.name.as_ref().unwrap()
    }

    fn set_name(&mut self, name: &str) {
        self.name = Some(name.to_string());
    }

    fn value_expected_shape(&self) -> &[usize] {
        &self.fixed_shape
    }

    fn dynamic_expected_shape(&self) -> DynamicShape {
        self.dynamic_shape.clone()
    }

    fn supports_dynamic_batch(&self) -> bool {
        self.supports_dynamic
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        // 使用 Tensor::maximum
        self.value = Some(parent_values[0].maximum(parent_values[1]));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_grad_to_parent(
        &self,
        target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // Maximum 的反向传播：
        // - 对 target: grad = upstream * (target >= other ? 1 : 0)
        // - 当 target == other 时，梯度各 0.5

        let other_parent = assistant_parent.ok_or_else(|| {
            GraphError::ComputationError(
                "Maximum calc_grad_to_parent 需要 assistant_parent".to_string(),
            )
        })?;

        let target_value = target_parent.value().ok_or_else(|| {
            GraphError::ComputationError("Maximum 梯度计算时 target_parent 没有值".to_string())
        })?;

        let other_value = other_parent.value().ok_or_else(|| {
            GraphError::ComputationError("Maximum 梯度计算时 assistant_parent 没有值".to_string())
        })?;

        let target_shape = target_value.shape().to_vec();
        let output_shape = upstream_grad.shape();

        // 1. 将 target 和 other 广播到输出形状
        let target_broadcast = target_value.broadcast_to(output_shape);
        let other_broadcast = other_value.broadcast_to(output_shape);

        // 2. 在广播后的形状上逐元素计算 mask
        let target_slice = target_broadcast.data_as_slice();
        let other_slice = other_broadcast.data_as_slice();
        let upstream_slice = upstream_grad.data_as_slice();

        let mut grad_data = Vec::with_capacity(upstream_slice.len());

        for i in 0..upstream_slice.len() {
            let t = target_slice[i];
            let o = other_slice[i];
            let u = upstream_slice[i];

            let mask = if t > o {
                1.0
            } else if t < o {
                0.0
            } else {
                0.5 // t == o，梯度各分一半
            };
            grad_data.push(u * mask);
        }

        let grad = Tensor::new(&grad_data, output_shape);

        // 3. 如果 target 被广播过，用 sum_to_shape 收缩回原始形状
        if output_shape != target_shape.as_slice() {
            Ok(grad.sum_to_shape(&target_shape))
        } else {
            Ok(grad)
        }
    }

    fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
    }

    fn set_grad(&mut self, grad: Option<&Tensor>) -> Result<(), GraphError> {
        self.grad = grad.cloned();
        Ok(())
    }

    fn clear_value(&mut self) -> Result<(), GraphError> {
        self.value = None;
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
