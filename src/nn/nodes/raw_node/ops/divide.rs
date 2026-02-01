/*
 * @Author       : 老董
 * @Date         : 2026-01-09
 * @Description  : 逐元素除法节点
 *                 实现 C = A / B（element-wise division）
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::shape::DynamicShape;
use crate::tensor::{Tensor, broadcast_shape};

/// Divide 节点：逐元素除法
/// 支持广播：两个父节点形状需广播兼容，输出形状为广播后的形状
#[derive(Clone)]
pub(crate) struct Divide {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 固定形状
    fixed_shape: Vec<usize>,
    /// 动态形状（支持动态 batch）
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    supports_dynamic: bool,
    parents_ids: Vec<NodeId>, // [left, right] 用于区分被除数和除数
}

impl Divide {
    /// 从父节点形状信息创建 Divide 节点（核心实现）
    pub(in crate::nn) fn new_from_shapes(
        parent_shapes: &[&[usize]],
        parent_dynamic_shapes: &[DynamicShape],
        parent_ids: Vec<NodeId>,
    ) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parent_shapes.len() != 2 {
            return Err(GraphError::InvalidOperation(
                "Divide 节点需要正好 2 个父节点".to_string(),
            ));
        }
        if parent_shapes.len() != parent_dynamic_shapes.len() {
            return Err(GraphError::InvalidOperation(
                "父节点形状数量与动态形状数量不匹配".to_string(),
            ));
        }

        // 2. 计算广播后的固定形状
        let fixed_shape = broadcast_shape(parent_shapes[0], parent_shapes[1]).ok_or_else(|| {
            GraphError::ShapeMismatch {
                expected: parent_shapes[0].to_vec(),
                got: parent_shapes[1].to_vec(),
                message: "Divide 节点的父节点形状无法广播".to_string(),
            }
        })?;

        // 3. 计算动态形状
        let supports_dynamic = parent_dynamic_shapes
            .iter()
            .any(|ds| ds.has_dynamic_dims());
        let dynamic_shape = parent_dynamic_shapes[0].broadcast_with(&parent_dynamic_shapes[1]);

        // 4. 返回
        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape,
            dynamic_shape,
            supports_dynamic,
            parents_ids: parent_ids,
        })
    }

    /// 从 NodeHandle 创建（过渡期 API，委托给 new_from_shapes）
    pub(crate) fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        let shapes: Vec<Vec<usize>> = parents
            .iter()
            .map(|p| p.value_expected_shape().to_vec())
            .collect();
        let shapes_ref: Vec<&[usize]> = shapes.iter().map(|s| s.as_slice()).collect();
        let dynamic_shapes: Vec<DynamicShape> =
            parents.iter().map(|p| p.dynamic_expected_shape()).collect();
        let parent_ids: Vec<NodeId> = parents.iter().map(|p| p.id()).collect();

        Self::new_from_shapes(&shapes_ref, &dynamic_shapes, parent_ids)
    }
}

impl TraitNode for Divide {
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
        // 计算逐元素除法
        self.value = Some(parent_values[0] / parent_values[1]);
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// 计算 Divide 节点对父节点的梯度（VJP）
    ///
    /// 对于 C = A / B（逐元素除法，支持广播）：
    /// - ∂L/∂A = `sum_to_shape(upstream_grad` / B, `shape_A`)
    /// - ∂L/∂B = sum_to_shape(-upstream_grad * A / B², `shape_B`)
    ///
    /// 当 A 或 B 被广播时，梯度需要沿广播维度求和
    fn calc_grad_to_parent(
        &self,
        target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // 获取辅助父节点
        let assistant = assistant_parent.ok_or_else(|| {
            GraphError::ComputationError("Divide 节点计算梯度需要辅助父节点".to_string())
        })?;

        // 使用实际值的形状（支持动态 batch）
        let target_shape = target_parent
            .value()
            .ok_or_else(|| {
                GraphError::ComputationError(format!("Divide 梯度计算时父{target_parent}没有值"))
            })?
            .shape();

        if target_parent.id() == self.parents_ids[0] {
            // target 是 left (A)，assistant 是 right (B)
            // ∂L/∂A = upstream_grad / B，然后 sum_to_shape
            let b_value = assistant.value().ok_or_else(|| {
                GraphError::ComputationError(format!("{} 的辅助父节点没有值", self.display_node()))
            })?;
            let local_grad = upstream_grad / b_value;

            // 如果 A 被广播过，需要沿广播维度求和
            if local_grad.shape() == target_shape {
                Ok(local_grad)
            } else {
                Ok(local_grad.sum_to_shape(target_shape))
            }
        } else if target_parent.id() == self.parents_ids[1] {
            // target 是 right (B)，assistant 是 left (A)
            // ∂L/∂B = -upstream_grad * A / B²，然后 sum_to_shape
            let a_value = assistant.value().ok_or_else(|| {
                GraphError::ComputationError(format!("{} 的辅助父节点没有值", self.display_node()))
            })?;
            let b_value = target_parent.value().ok_or_else(|| {
                GraphError::ComputationError(format!("{} 的目标父节点没有值", self.display_node()))
            })?;
            // -upstream_grad * A / B²
            let b_squared = b_value * b_value;
            let neg_grad_a = &(upstream_grad * a_value) * (-1.0_f32);
            let local_grad = &neg_grad_a / &b_squared;

            // 如果 B 被广播过，需要沿广播维度求和
            if local_grad.shape() == target_shape {
                Ok(local_grad)
            } else {
                Ok(local_grad.sum_to_shape(target_shape))
            }
        } else {
            Err(GraphError::ComputationError(format!(
                "{} 不是当前 {} 的父节点",
                target_parent,
                self.display_node()
            )))
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
