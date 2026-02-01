/*
 * @Author       : 老董
 * @Date         : 2026-01-19
 * @Description  : 逐元素减法节点
 *                 实现 C = A - B（element-wise subtraction）
 *                 支持 NumPy 风格广播
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::shape::DynamicShape;
use crate::tensor::{Tensor, broadcast_shape};

/// Subtract 节点：逐元素减法
/// 支持广播：两个父节点形状需广播兼容，输出形状为广播后的形状
#[derive(Clone)]
pub(crate) struct Subtract {
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
    parents_ids: Vec<NodeId>, // [left, right] 用于区分被减数和减数
}

impl Subtract {
    /// 从父节点形状信息创建 Subtract 节点（核心实现）
    ///
    /// # 参数
    /// - `parent_shapes`: 父节点的固定形状列表 [left, right]
    /// - `parent_dynamic_shapes`: 父节点的动态形状列表
    /// - `parent_ids`: 父节点 ID 列表（用于梯度计算时区分被减数和减数）
    pub(in crate::nn) fn new_from_shapes(
        parent_shapes: &[&[usize]],
        parent_dynamic_shapes: &[DynamicShape],
        parent_ids: Vec<NodeId>,
    ) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parent_shapes.len() != 2 {
            return Err(GraphError::InvalidOperation(
                "Subtract 节点需要正好 2 个父节点".to_string(),
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
                message: "Subtract 节点的父节点形状无法广播".to_string(),
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
        // 提取形状信息
        let shapes: Vec<Vec<usize>> = parents
            .iter()
            .map(|p| p.value_expected_shape().to_vec())
            .collect();
        let shapes_ref: Vec<&[usize]> = shapes.iter().map(|s| s.as_slice()).collect();
        let dynamic_shapes: Vec<DynamicShape> =
            parents.iter().map(|p| p.dynamic_expected_shape()).collect();
        let parent_ids: Vec<NodeId> = parents.iter().map(|p| p.id()).collect();

        // 委托给核心实现
        Self::new_from_shapes(&shapes_ref, &dynamic_shapes, parent_ids)
    }
}

impl TraitNode for Subtract {
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
        // 计算逐元素减法（ndarray 原生支持广播）
        self.value = Some(parent_values[0] - parent_values[1]);
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// 计算 Subtract 节点对父节点的梯度（VJP）
    ///
    /// 对于 C = A - B（逐元素减法，支持广播）：
    /// - ∂L/∂A = `sum_to_shape(upstream_grad`, `shape_A`)
    /// - ∂L/∂B = sum_to_shape(-upstream_grad, `shape_B`)
    ///
    /// 当 A 或 B 被广播时，梯度需要沿广播维度求和
    fn calc_grad_to_parent(
        &self,
        target_parent_index: usize,
        parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<Tensor, GraphError> {
        // 使用实际值的形状（支持动态 batch）
        let target_value = parent_values.get(target_parent_index).ok_or_else(|| {
            GraphError::ComputationError(format!(
                "Subtract 梯度计算时父节点索引 {} 超出范围",
                target_parent_index
            ))
        })?;
        let target_shape = target_value.shape();

        if target_parent_index == 0 {
            // target 是 left (A)：∂L/∂A = upstream_grad
            if upstream_grad.shape() == target_shape {
                Ok(upstream_grad.clone())
            } else {
                Ok(upstream_grad.sum_to_shape(target_shape))
            }
        } else if target_parent_index == 1 {
            // target 是 right (B)：∂L/∂B = -upstream_grad
            let neg_grad = upstream_grad * (-1.0_f32);
            if neg_grad.shape() == target_shape {
                Ok(neg_grad)
            } else {
                Ok(neg_grad.sum_to_shape(target_shape))
            }
        } else {
            Err(GraphError::ComputationError(format!(
                "Subtract 节点只有 2 个父节点，索引 {} 无效",
                target_parent_index
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
