/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : 逐元素乘法节点（Hadamard积）
 *                 参考自：MatrixSlow/matrixslow/ops/ops.py#L154 (class Multiply)
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::NodeId;
use crate::nn::shape::DynamicShape;
use crate::tensor::{Tensor, broadcast_shape};

/// Multiply节点：逐元素乘法（Hadamard积）
/// 支持广播：两个父节点形状需广播兼容，输出形状为广播后的形状
#[derive(Clone)]
pub(crate) struct Multiply {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 固定形状
    fixed_shape: Vec<usize>,
    /// 动态形状（支持动态 batch）
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    #[allow(dead_code)]
    supports_dynamic: bool,
    #[allow(dead_code)]
    parents_ids: Vec<NodeId>, // 用于区分左右父节点
}

impl Multiply {
    /// 从父节点形状信息创建 Multiply 节点（核心实现）
    pub(in crate::nn) fn new(
        parent_shapes: &[&[usize]],
        parent_dynamic_shapes: &[DynamicShape],
        parent_ids: Vec<NodeId>,
    ) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parent_shapes.len() != 2 {
            return Err(GraphError::InvalidOperation(
                "Multiply节点需要正好2个父节点".to_string(),
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
                message: "Multiply节点的父节点形状无法广播".to_string(),
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

}

impl TraitNode for Multiply {
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
        // 计算逐元素乘法
        self.value = Some(parent_values[0] * parent_values[1]);
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// 计算 Multiply 节点对父节点的梯度（VJP）
    ///
    /// 对于 C = A ⊙ B（逐元素乘法，支持广播）：
    /// - ∂L/∂A = `sum_to_shape`(`upstream_grad` ⊙ B, `shape_A`)
    /// - ∂L/∂B = `sum_to_shape`(`upstream_grad` ⊙ A, `shape_B`)
    ///
    /// 当 A 或 B 被广播时，梯度需要沿广播维度求和
    fn calc_grad_to_parent(
        &self,
        target_parent_index: usize,
        parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<Tensor, GraphError> {
        // 获取目标父节点和辅助父节点的值
        let target_value = parent_values.get(target_parent_index).ok_or_else(|| {
            GraphError::ComputationError(format!(
                "Multiply 梯度计算时父节点索引 {} 超出范围",
                target_parent_index
            ))
        })?;
        let target_shape = target_value.shape();

        // 确定哪个是 target，哪个是 assistant
        if target_parent_index == 0 {
            // target 是 left (A)，assistant 是 right (B)
            // ∂L/∂A = upstream_grad ⊙ B，然后 sum_to_shape
            let b_value = parent_values.get(1).ok_or_else(|| {
                GraphError::ComputationError(format!("{} 的右父节点没有值", self.display_node()))
            })?;
            let local_grad = upstream_grad * *b_value;

            // 如果 A 被广播过，需要沿广播维度求和
            if local_grad.shape() == target_shape {
                Ok(local_grad)
            } else {
                Ok(local_grad.sum_to_shape(target_shape))
            }
        } else if target_parent_index == 1 {
            // target 是 right (B)，assistant 是 left (A)
            // ∂L/∂B = upstream_grad ⊙ A，然后 sum_to_shape
            let a_value = parent_values.get(0).ok_or_else(|| {
                GraphError::ComputationError(format!("{} 的左父节点没有值", self.display_node()))
            })?;
            let local_grad = upstream_grad * *a_value;

            // 如果 B 被广播过，需要沿广播维度求和
            if local_grad.shape() == target_shape {
                Ok(local_grad)
            } else {
                Ok(local_grad.sum_to_shape(target_shape))
            }
        } else {
            Err(GraphError::ComputationError(format!(
                "Multiply 节点只有 2 个父节点，索引 {} 无效",
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
