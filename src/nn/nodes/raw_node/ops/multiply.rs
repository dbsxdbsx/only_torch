/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : 逐元素乘法节点（Hadamard积）
 *                 参考自：MatrixSlow/matrixslow/ops/ops.py#L154 (class Multiply)
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::tensor::{Tensor, broadcast_shape};

/// Multiply节点：逐元素乘法（Hadamard积）
/// 支持广播：两个父节点形状需广播兼容，输出形状为广播后的形状
#[derive(Clone)]
pub(crate) struct Multiply {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    shape: Vec<usize>,
    parents_ids: Vec<NodeId>, // 用于区分左右父节点
}

impl Multiply {
    pub(crate) fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parents.len() != 2 {
            return Err(GraphError::InvalidOperation(
                "Multiply节点需要正好2个父节点".to_string(),
            ));
        }

        // 2. 计算广播后的输出形状
        let left_shape = parents[0].value_expected_shape();
        let right_shape = parents[1].value_expected_shape();

        let shape = broadcast_shape(left_shape, right_shape).ok_or_else(|| {
            GraphError::ShapeMismatch {
                expected: left_shape.to_vec(),
                got: right_shape.to_vec(),
                message: "Multiply节点的父节点形状无法广播".to_string(),
            }
        })?;

        // 3. 返回
        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            shape,
            parents_ids: vec![parents[0].id(), parents[1].id()],
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
        &self.shape
    }

    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
        // 1. 获取两个父节点的值
        let left_value = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的第1个父节点{}没有值",
                self.display_node(),
                parents[0]
            ))
        })?;

        let right_value = parents[1].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的第2个父节点{}没有值",
                self.display_node(),
                parents[1]
            ))
        })?;

        // 2. 计算逐元素乘法
        self.value = Some(left_value * right_value);

        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// 计算 Multiply 节点对父节点的梯度（VJP）
    ///
    /// 对于 C = A ⊙ B（逐元素乘法，支持广播）：
    /// - ∂L/∂A = sum_to_shape(`upstream_grad` ⊙ B, shape_A)
    /// - ∂L/∂B = sum_to_shape(`upstream_grad` ⊙ A, shape_B)
    ///
    /// 当 A 或 B 被广播时，梯度需要沿广播维度求和
    fn calc_grad_to_parent(
        &self,
        target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // 获取辅助父节点（另一个操作数）
        let assistant = assistant_parent.ok_or_else(|| {
            GraphError::ComputationError("Multiply 节点计算梯度需要辅助父节点".to_string())
        })?;

        // 获取 target 的原始形状
        let target_shape = target_parent.value_expected_shape();

        // 确定哪个是 target，哪个是 assistant
        if target_parent.id() == self.parents_ids[0] {
            // target 是 left (A)，assistant 是 right (B)
            // ∂L/∂A = upstream_grad ⊙ B，然后 sum_to_shape
            let b_value = assistant.value().ok_or_else(|| {
                GraphError::ComputationError(format!("{} 的辅助父节点没有值", self.display_node()))
            })?;
            let local_grad = upstream_grad * b_value;

            // 如果 A 被广播过，需要沿广播维度求和
            if local_grad.shape() == target_shape {
                Ok(local_grad)
            } else {
                Ok(local_grad.sum_to_shape(target_shape))
            }
        } else if target_parent.id() == self.parents_ids[1] {
            // target 是 right (B)，assistant 是 left (A)
            // ∂L/∂B = upstream_grad ⊙ A，然后 sum_to_shape
            let a_value = assistant.value().ok_or_else(|| {
                GraphError::ComputationError(format!("{} 的辅助父节点没有值", self.display_node()))
            })?;
            let local_grad = upstream_grad * a_value;

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
