/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : 逐元素乘法节点（Hadamard积）
 *                 参考自：MatrixSlow/matrixslow/ops/ops.py#L154 (class Multiply)
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::tensor::Tensor;

/// Multiply节点：逐元素乘法（Hadamard积）
/// 两个父节点必须形状相同，输出形状与输入相同
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

        // 2. 验证两个父节点形状相同
        let shape = parents[0].value_expected_shape().to_vec();
        if parents[1].value_expected_shape() != shape {
            return Err(GraphError::ShapeMismatch {
                expected: shape,
                got: parents[1].value_expected_shape().to_vec(),
                message: "Multiply节点的两个父节点形状必须相同".to_string(),
            });
        }

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
    /// 对于 C = A ⊙ B（逐元素乘法）：
    /// - ∂L/∂A = `upstream_grad` ⊙ B
    /// - ∂L/∂B = `upstream_grad` ⊙ A
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

        // 确定哪个是 target，哪个是 assistant
        if target_parent.id() == self.parents_ids[0] {
            // target 是 left (A)，assistant 是 right (B)
            // ∂L/∂A = upstream_grad ⊙ B
            let b_value = assistant.value().ok_or_else(|| {
                GraphError::ComputationError(format!("{} 的辅助父节点没有值", self.display_node()))
            })?;
            Ok(upstream_grad * b_value)
        } else if target_parent.id() == self.parents_ids[1] {
            // target 是 right (B)，assistant 是 left (A)
            // ∂L/∂B = upstream_grad ⊙ A
            let a_value = assistant.value().ok_or_else(|| {
                GraphError::ComputationError(format!("{} 的辅助父节点没有值", self.display_node()))
            })?;
            Ok(upstream_grad * a_value)
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
