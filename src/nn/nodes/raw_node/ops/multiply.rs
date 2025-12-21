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
    jacobi: Option<Tensor>,
    shape: Vec<usize>,
    parents_ids: Vec<NodeId>, // 保留用于雅可比计算，[left_id, right_id]
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
            jacobi: None,
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

    /// 计算Multiply节点对父节点的雅可比矩阵
    /// 参考MatrixSlow: MatrixSlow/matrixslow/ops/ops.py#L162 (Multiply.get_jacobi)
    ///
    /// 设 C = A ⊙ B (逐元素乘法)，其中A和B形状相同(m,n)
    ///
    /// 对于A（第一个父节点）：
    ///   ∂C/∂A = diag(B.flatten()) → shape: [m*n, m*n]
    ///
    /// 对于B（第二个父节点）：
    ///   ∂C/∂B = diag(A.flatten()) → shape: [m*n, m*n]
    fn calc_jacobi_to_a_parent(
        &self,
        target_parent: &NodeHandle,
        assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // 获取两个父节点的值
        let (left_value, right_value) = if target_parent.id() == self.parents_ids[0] {
            // target是left，assistant是right
            let assistant = assistant_parent.ok_or_else(|| {
                GraphError::ComputationError("Multiply计算雅可比矩阵需要辅助父节点".to_string())
            })?;
            (
                target_parent.value().ok_or_else(|| {
                    GraphError::ComputationError(format!(
                        "{}的第1个父节点没有值",
                        self.display_node()
                    ))
                })?,
                assistant.value().ok_or_else(|| {
                    GraphError::ComputationError(format!(
                        "{}的第2个父节点没有值",
                        self.display_node()
                    ))
                })?,
            )
        } else if target_parent.id() == self.parents_ids[1] {
            // target是right，assistant是left
            let assistant = assistant_parent.ok_or_else(|| {
                GraphError::ComputationError("Multiply计算雅可比矩阵需要辅助父节点".to_string())
            })?;
            (
                assistant.value().ok_or_else(|| {
                    GraphError::ComputationError(format!(
                        "{}的第1个父节点没有值",
                        self.display_node()
                    ))
                })?,
                target_parent.value().ok_or_else(|| {
                    GraphError::ComputationError(format!(
                        "{}的第2个父节点没有值",
                        self.display_node()
                    ))
                })?,
            )
        } else {
            return Err(GraphError::ComputationError(format!(
                "{}不是当前{}的父节点",
                target_parent,
                self.display_node()
            )));
        };

        // 根据目标父节点计算雅可比矩阵
        if target_parent.id() == self.parents_ids[0] {
            // 对left的雅可比：∂C/∂A = diag(B.flatten())
            Ok(right_value.flatten().diag())
        } else {
            // 对right的雅可比：∂C/∂B = diag(A.flatten())
            Ok(left_value.flatten().diag())
        }
    }

    fn jacobi(&self) -> Option<&Tensor> {
        self.jacobi.as_ref()
    }

    fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError> {
        self.jacobi = jacobi.cloned();
        Ok(())
    }
}
