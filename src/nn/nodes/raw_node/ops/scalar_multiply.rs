/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : 标量乘法节点，用于将标量(1x1矩阵)广播乘以任意形状的矩阵。
 *                 参考自：MatrixSlow/matrixslow/ops/ops.py#L327 (class ScalarMultiply)
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::tensor::Tensor;

/// ScalarMultiply节点：标量(1x1) × 矩阵(m,n) = 矩阵(m,n)
/// 第1个父节点必须是标量(1x1)，第2个父节点可以是任意形状的矩阵
#[derive(Clone)]
pub(crate) struct ScalarMultiply {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    shape: Vec<usize>,
    parents_ids: Vec<NodeId>, // 保留用于雅可比计算，[标量id, 矩阵id]
}

impl ScalarMultiply {
    pub(crate) fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parents.len() != 2 {
            return Err(GraphError::InvalidOperation(
                "ScalarMultiply节点需要正好2个父节点".to_string(),
            ));
        }

        // 2. 验证第1个父节点是标量(1x1)
        let scalar_shape = parents[0].value_expected_shape();
        if scalar_shape != [1, 1] {
            return Err(GraphError::ShapeMismatch {
                expected: vec![1, 1],
                got: scalar_shape.to_vec(),
                message: "ScalarMultiply的第1个父节点必须是标量(形状为[1,1])".to_string(),
            });
        }

        // 3. 获取第2个父节点（矩阵）的形状作为输出形状
        let matrix_shape = parents[1].value_expected_shape().to_vec();

        // 4. 返回
        Ok(Self {
            id: None,
            name: None,
            value: None,
            jacobi: None,
            shape: matrix_shape,
            parents_ids: vec![parents[0].id(), parents[1].id()],
        })
    }
}

impl TraitNode for ScalarMultiply {
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
        // 1. 获取标量值
        let scalar_value = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的第1个父节点（标量）{}没有值",
                self.display_node(),
                parents[0]
            ))
        })?;

        // 2. 获取矩阵值
        let matrix_value = parents[1].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的第2个父节点（矩阵）{}没有值",
                self.display_node(),
                parents[1]
            ))
        })?;

        // 3. 提取标量数值并进行乘法
        let scalar = scalar_value.get_data_number().ok_or_else(|| {
            GraphError::ComputationError(format!("{}的第1个父节点不是标量", self.display_node()))
        })?;

        // 4. 计算结果：标量 * 矩阵
        self.value = Some(scalar * matrix_value);

        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// 计算ScalarMultiply节点对父节点的雅可比矩阵
    /// 参考MatrixSlow: MatrixSlow/matrixslow/ops/ops.py#L336 (ScalarMultiply.get_jacobi)
    ///
    /// 设 C = s * M，其中s是标量(1x1)，M是矩阵(m,n)
    ///
    /// 对于标量 s（第1个父节点）：
    ///   ∂C/∂s = M.flatten().T  → shape: [m*n, 1]
    ///
    /// 对于矩阵 M（第2个父节点）：
    ///   ∂C/∂M = s * I_{m*n}    → shape: [m*n, m*n]
    fn calc_jacobi_to_a_parent(
        &self,
        target_parent: &NodeHandle,
        assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // 获取两个父节点的值
        let (scalar_value, matrix_value) = if target_parent.id() == self.parents_ids[0] {
            // target是标量，assistant是矩阵
            let assistant = assistant_parent.ok_or_else(|| {
                GraphError::ComputationError(
                    "ScalarMultiply计算雅可比矩阵需要辅助父节点".to_string(),
                )
            })?;
            (
                target_parent.value().ok_or_else(|| {
                    GraphError::ComputationError(format!(
                        "{}的标量父节点没有值",
                        self.display_node()
                    ))
                })?,
                assistant.value().ok_or_else(|| {
                    GraphError::ComputationError(format!(
                        "{}的矩阵父节点没有值",
                        self.display_node()
                    ))
                })?,
            )
        } else if target_parent.id() == self.parents_ids[1] {
            // target是矩阵，assistant是标量
            let assistant = assistant_parent.ok_or_else(|| {
                GraphError::ComputationError(
                    "ScalarMultiply计算雅可比矩阵需要辅助父节点".to_string(),
                )
            })?;
            (
                assistant.value().ok_or_else(|| {
                    GraphError::ComputationError(format!(
                        "{}的标量父节点没有值",
                        self.display_node()
                    ))
                })?,
                target_parent.value().ok_or_else(|| {
                    GraphError::ComputationError(format!(
                        "{}的矩阵父节点没有值",
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
            // 对标量的雅可比：∂C/∂s = M.flatten().T → shape: [m*n, 1]
            // 将矩阵展平并reshape为列向量
            let size = matrix_value.size();
            let flattened = matrix_value.flatten().reshape(&[size, 1]);
            Ok(flattened)
        } else {
            // 对矩阵的雅可比：∂C/∂M = s * I_{m*n} → shape: [m*n, m*n]
            let scalar = scalar_value
                .get_data_number()
                .ok_or_else(|| GraphError::ComputationError("标量父节点不是1x1矩阵".to_string()))?;
            let size = matrix_value.size();
            Ok(Tensor::eyes(size) * scalar)
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
