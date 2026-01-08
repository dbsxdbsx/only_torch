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
    grad: Option<Tensor>,
    shape: Vec<usize>,
    parents_ids: Vec<NodeId>, // 用于区分标量和矩阵父节点
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
            grad: None,
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

    /// 计算 `ScalarMultiply` 节点对父节点的梯度（VJP）
    ///
    /// 对于 C = s * M，其中 s 是标量，M 是矩阵：
    /// - ∂L/∂M = s * `upstream_grad`
    /// - ∂L/∂s = `sum(upstream_grad` ⊙ M) → 形状 [1, 1]
    fn calc_grad_to_parent(
        &self,
        target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // 获取辅助父节点
        let assistant = assistant_parent.ok_or_else(|| {
            GraphError::ComputationError("ScalarMultiply 节点计算梯度需要辅助父节点".to_string())
        })?;

        if target_parent.id() == self.parents_ids[0] {
            // target 是标量 s，assistant 是矩阵 M
            // ∂L/∂s = sum(upstream_grad ⊙ M)
            let m_value = assistant.value().ok_or_else(|| {
                GraphError::ComputationError(format!("{} 的矩阵父节点没有值", self.display_node()))
            })?;
            // 逐元素乘积后求和
            let elementwise_product = upstream_grad * m_value;
            let sum: f32 = elementwise_product.data_as_slice().iter().sum();
            Ok(Tensor::new(&[sum], &[1, 1]))
        } else if target_parent.id() == self.parents_ids[1] {
            // target 是矩阵 M，assistant 是标量 s
            // ∂L/∂M = s * upstream_grad
            let s_value = assistant.value().ok_or_else(|| {
                GraphError::ComputationError(format!("{} 的标量父节点没有值", self.display_node()))
            })?;
            let scalar = s_value.get_data_number().ok_or_else(|| {
                GraphError::ComputationError("标量父节点不是 1x1 矩阵".to_string())
            })?;
            Ok(scalar * upstream_grad)
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
