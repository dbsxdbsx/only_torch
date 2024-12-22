use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::NodeHandle;
use crate::nn::{GraphError, NodeId};
use crate::tensor::Tensor;

#[derive(Clone)]
pub(crate) struct MatMul {
    name: String,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    trainable: bool,
    parents_ids: Vec<NodeId>, // NOTE: 注意顺序
}

impl MatMul {
    pub(crate) fn new(name: &str, trainable: bool, parents_ids: &[NodeId]) -> Self {
        Self {
            name: name.to_string(),
            value: None,
            jacobi: None,
            trainable,
            parents_ids: parents_ids.to_vec(),
        }
    }
}

impl TraitNode for MatMul {
    fn name(&self) -> &str {
        &self.name
    }

    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
        // 1. 获取父节点的值
        let parent1_value = parents[0]
            .value()
            .ok_or_else(|| GraphError::ComputationError("第一个父节点没有值".to_string()))?;
        let parent2_value = parents[1]
            .value()
            .ok_or_else(|| GraphError::ComputationError("第二个父节点没有值".to_string()))?;

        // 2. 验证矩阵乘法的形状兼容性
        if parent1_value.shape()[1] != parent2_value.shape()[0] {
            return Err(GraphError::ShapeMismatch {
                expected: vec![parent1_value.shape()[0], parent2_value.shape()[1]],
                got: vec![parent1_value.shape()[1], parent2_value.shape()[0]],
                message: format!(
                    "MatMul节点 '{}' 的两个父节点形状不兼容：父节点1的列数({})与父节点2的行数({})不相等。",
                    self.name(),
                    parent1_value.shape()[1],
                    parent2_value.shape()[0],
                ),
            });
        }

        // 3. 计算结果
        self.value = Some(parent1_value.mat_mul(parent2_value));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// NOTE: 这里的逻辑本想取巧参考：https://github.com/zc911/MatrixSlow/blob/a6db0d38802004449941e6644e609a2455b26327/matrixslow/ops/ops.py#L61
    /// 但太难懂了，所以还是用最原始的实现吧
    fn calc_jacobi_to_a_parent(
        &self,
        target_parent: &NodeHandle,
        another_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        let other = another_parent.ok_or_else(|| {
            GraphError::ComputationError("MatMul需要另一个父节点的值".to_string())
        })?;

        let parent1_value = target_parent
            .value()
            .ok_or_else(|| GraphError::ComputationError("第一个父节点没有值".to_string()))?;
        let parent2_value = other
            .value()
            .ok_or_else(|| GraphError::ComputationError("第二个父节点没有值".to_string()))?;

        // 根据父节点位置计算雅可比矩阵
        if target_parent.id() == self.parents_ids[0] {
            // 对于矩阵乘法 C = AB，计算 dC/dA，需要用到B的值
            let m = parent1_value.shape()[0];
            let n = parent1_value.shape()[1];
            let p = parent2_value.shape()[1];

            let mut jacobi = Tensor::zeros(&[m * p, m * n]);
            for i in 0..m {
                for j in 0..p {
                    for k in 0..n {
                        jacobi[[i * p + j, i * n + k]] = parent2_value[[k, j]];
                    }
                }
            }
            Ok(jacobi)
        } else if target_parent.id() == self.parents_ids[1] {
            // 对于矩阵乘法 C = AB，计算 dC/dB，需要用到A的值
            let m = parent1_value.shape()[0];
            let n = parent2_value.shape()[0];
            let p = parent2_value.shape()[1];

            let mut jacobi = Tensor::zeros(&[m * p, n * p]);
            for i in 0..m {
                for j in 0..p {
                    for k in 0..n {
                        jacobi[[i * p + j, k * p + j]] = parent1_value[[i, k]];
                    }
                }
            }
            Ok(jacobi)
        } else {
            Err(GraphError::ComputationError(format!(
                "节点id `{:?}` 不是当前节点的父节点id `{:?}` 或 `{:?}`",
                target_parent.id(),
                self.parents_ids[0],
                self.parents_ids[1]
            )))
        }
    }

    fn jacobi(&self) -> Option<&Tensor> {
        self.jacobi.as_ref()
    }

    fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError> {
        self.jacobi = jacobi.map(|j| j.clone());
        Ok(())
    }

    fn is_trainable(&self) -> bool {
        self.trainable
    }

    fn set_trainable(&mut self, trainable: bool) -> Result<(), GraphError> {
        self.trainable = trainable;
        Ok(())
    }
}
