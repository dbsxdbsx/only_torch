use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::GraphError;
use crate::tensor::Tensor;

#[derive(Clone)]
pub(crate) struct MatMul {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    trainable: bool,
    shape: Vec<usize>,
    parents_ids: Vec<NodeId>, // 保留这个字段用于雅可比计算，注意元素的存储顺序
}

impl MatMul {
    pub(crate) fn new(parents: &[&NodeHandle], trainable: bool) -> Result<Self, GraphError> {
        // 1. 必要的验证
        // 1.1 父节点数量验证
        if parents.len() != 2 {
            return Err(GraphError::InvalidOperation(
                "MatMul节点需要正好2个父节点".to_string(),
            ));
        }

        // 1.2 验证矩阵乘法的形状兼容性
        let parent1_shape = parents[0].value_expected_shape();
        let parent2_shape = parents[1].value_expected_shape();
        if parent1_shape[1] != parent2_shape[0] {
            return Err(GraphError::ShapeMismatch {
                expected: vec![parent1_shape[0], parent2_shape[1]],
                got: vec![parent1_shape[1], parent2_shape[0]],
                message: format!(
                    "MatMul节点的2个父节点形状不兼容：父节点1的列数({})与父节点2的行数({})不相等。",
                    parent1_shape[1], parent2_shape[0],
                ),
            });
        }

        // 2. 返回
        Ok(Self {
            id: None,
            name: None,
            value: None,
            jacobi: None,
            trainable,
            shape: vec![parent1_shape[0], parent2_shape[1]],
            parents_ids: vec![parents[0].id(), parents[1].id()],
        })
    }
}

impl TraitNode for MatMul {
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
        // 1. 获取父节点的值
        let parent1_value = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的第1个父{}没有值。不该触及本错误，否则说明crate代码有问题",
                self.display_node(),
                parents[0]
            ))
        })?;
        let parent2_value = parents[1].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的第2个父{}没有值。不该触及本错误，否则说明crate代码有问题",
                self.display_node(),
                parents[1]
            ))
        })?;

        // 2. 计算结果
        self.value = Some(parent1_value.mat_mul(parent2_value));

        // 3. 返回
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
        assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // 1. 先获取父节点的a值和b值
        let (a_value, b_value) = if target_parent.id() == self.parents_ids[0]
            && assistant_parent.unwrap().id() == self.parents_ids[1]
        {
            (
                target_parent.value().ok_or_else(|| {
                    GraphError::ComputationError(format!(
                        "{}的第1个父{}没有值。不该触及本错误，否则说明crate代码有问题",
                        self.display_node(),
                        target_parent
                    ))
                })?,
                assistant_parent.unwrap().value().ok_or_else(|| {
                    GraphError::ComputationError(format!(
                        "{}的第2个父{}没有值。不该触及本错误，否则说明crate代码有问题",
                        self.display_node(),
                        assistant_parent.unwrap()
                    ))
                })?,
            )
        } else if target_parent.id() == self.parents_ids[1]
            && assistant_parent.unwrap().id() == self.parents_ids[0]
        {
            (
                assistant_parent.unwrap().value().ok_or_else(|| {
                    GraphError::ComputationError(format!(
                        "{}的第1个父{}没有值。不该触及本错误，否则说明crate代码有问题",
                        self.display_node(),
                        assistant_parent.unwrap()
                    ))
                })?,
                target_parent.value().ok_or_else(|| {
                    GraphError::ComputationError(format!(
                        "{}的第2个父{}没有值。不该触及本错误，否则说明crate代码有问题",
                        self.display_node(),
                        target_parent
                    ))
                })?,
            )
        } else {
            return Err(GraphError::ComputationError(format!(
                "{}或{}不是当前{}的父节点。不该触及本错误，否则说明crate代码有问题",
                target_parent,
                assistant_parent.unwrap(),
                self.display_node()
            )));
        };

        // 2. 根据父节点位置计算雅可比矩阵
        if target_parent.id() == self.parents_ids[0] {
            // 计算 dC/dA
            let m = a_value.shape()[0];
            let n = a_value.shape()[1];
            let p = b_value.shape()[1];

            let mut jacobi = Tensor::zeros(&[m * p, m * n]);

            for i in 0..m {
                for j in 0..p {
                    for k in 0..n {
                        // ∂C[i][j]/∂A[i][k] = B[k][j]
                        jacobi[[i * p + j, i * n + k]] = b_value[[k, j]];
                    }
                }
            }

            Ok(jacobi)
        } else {
            // 计算 dC/dB
            let m = a_value.shape()[0];
            let n = b_value.shape()[0];
            let p = b_value.shape()[1];

            let mut jacobi = Tensor::zeros(&[m * p, n * p]);

            for i in 0..m {
                for j in 0..p {
                    for k in 0..n {
                        // ∂C[i][j]/∂B[k][j] = A[i][k]
                        jacobi[[i * p + j, k * p + j]] = a_value[[i, k]];
                    }
                }
            }

            Ok(jacobi)
        }
    }
    // fn calc_jacobi_to_a_parent(&self, target_parent: &NodeHandle) -> Result<Tensor, GraphError> {
    //     // 获取当前节点的值(已在前向传播时计算)
    //     let value = self.value().ok_or_else(|| {
    //         GraphError::ComputationError(format!(
    //             "{}没有值。不该触及本错误，否则说明crate代码有问题",
    //             self.display_node()
    //         ))
    //     })?;

    //     // 根据父节点位置计算雅可比矩阵
    //     if target_parent.id() == self.parents_ids[0] {
    //         // 对于矩阵乘法 C = AB，计算 dC/dA
    //         let m = value.shape()[0];
    //         let n = target_parent.value().unwrap().shape()[1];
    //         let p = value.shape()[1];

    //         let mut jacobi = Tensor::zeros(&[m * p, m * n]);
    //         for i in 0..m {
    //             for j in 0..p {
    //                 for k in 0..n {
    //                     jacobi[[i * p + j, i * n + k]] = value[[k, j]];
    //                 }
    //             }
    //         }
    //         Ok(jacobi)
    //     } else if target_parent.id() == self.parents_ids[1] {
    //         // 对于矩阵乘法 C = AB，计算 dC/dB
    //         let m = value.shape()[0];
    //         let n = target_parent.value().unwrap().shape()[0];
    //         let p = value.shape()[1];

    //         let mut jacobi = Tensor::zeros(&[m * p, n * p]);
    //         for i in 0..m {
    //             for j in 0..p {
    //                 for k in 0..n {
    //                     jacobi[[i * p + j, k * p + j]] = value[[i, k]];
    //                 }
    //             }
    //         }
    //         Ok(jacobi)
    //     } else {
    //         Err(GraphError::ComputationError(format!(
    //             "{}不是当前节点{}的父节点。不该触及本错误，否则说明crate代码有问题",
    //             target_parent,
    //             self.display_node()
    //         )))
    //     }
    // }

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
