use crate::nn::{Graph, GraphError, NodeHandle, NodeId, TraitNode};
use crate::tensor::Tensor;

pub struct MatMul {
    name: String,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    parents: Vec<NodeId>,
    children: Vec<NodeId>,
}

impl MatMul {
    pub fn new(parents: &[NodeId], name: Option<&str>) -> Self {
        // 1. 基本验证：矩阵乘法需要恰好两个父节点
        assert!(parents.len() == 2, "MatMul节点需恰好2个父节点");

        // 2. 返回实例
        Self {
            name: name.unwrap_or_default().to_string(),
            value: None,
            jacobi: None,
            parents: parents.to_vec(),
            children: Vec::new(),
        }
    }
}

impl TraitNode for MatMul {
    fn name(&self) -> &str {
        &self.name
    }

    fn compute_value(&mut self, parents: &[&NodeHandle]) -> Result<(), GraphError> {
        // 1. 获取父节点的值
        let parent1_value = graph.get_node_value(self.parents[0])?;
        let parent2_value = graph.get_node_value(self.parents[1])?;

        // 2. 验证输入维度
        if parent1_value.shape().len() != 2 || parent2_value.shape().len() != 2 {
            return Err(GraphError::ComputationError(
                "MatMul节点的输入必须是2阶张量".to_string(),
            ));
        }

        // 3. 验证矩阵乘法的形状兼容性
        if parent1_value.shape()[1] != parent2_value.shape()[0] {
            return Err(GraphError::ShapeMismatch {
                expected: vec![parent1_value.shape()[0], parent2_value.shape()[1]],
                got: vec![parent1_value.shape()[1], parent2_value.shape()[0]],
            });
        }

        // 4. 计算结果
        self.value = Some(parent1_value.mat_mul(parent2_value));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// NOTE: 这里的逻辑本想取巧参考：https://github.com/zc911/MatrixSlow/blob/a6db0d38802004449941e6644e609a2455b26327/matrixslow/ops/ops.py#L61
    /// 但发现太难懂了，所以还是用最原始的实现吧
    fn calc_jacobi_to_a_parent(&self, parent: &NodeHandle) -> Result<Tensor, GraphError> {
        // 1. 首先验证输入的节点是否为父节点
        if parent_id != self.parents[0] && parent_id != self.parents[1] {
            return Err(GraphError::InvalidOperation(
                "输入的节点不是该MatMul节点的父节点",
            ));
        }

        // 2. 根据计算目标父节点计算雅可比矩阵
        let parent1_value = graph.get_node_value(self.parents[0])?;
        let parent2_value = graph.get_node_value(self.parents[1])?;

        if parent_id == self.parents[0] {
            // 对于矩阵乘法 C = AB，计算 dC/dA
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
        } else {
            // 对于矩阵乘法 C = AB，计算 dC/dB
            let m = parent1_value.shape()[0];
            let n = parent1_value.shape()[1];
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
        }
    }

    fn jacobi(&self) -> Option<&Tensor> {
        self.jacobi.as_ref()
    }

    fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError> {
        self.jacobi = jacobi.map(|j| j.clone());
        Ok(())
    }

    fn parents_ids(&self) -> &[NodeId] {
        &self.parents
    }

    fn children_ids(&self) -> &[NodeId] {
        &self.children
    }

    fn is_trainable(&self) -> bool {
        true
    }
}
