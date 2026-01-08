use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::tensor::Tensor;

#[derive(Clone)]
pub(crate) struct MatMul {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    shape: Vec<usize>,
    parents_ids: Vec<NodeId>, // 用于区分左右父节点
}

impl MatMul {
    pub(crate) fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
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
            grad: None,
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

    /// `MatMul` 的 VJP 梯度计算
    ///
    /// 对于 C = A @ B（A: [batch, n], B: [n, k], C: [batch, k]）：
    /// - dL/dA = `upstream_grad` @ B^T，shape: [batch, k] @ [k, n] = [batch, n]
    /// - dL/dB = A^T @ `upstream_grad，shape`: [n, batch] @ [batch, k] = [n, k]
    ///           这个乘法自然地对 batch 维度求和
    fn calc_grad_to_parent(
        &self,
        target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        let assistant = assistant_parent.ok_or_else(|| {
            GraphError::ComputationError("MatMul 需要辅助父节点来计算梯度".to_string())
        })?;

        // 获取父节点的值
        let (a_value, b_value) = if target_parent.id() == self.parents_ids[0] {
            // target 是左父节点 A
            (
                target_parent.value().ok_or_else(|| {
                    GraphError::ComputationError(format!("{}的左父节点没有值", self.display_node()))
                })?,
                assistant.value().ok_or_else(|| {
                    GraphError::ComputationError(format!("{}的右父节点没有值", self.display_node()))
                })?,
            )
        } else {
            // target 是右父节点 B
            (
                assistant.value().ok_or_else(|| {
                    GraphError::ComputationError(format!("{}的左父节点没有值", self.display_node()))
                })?,
                target_parent.value().ok_or_else(|| {
                    GraphError::ComputationError(format!("{}的右父节点没有值", self.display_node()))
                })?,
            )
        };

        if target_parent.id() == self.parents_ids[0] {
            // 计算 dL/dA = upstream_grad @ B^T
            // upstream_grad: [batch, k], B: [n, k] -> B^T: [k, n]
            // 结果: [batch, n]
            Ok(upstream_grad.mat_mul(&b_value.transpose()))
        } else {
            // 计算 dL/dB = A^T @ upstream_grad
            // A: [batch, n] -> A^T: [n, batch]
            // upstream_grad: [batch, k]
            // 结果: [n, k]（自然对 batch 求和）
            Ok(a_value.transpose().mat_mul(upstream_grad))
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
