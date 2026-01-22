use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

#[derive(Clone)]
pub(crate) struct MatMul {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 固定形状（用于 `value_expected_shape`）
    fixed_shape: Vec<usize>,
    /// 动态形状（支持动态 batch）
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch（继承自父节点）
    supports_dynamic: bool,
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
        let parent1_dyn = parents[0].dynamic_expected_shape();
        let parent2_dyn = parents[1].dynamic_expected_shape();
        let parent1_fixed = parents[0].value_expected_shape();
        let parent2_fixed = parents[1].value_expected_shape();

        // 获取内层维度进行验证：parent1[1] 必须等于 parent2[0]
        // 注意：必须使用 fixed_shape 验证内层维度，因为：
        // - 对于 Input 节点，dynamic_shape 的 dim(0) 是 None（动态 batch）
        // - 但矩阵乘法的内层维度必须是确定值且必须匹配
        let parent1_cols = parent1_fixed[1]; // parent1 的列数
        let parent2_rows = parent2_fixed[0]; // parent2 的行数

        // 验证内层维度兼容性
        if parent1_cols != parent2_rows {
            return Err(GraphError::ShapeMismatch {
                expected: vec![parent1_fixed[0], parent2_fixed[1]],
                got: vec![parent1_cols, parent2_rows],
                message: format!(
                    "MatMul节点的2个父节点形状不兼容：父节点1的列数({parent1_cols})与父节点2的行数({parent2_rows})不相等。",
                ),
            });
        }

        // 2. 计算输出形状
        // 输出形状 = [parent1的batch/行, parent2的列]
        // 如果 parent1 支持动态 batch，输出也支持
        let supports_dynamic = parents[0].supports_dynamic_batch();
        let output_batch = parent1_dyn.dim(0); // 可能是 None（动态）或 Some(n)
        let output_cols = parent2_dyn.dim(1).or(Some(parent2_fixed[1]));

        let dynamic_shape = DynamicShape::new(&[output_batch, output_cols]);
        let fixed_shape = vec![parent1_fixed[0], parent2_fixed[1]];

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape,
            dynamic_shape,
            supports_dynamic,
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
        &self.fixed_shape
    }

    fn dynamic_expected_shape(&self) -> DynamicShape {
        self.dynamic_shape.clone()
    }

    fn supports_dynamic_batch(&self) -> bool {
        self.supports_dynamic
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
            let b_t = b_value.transpose();
            if upstream_grad.shape()[1] != b_t.shape()[0] {
                return Err(GraphError::ShapeMismatch {
                    expected: vec![upstream_grad.shape()[0], b_t.shape()[1]],
                    got: vec![upstream_grad.shape()[1], b_t.shape()[0]],
                    message: format!(
                        "MatMul ({}) dL/dA 形状不匹配: upstream_grad {:?} @ B^T {:?}",
                        self.display_node(),
                        upstream_grad.shape(),
                        b_t.shape()
                    ),
                });
            }
            Ok(upstream_grad.mat_mul(&b_t))
        } else {
            // 计算 dL/dB = A^T @ upstream_grad
            // A: [batch, n] -> A^T: [n, batch]
            // upstream_grad: [batch, k]
            // 结果: [n, k]（自然对 batch 求和）
            let a_t = a_value.transpose();
            if a_t.shape()[1] != upstream_grad.shape()[0] {
                return Err(GraphError::ShapeMismatch {
                    expected: vec![a_t.shape()[0], upstream_grad.shape()[1]],
                    got: vec![a_t.shape()[1], upstream_grad.shape()[0]],
                    message: format!(
                        "MatMul ({}) dL/dB 形状不匹配: A^T {:?} (A={:?}) @ upstream_grad {:?}",
                        self.display_node(),
                        a_t.shape(),
                        a_value.shape(),
                        upstream_grad.shape()
                    ),
                });
            }
            Ok(a_t.mat_mul(upstream_grad))
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
