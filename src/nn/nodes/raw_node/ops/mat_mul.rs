use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::raw_node::GradResult;
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
    #[allow(dead_code)]
    supports_dynamic: bool,
    #[allow(dead_code)]
    parents_ids: Vec<NodeId>, // 用于区分左右父节点
}

impl MatMul {
    /// 从父节点形状信息创建 MatMul 节点（核心实现）
    ///
    /// # 参数
    /// - `parent_shapes`: 父节点的固定形状 [left, right]
    /// - `parent_dynamic_shapes`: 父节点的动态形状
    /// - `parent_ids`: 父节点 ID（用于梯度计算时区分左右）
    pub(in crate::nn) fn new(
        parent_shapes: &[&[usize]],
        parent_dynamic_shapes: &[DynamicShape],
        parent_ids: Vec<NodeId>,
    ) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parent_shapes.len() != 2 {
            return Err(GraphError::InvalidOperation(
                "MatMul节点需要正好2个父节点".to_string(),
            ));
        }
        if parent_shapes.len() != parent_dynamic_shapes.len() {
            return Err(GraphError::InvalidOperation(
                "父节点形状数量与动态形状数量不匹配".to_string(),
            ));
        }

        let parent1_fixed = parent_shapes[0];
        let parent2_fixed = parent_shapes[1];

        // 2. 验证矩阵乘法的形状兼容性
        // parent1[1] 必须等于 parent2[0]
        let parent1_cols = parent1_fixed[1];
        let parent2_rows = parent2_fixed[0];

        if parent1_cols != parent2_rows {
            return Err(GraphError::ShapeMismatch {
                expected: vec![parent1_fixed[0], parent2_fixed[1]],
                got: vec![parent1_cols, parent2_rows],
                message: format!(
                    "MatMul节点的2个父节点形状不兼容：父节点1的列数({parent1_cols})与父节点2的行数({parent2_rows})不相等。",
                ),
            });
        }

        // 3. 计算输出形状
        let parent1_dyn = &parent_dynamic_shapes[0];
        let parent2_dyn = &parent_dynamic_shapes[1];

        let supports_dynamic = parent1_dyn.has_dynamic_dims();
        let output_batch = parent1_dyn.dim(0);
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
            parents_ids: parent_ids,
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

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        // 计算矩阵乘法
        self.value = Some(parent_values[0].mat_mul(parent_values[1]));
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
        target_parent_index: usize,
        parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        // 获取两个父节点的值
        let a_value = parent_values.get(0).ok_or_else(|| {
            GraphError::ComputationError(format!("{}的左父节点没有值", self.display_node()))
        })?;
        let b_value = parent_values.get(1).ok_or_else(|| {
            GraphError::ComputationError(format!("{}的右父节点没有值", self.display_node()))
        })?;

        if target_parent_index == 0 {
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
            Ok(GradResult::Computed(upstream_grad.mat_mul(&b_t)))
        } else if target_parent_index == 1 {
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
            Ok(GradResult::Computed(a_t.mat_mul(upstream_grad)))
        } else {
            Err(GraphError::ComputationError(format!(
                "MatMul 节点只有 2 个父节点，索引 {} 无效",
                target_parent_index
            )))
        }
    }

    fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
    }

    fn grad_mut(&mut self) -> Option<&mut Tensor> {
        self.grad.as_mut()
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
