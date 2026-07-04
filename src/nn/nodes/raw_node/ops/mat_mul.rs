use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
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
    /// 支持两种形态（两个父节点必须同秩，不做混秩隐式广播）：
    /// - 2D：`[m, k] @ [k, n] → [m, n]`
    /// - 3D 批量：`[B, m, k] @ [B, k, n] → [B, m, n]`（batch 维严格相等）
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

        match (parent1_fixed.len(), parent2_fixed.len()) {
            // ── 2D 矩阵乘法 ──
            (2, 2) => {
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

            // ── 3D 批量矩阵乘法（bmm）──
            (3, 3) => {
                let (b1, m, k_a) = (parent1_fixed[0], parent1_fixed[1], parent1_fixed[2]);
                let (b2, k_b, n) = (parent2_fixed[0], parent2_fixed[1], parent2_fixed[2]);

                if b1 != b2 {
                    return Err(GraphError::ShapeMismatch {
                        expected: parent1_fixed.to_vec(),
                        got: parent2_fixed.to_vec(),
                        message: format!(
                            "批量 MatMul 的 batch 维必须严格相等（{b1} vs {b2}），本框架不做跨 batch 隐式广播。",
                        ),
                    });
                }
                if k_a != k_b {
                    return Err(GraphError::ShapeMismatch {
                        expected: vec![b1, m, n],
                        got: vec![k_a, k_b],
                        message: format!(
                            "批量 MatMul 内维不匹配：父节点1的列数({k_a})与父节点2的行数({k_b})不相等。",
                        ),
                    });
                }

                let fixed_shape = vec![b1, m, n];
                // 3D 批量路径按静态形状建图（attention 等场景 NH/T 均为编译期已知），
                // 不参与动态 batch 机制。
                let dyn_dims: Vec<Option<usize>> = fixed_shape.iter().map(|&d| Some(d)).collect();
                let dynamic_shape = DynamicShape::new(&dyn_dims);

                Ok(Self {
                    id: None,
                    name: None,
                    value: None,
                    grad: None,
                    fixed_shape,
                    dynamic_shape,
                    supports_dynamic: false,
                    parents_ids: parent_ids,
                })
            }

            _ => Err(GraphError::InvalidOperation(format!(
                "MatMul 仅支持 2D@2D 或 3D@3D（批量），得到 {}D @ {}D",
                parent1_fixed.len(),
                parent2_fixed.len()
            ))),
        }
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
        // 计算矩阵乘法（3D 走批量 GEMM，2D 走普通 GEMM）
        let a = parent_values[0];
        let b = parent_values[1];
        self.value = Some(if a.dimension() == 3 {
            a.batched_mat_mul(b)
        } else {
            a.mat_mul(b)
        });
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// `MatMul` 的 VJP 梯度计算
    ///
    /// 2D，对于 C = A @ B（A: [batch, n], B: [n, k], C: [batch, k]）：
    /// - dL/dA = `upstream_grad` @ B^T，shape: [batch, k] @ [k, n] = [batch, n]
    /// - dL/dB = A^T @ `upstream_grad，shape`: [n, batch] @ [batch, k] = [n, k]
    ///   这个乘法自然地对 batch 维度求和
    ///
    /// 3D 批量（C = A @ B，A: [B, m, k], B: [B, k, n]）为逐 batch 的同款公式，
    /// 无跨 batch 广播因此**无需**任何求和归约：
    /// - dL/dA = `upstream_grad` bmm B^T → [B, m, k]
    /// - dL/dB = A^T bmm `upstream_grad` → [B, k, n]
    fn calc_grad_to_parent(
        &self,
        target_parent_index: usize,
        parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        // 获取两个父节点的值
        let a_value = parent_values.first().ok_or_else(|| {
            GraphError::ComputationError(format!("{}的左父节点没有值", self.display_node()))
        })?;
        let b_value = parent_values.get(1).ok_or_else(|| {
            GraphError::ComputationError(format!("{}的右父节点没有值", self.display_node()))
        })?;

        let batched = a_value.dimension() == 3;

        if target_parent_index == 0 {
            // 计算 dL/dA = upstream_grad @ B^T
            // 2D: upstream_grad: [batch, k], B: [n, k] -> B^T: [k, n]，结果 [batch, n]
            // mat_mul_nt / batched_mat_mul_nt 以转置视图参与（仅翻转 stride 元数据），
            // 免去旧 `transpose()` 对 B 的整块物化拷贝。
            let (up_cols, b_cols) = if batched {
                (upstream_grad.shape()[2], b_value.shape()[2])
            } else {
                (upstream_grad.shape()[1], b_value.shape()[1])
            };
            if up_cols != b_cols {
                return Err(GraphError::ShapeMismatch {
                    expected: vec![upstream_grad.shape()[0], b_value.shape()[0]],
                    got: vec![up_cols, b_cols],
                    message: format!(
                        "MatMul ({}) dL/dA 形状不匹配: upstream_grad {:?} @ B^T (B={:?})",
                        self.display_node(),
                        upstream_grad.shape(),
                        b_value.shape()
                    ),
                });
            }
            Ok(GradResult::Computed(if batched {
                upstream_grad.batched_mat_mul_nt(b_value)
            } else {
                upstream_grad.mat_mul_nt(b_value)
            }))
        } else if target_parent_index == 1 {
            // 计算 dL/dB = A^T @ upstream_grad
            // 2D: A: [batch, n] -> A^T: [n, batch]，upstream_grad: [batch, k]，
            //     结果 [n, k]（自然对 batch 求和）
            // mat_mul_tn / batched_mat_mul_tn 以转置视图参与，免去物化拷贝。
            let (a_rows, up_rows) = if batched {
                (a_value.shape()[1], upstream_grad.shape()[1])
            } else {
                (a_value.shape()[0], upstream_grad.shape()[0])
            };
            if a_rows != up_rows {
                return Err(GraphError::ShapeMismatch {
                    expected: vec![a_value.shape()[1], upstream_grad.shape()[1]],
                    got: vec![a_rows, up_rows],
                    message: format!(
                        "MatMul ({}) dL/dB 形状不匹配: A^T (A={:?}) @ upstream_grad {:?}",
                        self.display_node(),
                        a_value.shape(),
                        upstream_grad.shape()
                    ),
                });
            }
            Ok(GradResult::Computed(if batched {
                a_value.batched_mat_mul_tn(upstream_grad)
            } else {
                a_value.mat_mul_tn(upstream_grad)
            }))
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
