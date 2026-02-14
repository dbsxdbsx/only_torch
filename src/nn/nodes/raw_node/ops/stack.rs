use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Stack 节点：沿新维度堆叠多个张量（类似 `torch.stack`）
///
/// 在 `axis` 位置插入新维度，所有父节点形状必须完全相同。
#[derive(Clone)]
pub(crate) struct Stack {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 插入新维度的位置
    axis: usize,
    /// 父节点 ID 列表
    #[allow(dead_code)]
    parent_ids: Vec<NodeId>,
    /// 父节点数量（backward 时按 axis 维度逐个 select）
    #[allow(dead_code)]
    num_parents: usize,
    /// 固定形状
    fixed_shape: Vec<usize>,
    /// 动态形状
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    #[allow(dead_code)]
    supports_dynamic: bool,
}

impl Stack {
    /// 获取操作的轴
    #[allow(dead_code)]
    pub(crate) const fn axis(&self) -> usize {
        self.axis
    }

    /// 从父节点形状信息创建 Stack 节点
    pub(in crate::nn) fn new(
        parent_shapes: &[&[usize]],
        parent_dynamic_shapes: &[DynamicShape],
        parent_ids: Vec<NodeId>,
        axis: usize,
    ) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parent_shapes.is_empty() {
            return Err(GraphError::InvalidOperation(
                "Stack 节点至少需要 1 个父节点".to_string(),
            ));
        }

        let first_shape = parent_shapes[0];
        let ndim = first_shape.len();

        // 2. 验证 axis（stack 模式 axis 可以等于 ndim）
        if axis > ndim {
            return Err(GraphError::InvalidOperation(format!(
                "Stack: axis {axis} 超出有效范围 [0, {ndim}]"
            )));
        }

        // 3. 验证所有父节点形状完全相同
        for (i, shape) in parent_shapes.iter().enumerate().skip(1) {
            if *shape != first_shape {
                return Err(GraphError::ShapeMismatch {
                    expected: first_shape.to_vec(),
                    got: shape.to_vec(),
                    message: format!("Stack: 父节点 {i} 形状不一致"),
                });
            }
        }

        let num_parents = parent_shapes.len();

        // 4. 计算输出形状：在 axis 位置插入 num_parents
        let mut fixed_shape = first_shape.to_vec();
        fixed_shape.insert(axis, num_parents);

        // 5. 计算动态形状
        let supports_dynamic = parent_dynamic_shapes.iter().any(|d| d.has_dynamic_dims());
        let first_dyn = &parent_dynamic_shapes[0];

        let mut dims: Vec<Option<usize>> = Vec::with_capacity(fixed_shape.len());
        for (i, &size) in fixed_shape.iter().enumerate() {
            if i == axis {
                dims.push(Some(size));
            } else {
                let orig_idx = if i < axis { i } else { i - 1 };
                dims.push(first_dyn.dim(orig_idx));
            }
        }
        let dynamic_shape = DynamicShape::new(&dims);

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            axis,
            parent_ids,
            num_parents,
            fixed_shape,
            dynamic_shape,
            supports_dynamic,
        })
    }
}

impl TraitNode for Stack {
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

    fn dedup_fingerprint(&self) -> Option<u64> {
        Some(self.axis as u64)
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        self.value = Some(Tensor::stack(parent_values, self.axis));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_grad_to_parent(
        &self,
        target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        // stack 反向：沿 axis 维度 select 第 idx 个切片
        Ok(GradResult::Computed(upstream_grad.select(self.axis, target_parent_index)))
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
