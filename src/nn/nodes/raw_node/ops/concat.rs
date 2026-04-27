use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Concat 节点：沿现有维度拼接多个张量（类似 `torch.cat` / `tf.concat`）
///
/// 沿 `axis` 轴拼接，该轴大小可以不同，但其他维度必须相同。
#[derive(Clone)]
pub(crate) struct Concat {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 沿哪个轴拼接
    axis: usize,
    /// 父节点 ID 列表
    #[allow(dead_code)]
    parent_ids: Vec<NodeId>,
    /// 各父节点在 axis 维度的大小（用于 backward 时按偏移分段）
    parent_sizes: Vec<usize>,
    /// 固定形状
    fixed_shape: Vec<usize>,
    /// 动态形状
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    #[allow(dead_code)]
    supports_dynamic: bool,
}

impl Concat {
    /// 获取操作的轴
    #[allow(dead_code)]
    pub(crate) const fn axis(&self) -> usize {
        self.axis
    }

    /// 从父节点形状信息创建 Concat 节点
    pub(in crate::nn) fn new(
        parent_shapes: &[&[usize]],
        parent_dynamic_shapes: &[DynamicShape],
        parent_ids: Vec<NodeId>,
        axis: usize,
    ) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parent_shapes.is_empty() {
            return Err(GraphError::InvalidOperation(
                "Concat 节点至少需要 1 个父节点".to_string(),
            ));
        }

        let first_shape = parent_shapes[0];
        let ndim = first_shape.len();

        // 2. 验证 axis（concat 模式 axis 必须小于 ndim）
        let max_axis = ndim.saturating_sub(1);
        if axis > max_axis {
            return Err(GraphError::InvalidOperation(format!(
                "Concat: axis {axis} 超出有效范围 [0, {max_axis}]"
            )));
        }

        // 3. 验证形状并收集 parent_sizes
        let mut parent_sizes = Vec::with_capacity(parent_shapes.len());

        for (i, shape) in parent_shapes.iter().enumerate() {
            if shape.len() != ndim {
                return Err(GraphError::ShapeMismatch {
                    expected: first_shape.to_vec(),
                    got: shape.to_vec(),
                    message: format!("Concat: 父节点 {i} 维度数不一致"),
                });
            }
            for (d, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if d != axis && s1 != s2 {
                    return Err(GraphError::ShapeMismatch {
                        expected: first_shape.to_vec(),
                        got: shape.to_vec(),
                        message: format!("Concat: 父节点 {i} 在维度 {d} 大小不一致"),
                    });
                }
            }
            parent_sizes.push(shape[axis]);
        }

        // 4. 计算输出形状
        let mut fixed_shape = first_shape.to_vec();
        fixed_shape[axis] = parent_sizes.iter().sum();

        // 5. 计算动态形状
        let supports_dynamic = parent_dynamic_shapes.iter().any(|d| d.has_dynamic_dims());
        let first_dyn = &parent_dynamic_shapes[0];

        let mut dims: Vec<Option<usize>> = Vec::with_capacity(fixed_shape.len());
        for (i, &size) in fixed_shape.iter().enumerate() {
            if i == axis {
                dims.push(Some(size));
            } else {
                dims.push(first_dyn.dim(i));
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
            parent_sizes,
            fixed_shape,
            dynamic_shape,
            supports_dynamic,
        })
    }
}

impl TraitNode for Concat {
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
        self.value = Some(Tensor::concat(parent_values, self.axis));
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
        let idx = target_parent_index;

        // 计算该父节点在 axis 维度的起始偏移
        let start_offset: usize = self.parent_sizes[..idx].iter().sum();
        let size = self.parent_sizes[idx];

        // concat 反向：按偏移分段提取梯度（使用 Tensor::narrow）
        Ok(GradResult::Computed(upstream_grad.narrow(
            self.axis,
            start_offset,
            size,
        )))
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
