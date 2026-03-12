use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::hash_dedup_params;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;
use ndarray::Dimension;

/// TopK 节点 — 沿指定轴选取前 k 大元素
///
/// forward: (values, indices) = input.topk(k, axis, sorted)
///          节点输出 values，indices 存储在内部用于 backward
/// backward: 将上游梯度 scatter 回原位置
///           grad = zeros(parent_shape); 对每个 topk 位置，grad[..., indices[j], ...] += upstream[..., j, ...]
///
/// 语义对标 PyTorch 的 `torch.topk(input, k, dim, sorted=True)`。
#[derive(Clone)]
pub(crate) struct TopK {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 固定形状（topk 后的形状）
    fixed_shape: Vec<usize>,
    /// 动态形状
    dynamic_shape: DynamicShape,
    #[allow(dead_code)]
    supports_dynamic: bool,
    /// 选取数量
    k: usize,
    /// 操作的轴
    axis: usize,
    /// 是否排序
    sorted: bool,
    /// forward 时保存的 indices（用于 backward scatter）
    indices: Option<Tensor>,
}

impl TopK {
    pub(crate) const fn k(&self) -> usize { self.k }
    pub(crate) const fn axis(&self) -> usize { self.axis }
    pub(crate) const fn sorted(&self) -> bool { self.sorted }

    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
        k: usize,
        axis: usize,
        sorted: bool,
    ) -> Result<Self, GraphError> {
        // 验证参数
        if axis >= parent_shape.len() {
            return Err(GraphError::InvalidOperation(format!(
                "TopK: axis {axis} 超出维度 {}",
                parent_shape.len()
            )));
        }
        if k == 0 || k > parent_shape[axis] {
            return Err(GraphError::InvalidOperation(format!(
                "TopK: k={k} 必须在 1..={} 范围内",
                parent_shape[axis]
            )));
        }

        // 计算输出形状：与 parent 相同，但 axis 维度变为 k
        let mut output_shape = parent_shape.to_vec();
        output_shape[axis] = k;

        // 计算动态形状
        let output_dims: Vec<Option<usize>> = (0..parent_shape.len())
            .map(|i| {
                if parent_dynamic_shape.is_dynamic(i) {
                    None
                } else if i == axis {
                    Some(k)
                } else {
                    Some(parent_shape[i])
                }
            })
            .collect();
        let output_dynamic = DynamicShape::new(&output_dims);

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape: output_shape,
            dynamic_shape: output_dynamic,
            supports_dynamic: parent_dynamic_shape.has_dynamic_dims(),
            k,
            axis,
            sorted,
            indices: None,
        })
    }

    /// 获取 forward 后保存的 indices 张量
    #[allow(dead_code)]
    pub(in crate::nn) fn indices(&self) -> Option<&Tensor> {
        self.indices.as_ref()
    }
}

impl TraitNode for TopK {
    fn id(&self) -> NodeId { self.id.unwrap() }
    fn set_id(&mut self, id: NodeId) { self.id = Some(id); }
    fn name(&self) -> &str { self.name.as_ref().unwrap() }
    fn set_name(&mut self, name: &str) { self.name = Some(name.to_string()); }
    fn value_expected_shape(&self) -> &[usize] { &self.fixed_shape }
    fn dynamic_expected_shape(&self) -> DynamicShape { self.dynamic_shape.clone() }
    fn supports_dynamic_batch(&self) -> bool { self.supports_dynamic }

    fn dedup_fingerprint(&self) -> Option<u64> {
        Some(hash_dedup_params(&[
            self.k as u64,
            self.axis as u64,
            self.sorted as u64,
        ]))
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        let (values, indices) = parent_values[0].topk(self.k, self.axis, self.sorted);
        self.value = Some(values);
        self.indices = Some(indices);
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> { self.value.as_ref() }

    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        let parent_shape = parent_values[0].shape();
        let indices = self.indices.as_ref().ok_or_else(|| {
            GraphError::InvalidOperation(
                "TopK: 未找到 forward 保存的 indices，请先 forward".to_string(),
            )
        })?;

        // 创建与 parent 同形状的零张量
        let mut grad = Tensor::zeros(parent_shape);

        let axis = self.axis;

        // 将 upstream_grad 按 indices 散布到原始位置
        // indices 和 upstream_grad 形状相同（parent shape with axis_dim=k）
        let idx_shape = indices.shape();
        for idx in ndarray::indices(idx_shape) {
            let idx_slice: Vec<usize> = idx.as_array_view().to_vec();

            // 获取原始位置索引
            let orig_idx = indices.get_dyn(&idx_slice) as usize;

            // 获取 upstream_grad 对应位置的梯度值
            let grad_val = upstream_grad.get_dyn(&idx_slice);

            // 构建 parent 的索引：将 axis 维度替换为原始索引
            let mut parent_idx = idx_slice;
            parent_idx[axis] = orig_idx;

            // 累加梯度
            grad.add_at_dyn(&parent_idx, grad_val);
        }

        Ok(GradResult::Computed(grad))
    }

    fn grad(&self) -> Option<&Tensor> { self.grad.as_ref() }
    fn set_grad(&mut self, grad: Option<&Tensor>) -> Result<(), GraphError> {
        self.grad = grad.cloned();
        Ok(())
    }

    fn grad_mut(&mut self) -> Option<&mut Tensor> {
        self.grad.as_mut()
    }
    fn clear_value(&mut self) -> Result<(), GraphError> {
        self.value = None;
        self.indices = None;
        Ok(())
    }
    fn set_value_unchecked(&mut self, value: Option<&Tensor>) { self.value = value.cloned(); }
}
