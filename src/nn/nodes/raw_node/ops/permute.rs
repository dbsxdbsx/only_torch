use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::hash_dedup_params;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Permute 节点 — 维度重排（转置的一般形式）
///
/// forward: output = input.permute(dims)
/// backward: upstream_grad.permute(inverse_dims)
///
/// 命名遵循 PyTorch 的 `tensor.permute(dims)`。
/// `transpose(dim0, dim1)` 是 permute 的特例（交换两个维度）。
#[derive(Clone)]
pub(crate) struct Permute {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 输出形状（维度重排后）
    fixed_shape: Vec<usize>,
    /// 动态形状
    dynamic_shape: DynamicShape,
    #[allow(dead_code)]
    supports_dynamic: bool,
    /// 排列顺序，如 [0, 2, 1]
    dims: Vec<usize>,
    /// 逆排列（反向传播用），inverse_dims[dims[i]] = i
    inverse_dims: Vec<usize>,
}

impl Permute {
    pub(crate) fn dims(&self) -> &[usize] { &self.dims }

    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
        dims: &[usize],
    ) -> Result<Self, GraphError> {
        let ndim = parent_shape.len();

        // 验证 dims 长度
        if dims.len() != ndim {
            return Err(GraphError::InvalidOperation(format!(
                "Permute: dims 长度 {} 与输入维度 {ndim} 不匹配",
                dims.len()
            )));
        }

        // 验证 dims 是 [0..ndim) 的合法排列
        let mut seen = vec![false; ndim];
        for &d in dims {
            if d >= ndim {
                return Err(GraphError::InvalidOperation(format!(
                    "Permute: dims 中包含无效维度 {d}，输入只有 {ndim} 维"
                )));
            }
            if seen[d] {
                return Err(GraphError::InvalidOperation(format!(
                    "Permute: dims 中维度 {d} 重复"
                )));
            }
            seen[d] = true;
        }

        // 计算输出形状：output_shape[i] = parent_shape[dims[i]]
        let output_shape: Vec<usize> = dims.iter().map(|&d| parent_shape[d]).collect();

        // 计算逆排列：inverse_dims[dims[i]] = i
        let mut inverse_dims = vec![0usize; ndim];
        for (i, &d) in dims.iter().enumerate() {
            inverse_dims[d] = i;
        }

        // 计算动态形状：按 dims 重排动态维度信息
        let output_dynamic_dims: Vec<Option<usize>> = dims
            .iter()
            .map(|&d| {
                if parent_dynamic_shape.is_dynamic(d) {
                    None // 动态维度保持动态
                } else {
                    Some(parent_shape[d])
                }
            })
            .collect();
        let output_dynamic = DynamicShape::new(&output_dynamic_dims);

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape: output_shape,
            dynamic_shape: output_dynamic,
            supports_dynamic: parent_dynamic_shape.has_dynamic_dims(),
            dims: dims.to_vec(),
            inverse_dims,
        })
    }
}

impl TraitNode for Permute {
    fn id(&self) -> NodeId { self.id.unwrap() }
    fn set_id(&mut self, id: NodeId) { self.id = Some(id); }
    fn name(&self) -> &str { self.name.as_ref().unwrap() }
    fn set_name(&mut self, name: &str) { self.name = Some(name.to_string()); }
    fn value_expected_shape(&self) -> &[usize] { &self.fixed_shape }
    fn dynamic_expected_shape(&self) -> DynamicShape { self.dynamic_shape.clone() }
    fn supports_dynamic_batch(&self) -> bool { self.supports_dynamic }

    fn dedup_fingerprint(&self) -> Option<u64> {
        let vals: Vec<u64> = self.dims.iter().map(|&d| d as u64).collect();
        Some(hash_dedup_params(&vals))
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        self.value = Some(parent_values[0].permute(&self.dims));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> { self.value.as_ref() }

    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        // 逆排列恢复原始维度顺序
        Ok(GradResult::Computed(upstream_grad.permute(&self.inverse_dims)))
    }

    fn grad(&self) -> Option<&Tensor> { self.grad.as_ref() }
    fn set_grad(&mut self, grad: Option<&Tensor>) -> Result<(), GraphError> {
        self.grad = grad.cloned();
        Ok(())
    }

    fn grad_mut(&mut self) -> Option<&mut Tensor> {
        self.grad.as_mut()
    }
    fn clear_value(&mut self) -> Result<(), GraphError> { self.value = None; Ok(()) }
    fn set_value_unchecked(&mut self, value: Option<&Tensor>) { self.value = value.cloned(); }
}
