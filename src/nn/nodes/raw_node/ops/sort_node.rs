/*
 * @Author       : 老董
 * @Date         : 2026-02-14
 * @Description  : SortNode — 沿指定轴排序（可微）
 *
 * 前向：调用 Tensor::sort_along_axis，输出排序后的值，内部缓存索引
 * 反向：利用缓存的索引将上游梯度 scatter 回原始位置（逆置换）
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// SortNode：沿指定轴排序，返回排序后的值（可微）
///
/// 前向：`(sorted_values, indices) = input.sort_along_axis(axis, descending)`
///       输出 sorted_values，indices 缓存供反向使用
/// 反向：`grad_input[indices[i]] += upstream_grad[i]`（沿排序轴的逆置换）
#[derive(Clone)]
pub(crate) struct SortNode {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 输出形状（与输入相同）
    fixed_shape: Vec<usize>,
    /// 动态形状
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    #[allow(dead_code)]
    supports_dynamic: bool,
    /// 排序轴
    axis: usize,
    /// 是否降序
    descending: bool,
    /// 前向缓存的排序索引（反向传播用）
    indices: Option<Tensor>,
}

impl SortNode {
    pub(crate) const fn axis(&self) -> usize {
        self.axis
    }
    pub(crate) const fn descending(&self) -> bool {
        self.descending
    }

    /// 创建 SortNode
    ///
    /// # 参数
    /// - `parent_shape`: 父节点形状
    /// - `parent_dynamic_shape`: 父节点动态形状
    /// - `axis`: 排序轴
    /// - `descending`: 是否降序
    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
        axis: usize,
        descending: bool,
    ) -> Result<Self, GraphError> {
        // 验证轴范围
        if axis >= parent_shape.len() {
            return Err(GraphError::InvalidOperation(format!(
                "SortNode: axis {} 超出张量维度 {}",
                axis,
                parent_shape.len()
            )));
        }

        let fixed_shape = parent_shape.to_vec();
        let dynamic_shape = parent_dynamic_shape.clone();
        let supports_dynamic = parent_dynamic_shape.has_dynamic_dims();

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape,
            dynamic_shape,
            supports_dynamic,
            axis,
            descending,
            indices: None,
        })
    }
}

impl TraitNode for SortNode {
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
        use crate::nn::nodes::raw_node::hash_dedup_params;
        Some(hash_dedup_params(&[
            self.axis as u64,
            self.descending as u64,
        ]))
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        let (sorted, indices) = parent_values[0].sort_along_axis(self.axis, self.descending);
        self.value = Some(sorted);
        self.indices = Some(indices);
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        // Sort 反向传播：逆置换
        //
        // 前向：sorted[..., i, ...] = input[..., indices[i], ...]
        // 反向：grad_input[..., indices[i], ...] += upstream_grad[..., i, ...]
        //
        // 利用 Tensor::scatter_by_sort_indices 将上游梯度 scatter 回原始位置

        let indices = self.indices.as_ref().ok_or_else(|| {
            GraphError::InvalidOperation(format!(
                "{}：indices 未缓存，需先执行前向传播",
                self.display_node()
            ))
        })?;

        let parent_shape = parent_values[0].shape();
        Ok(GradResult::Computed(upstream_grad.scatter_by_sort_indices(
            self.axis,
            indices,
            parent_shape,
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
        self.indices = None;
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
