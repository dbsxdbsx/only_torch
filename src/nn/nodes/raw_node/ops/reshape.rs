/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : Reshape 节点 - 改变张量形状而不改变数据
 *                 参考 MatrixSlow/matrixslow/ops/ops.py 中的 Reshape 类
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Reshape 节点 - 改变父节点值的形状
///
/// # 特性
/// - Forward: 将输入张量 reshape 为目标形状
/// - Backward (Jacobi): 单位矩阵（因为只是重新排列元素）
/// - Backward (Gradient): 将上游梯度 reshape 回父节点形状
///
/// # 动态 batch 支持
/// - 如果目标形状的第一维等于原始输入的 batch 大小，则认为是保留 batch 维度
/// - 运行时会按比例调整第一维以适应实际 batch 大小
///
/// # 约束
/// - 目标形状的元素总数必须与输入相同
#[derive(Clone)]
pub(crate) struct Reshape {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 目标形状（创建时的固定形状）
    target_shape: Vec<usize>,
    /// 动态形状（支持动态 batch）
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    #[allow(dead_code)]
    supports_dynamic: bool,
    /// 父节点的原始形状（用于反向传播）
    parent_shape: Vec<usize>,
    /// 原始 batch 大小（用于运行时按比例调整）
    original_batch_size: usize,
}

impl Reshape {
    /// 从父节点形状信息创建 Reshape 节点（核心实现）
    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
        target_shape: &[usize],
    ) -> Result<Self, GraphError> {
        // 1. 验证目标形状
        if target_shape.is_empty() {
            return Err(GraphError::InvalidOperation(
                "Reshape 目标形状不能为空".to_string(),
            ));
        }

        let parent_size: usize = parent_shape.iter().product();
        let target_size: usize = target_shape.iter().product();

        if parent_size != target_size {
            return Err(GraphError::ShapeMismatch {
                expected: parent_shape.to_vec(),
                got: target_shape.to_vec(),
                message: format!(
                    "Reshape 目标形状 {target_shape:?}（{target_size}个元素）与输入形状 {parent_shape:?}（{parent_size}个元素）的元素总数不匹配"
                ),
            });
        }

        // 2. 计算动态形状
        let supports_dynamic = parent_dynamic_shape.has_dynamic_dims();
        let dynamic_shape = if supports_dynamic && parent_dynamic_shape.is_dynamic(0) {
            let mut dims: Vec<Option<usize>> = target_shape.iter().map(|&d| Some(d)).collect();
            dims[0] = None;
            DynamicShape::new(&dims)
        } else {
            DynamicShape::fixed(target_shape)
        };

        let original_batch_size = parent_shape[0];

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            target_shape: target_shape.to_vec(),
            dynamic_shape,
            supports_dynamic,
            original_batch_size,
            parent_shape: parent_shape.to_vec(),
        })
    }

    /// 获取目标形状
    #[allow(dead_code)]
    pub(crate) fn target_shape(&self) -> &[usize] {
        &self.target_shape
    }

    /// 获取父节点原始形状
    #[allow(dead_code)]
    pub(crate) fn parent_shape(&self) -> &[usize] {
        &self.parent_shape
    }
}

impl TraitNode for Reshape {
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
        &self.target_shape
    }

    fn dynamic_expected_shape(&self) -> DynamicShape {
        self.dynamic_shape.clone()
    }

    fn supports_dynamic_batch(&self) -> bool {
        self.supports_dynamic
    }

    fn dedup_fingerprint(&self) -> Option<u64> {
        use crate::nn::nodes::raw_node::hash_dedup_params;
        let vals: Vec<u64> = self.target_shape.iter().map(|&s| s as u64).collect();
        Some(hash_dedup_params(&vals))
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        let parent_value = parent_values[0];
        // 动态计算目标形状（支持动态 batch）
        let actual_batch = parent_value.shape()[0];
        let runtime_target_shape =
            if self.supports_dynamic && actual_batch != self.original_batch_size {
                let mut new_shape = self.target_shape.clone();
                new_shape[0] = self.target_shape[0] * actual_batch / self.original_batch_size;
                new_shape
            } else {
                self.target_shape.clone()
            };
        self.value = Some(parent_value.reshape(&runtime_target_shape));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_grad_to_parent(
        &self,
        target_parent_index: usize,
        parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        // 使用父节点的运行时形状（支持动态 batch）
        // 注意：必须用 parent_values 的运行时形状，不能用 self.parent_shape（构建时静态形状）
        let parent_actual_shape = parent_values[target_parent_index].shape();
        // Reshape 的梯度就是将上游梯度 reshape 回父节点的实际形状
        Ok(GradResult::Computed(
            upstream_grad.reshape(parent_actual_shape),
        ))
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
