/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : Flatten 节点 - 将张量展平为 2D（保留首维度或完全展平）
 *                 这是 Reshape 的便捷封装，常用于 CNN 与全连接层之间的转换
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::NodeId;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Flatten 节点 - 展平张量
///
/// # 功能
/// 将输入张量展平为 2D 张量，有两种模式：
/// - `keep_first_dim = true`: 保留首维度，展平其余维度 → `[m, n] → [m, n]`（2D 不变）或高维到 2D
/// - `keep_first_dim = false`: 完全展平为行向量 → `[m, n] → [1, m*n]`
///
/// # 特性
/// - Forward: 将输入展平为目标形状
/// - Backward (Jacobi): 单位矩阵（与 Reshape 相同）
/// - Backward (Gradient): 将上游梯度 reshape 回父节点形状
///
/// # 典型用途
/// CNN 输出 `[batch, features]` → Flatten → 全连接层输入
#[derive(Clone)]
pub(crate) struct Flatten {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 目标形状
    target_shape: Vec<usize>,
    /// 动态形状（支持动态 batch）
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    #[allow(dead_code)]
    supports_dynamic: bool,
    /// 父节点的原始形状（反向传播 / NEAT 可视化预留）
    #[allow(dead_code)]
    parent_shape: Vec<usize>,
    /// 是否保留首维度
    keep_first_dim: bool,
}

impl Flatten {
    /// 从父节点形状信息创建 Flatten 节点（核心实现）
    ///
    /// # 参数
    /// - `parent_shape`: 父节点形状
    /// - `parent_dynamic_shape`: 父节点的动态形状
    /// - `keep_first_dim`: 是否保留首维度
    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
        keep_first_dim: bool,
    ) -> Result<Self, GraphError> {
        let total_elements: usize = parent_shape.iter().product();

        // 计算目标形状
        let target_shape = if keep_first_dim {
            if parent_shape.len() == 2 {
                parent_shape.to_vec()
            } else {
                let first_dim = parent_shape[0];
                let rest_dim = total_elements / first_dim;
                vec![first_dim, rest_dim]
            }
        } else {
            vec![1, total_elements]
        };

        // 计算动态形状
        let supports_dynamic = parent_dynamic_shape.has_dynamic_dims();
        let dynamic_shape =
            if supports_dynamic && parent_dynamic_shape.is_dynamic(0) && keep_first_dim {
                let mut dims: Vec<Option<usize>> = target_shape.iter().map(|&d| Some(d)).collect();
                dims[0] = None;
                DynamicShape::new(&dims)
            } else {
                DynamicShape::fixed(&target_shape)
            };

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            target_shape,
            dynamic_shape,
            supports_dynamic,
            parent_shape: parent_shape.to_vec(),
            keep_first_dim,
        })
    }


    /// 获取目标形状
    #[allow(dead_code)]
    pub(crate) fn target_shape(&self) -> &[usize] {
        &self.target_shape
    }

    /// 是否保留首维度
    #[allow(dead_code)]
    pub(crate) const fn keep_first_dim(&self) -> bool {
        self.keep_first_dim
    }
}

impl TraitNode for Flatten {
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

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        let parent_value = parent_values[0];
        // 动态计算目标形状（支持动态 batch）
        let actual_shape = parent_value.shape();
        let runtime_target_shape = if self.keep_first_dim {
            if actual_shape.len() == 2 {
                actual_shape.to_vec()
            } else {
                let batch_dim = actual_shape[0];
                let rest_dim: usize = actual_shape[1..].iter().product();
                vec![batch_dim, rest_dim]
            }
        } else {
            let total: usize = actual_shape.iter().product();
            vec![1, total]
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
    ) -> Result<Tensor, GraphError> {
        // 使用父节点的运行时形状（支持动态 batch）
        // 注意：必须用 parent_values 的运行时形状，不能用 self.parent_shape（构建时静态形状）
        let parent_actual_shape = parent_values[target_parent_index].shape();
        // 将上游梯度 reshape 回父节点的实际形状
        Ok(upstream_grad.reshape(parent_actual_shape))
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
