/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : Flatten 节点 - 将张量展平为 2D（保留首维度或完全展平）
 *                 这是 Reshape 的便捷封装，常用于 CNN 与全连接层之间的转换
 */

use crate::nn::shape::DynamicShape;
use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
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
    supports_dynamic: bool,
    /// 父节点的原始形状（用于反向传播）
    parent_shape: Vec<usize>,
    /// 是否保留首维度
    keep_first_dim: bool,
}

impl Flatten {
    /// 创建 Flatten 节点
    ///
    /// # 参数
    /// - `parents`: 父节点（只能有 1 个）
    /// - `keep_first_dim`: 是否保留首维度
    ///   - `true`: `[m, n]` → `[m, n]`（2D 已经是展平状态）
    ///   - `false`: `[m, n]` → `[1, m*n]`（完全展平为行向量）
    pub(crate) fn new(parents: &[&NodeHandle], keep_first_dim: bool) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "Flatten 节点只需要 1 个父节点".to_string(),
            ));
        }

        // 2. 获取父节点形状
        let parent = &parents[0];
        let parent_shape = parent.value_expected_shape().to_vec();
        let total_elements: usize = parent_shape.iter().product();

        // 3. 计算目标形状
        let target_shape = if keep_first_dim {
            // 保留首维度：对于 2D 张量，形状不变
            // 对于高维张量（未来扩展），会展平为 [first_dim, rest]
            if parent_shape.len() == 2 {
                // 2D 已经是展平状态
                parent_shape.clone()
            } else {
                // 高维：[d0, d1, d2, ...] → [d0, d1*d2*...]
                let first_dim = parent_shape[0];
                let rest_dim = total_elements / first_dim;
                vec![first_dim, rest_dim]
            }
        } else {
            // 完全展平为行向量
            vec![1, total_elements]
        };

        // 4. 计算动态形状
        let parent_dyn = parent.dynamic_expected_shape();
        let supports_dynamic = parent.supports_dynamic_batch();

        // 如果父节点第一维是动态的且保留首维度，输出也保持第一维动态
        let dynamic_shape = if supports_dynamic && parent_dyn.is_dynamic(0) && keep_first_dim {
            let mut dims: Vec<Option<usize>> = target_shape.iter().map(|&d| Some(d)).collect();
            dims[0] = None; // 第一维动态
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
            parent_shape,
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

    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
        // 1. 获取父节点的值
        let parent_value = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{} 的父节点 {} 没有值。不该触及本错误，否则说明 crate 代码有问题",
                self.display_node(),
                parents[0]
            ))
        })?;

        // 2. 动态计算目标形状（支持动态 batch）
        // 使用实际输入的 batch 大小，而不是创建时固定的 target_shape
        let actual_shape = parent_value.shape();
        let runtime_target_shape = if self.keep_first_dim {
            if actual_shape.len() == 2 {
                // 2D 已经是展平状态
                actual_shape.to_vec()
            } else {
                // 高维：[batch, d1, d2, ...] → [batch, d1*d2*...]
                let batch_dim = actual_shape[0];
                let rest_dim: usize = actual_shape[1..].iter().product();
                vec![batch_dim, rest_dim]
            }
        } else {
            // 完全展平为行向量
            let total: usize = actual_shape.iter().product();
            vec![1, total]
        };

        // 3. Reshape 到运行时计算的目标形状
        self.value = Some(parent_value.reshape(&runtime_target_shape));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_grad_to_parent(
        &self,
        target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // 使用父节点的实际形状（支持动态 batch）
        let parent_actual_shape = target_parent.value().map_or_else(
            || self.parent_shape.clone(), // fallback 到固定形状
            |v| v.shape().to_vec(),
        );
        // 将上游梯度 reshape 回父节点的实际形状
        Ok(upstream_grad.reshape(&parent_actual_shape))
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
