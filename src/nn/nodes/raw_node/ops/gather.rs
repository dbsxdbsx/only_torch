/*
 * @Author       : 老董
 * @Date         : 2026-01-31
 * @Description  : Gather 节点 - 按索引张量从指定维度收集元素
 *
 * 用于强化学习（SAC/DQN 等）：按动作索引从 Q 值张量中选择对应的 Q 值
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;
use ndarray::Dimension;

/// Gather 节点：按索引张量从指定维度收集元素
///
/// 前向：`output = input.gather(dim, index)`
/// 反向：scatter 梯度回原位置（只对 input 计算梯度，index 不需要梯度）
///
/// # 主要用途
/// SAC/DQN 等强化学习算法中，按动作索引选择 Q 值。
///
/// # 父节点顺序
/// - parents[0]: 输入数据节点（如 Q 值）
/// - parents[1]: 索引节点（如动作索引）
#[derive(Clone)]
pub(crate) struct Gather {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 输出形状（与 index 形状相同）
    fixed_shape: Vec<usize>,
    /// 动态形状
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    supports_dynamic: bool,
    /// gather 的维度
    dim: usize,
    /// 输入数据的形状（用于反向传播时创建 scatter tensor）
    input_shape: Vec<usize>,
}

impl Gather {
    /// 获取 gather 的维度
    pub(in crate::nn) const fn dim(&self) -> usize {
        self.dim
    }

    /// 从父节点形状信息创建 Gather 节点（核心实现）
    pub(in crate::nn) fn new_from_shapes(
        input_shape: &[usize],
        index_shape: &[usize],
        input_dynamic_shape: &DynamicShape,
        index_dynamic_shape: &DynamicShape,
        dim: usize,
    ) -> Result<Self, GraphError> {
        let ndim = input_shape.len();

        // 1. 验证 dim
        if dim >= ndim {
            return Err(GraphError::InvalidOperation(format!(
                "Gather dim {} 超出输入张量维度 {}",
                dim, ndim
            )));
        }

        // 2. 验证 index 维度数与 input 相同
        if index_shape.len() != ndim {
            return Err(GraphError::InvalidOperation(format!(
                "Gather: index 维度数 {} 必须与输入张量维度数 {} 相同",
                index_shape.len(),
                ndim
            )));
        }

        // 3. 验证除 dim 外的其他维度大小一致
        for d in 0..ndim {
            if d != dim && index_shape[d] != input_shape[d] {
                return Err(GraphError::InvalidOperation(format!(
                    "Gather: 维度 {} 上 index 大小 {} 与输入张量大小 {} 不匹配",
                    d, index_shape[d], input_shape[d]
                )));
            }
        }

        // 4. 输出形状与 index 形状相同
        let fixed_shape = index_shape.to_vec();

        // 5. 计算动态形状
        let dynamic_shape = index_dynamic_shape.clone();
        let supports_dynamic =
            input_dynamic_shape.has_dynamic_dims() && index_dynamic_shape.has_dynamic_dims();

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape,
            dynamic_shape,
            supports_dynamic,
            dim,
            input_shape: input_shape.to_vec(),
        })
    }

    /// 从 NodeHandle 创建（过渡期 API，委托给 new_from_shapes）
    pub(crate) fn new(parents: &[&NodeHandle], dim: usize) -> Result<Self, GraphError> {
        if parents.len() != 2 {
            return Err(GraphError::InvalidOperation(
                "Gather 节点需要 2 个父节点（input 和 index）".to_string(),
            ));
        }

        let input = &parents[0];
        let index = &parents[1];
        let input_shape = input.value_expected_shape();
        let index_shape = index.value_expected_shape();
        let input_dynamic_shape = input.dynamic_expected_shape();
        let index_dynamic_shape = index.dynamic_expected_shape();

        Self::new_from_shapes(
            &input_shape,
            &index_shape,
            &input_dynamic_shape,
            &index_dynamic_shape,
            dim,
        )
    }
}

impl TraitNode for Gather {
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

    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
        let input_value = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{} 的父 {} 没有值",
                self.display_node(),
                parents[0]
            ))
        })?;

        let index_value = parents[1].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{} 的索引父 {} 没有值",
                self.display_node(),
                parents[1]
            ))
        })?;

        // 使用 Tensor::gather
        self.value = Some(input_value.gather(self.dim, index_value));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_grad_to_parent(
        &self,
        target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // Gather 的反向传播：
        // - 对 input（parents[0]）：scatter 梯度回原位置
        // - 对 index（parents[1]）：不需要梯度（返回全零张量）

        // 判断 target_parent 是 input 还是 index
        let index_parent = assistant_parent.ok_or_else(|| {
            GraphError::ComputationError(
                "Gather calc_grad_to_parent 需要 assistant_parent（index 节点）".to_string(),
            )
        })?;

        // 检查 target_parent 是否是 index 节点
        if target_parent.id() == index_parent.id() {
            // 对 index 节点，返回全零梯度（index 不参与梯度计算）
            return Ok(Tensor::zeros(target_parent.value_expected_shape()));
        }

        // 对 input 节点，执行 scatter 操作
        let index_value = index_parent.value().ok_or_else(|| {
            GraphError::ComputationError("Gather 反向传播时 index 节点没有值".to_string())
        })?;

        // 获取实际的输入形状（可能是动态 batch）
        let actual_input_shape = if let Some(parent_value) = target_parent.value() {
            parent_value.shape().to_vec()
        } else {
            self.input_shape.clone()
        };

        // 创建全零梯度张量
        let mut grad_input = Tensor::zeros(&actual_input_shape);

        // Scatter：将 upstream_grad 的值按 index 放回 grad_input 对应位置
        // 使用 ndarray 的索引遍历
        let index_shape = index_value.shape();
        for idx in ndarray::indices(index_shape) {
            // 将 ndarray 索引转换为切片
            let idx_slice: Vec<usize> = idx.as_array_view().to_vec();

            // 获取 index 中的值作为 scatter 索引
            let gather_idx = index_value.get_dyn(&idx_slice) as usize;

            // 获取 upstream_grad 对应位置的梯度值
            let grad_val = upstream_grad.get_dyn(&idx_slice);

            // 构建 grad_input 的索引
            let mut input_idx = idx_slice;
            input_idx[self.dim] = gather_idx;

            // 累加梯度（同一位置可能被多次 gather）
            grad_input.add_at_dyn(&input_idx, grad_val);
        }

        Ok(grad_input)
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
