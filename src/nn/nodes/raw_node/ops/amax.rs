/*
 * @Author       : 老董
 * @Date         : 2026-01-31
 * @Description  : Amax 节点 - 沿指定轴取最大值（只返回值，不返回索引）
 *
 * 用于强化学习（DQN 选最优动作 Q 值）、特征池化等场景
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Amax 节点：沿指定轴取最大值（只返回值，不返回索引）
///
/// 对应 `PyTorch` 的 `tensor.amax(dim=axis)`。
///
/// 前向：`output = amax(input, axis=dim)`
/// 反向：梯度只流向产生最大值的位置，如果有多个并列最大值则平分
///
/// # 主要用途
/// - DQN: `amax(Q_values, axis=1)` 选最优动作的 Q 值
/// - 特征池化：沿特征维度取最大
///
/// # 父节点顺序
/// - parents[0]: 输入张量
#[derive(Clone)]
pub(crate) struct Amax {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// reduction 的轴
    axis: usize,
    /// 固定形状（reduction 后的形状）
    fixed_shape: Vec<usize>,
    /// 动态形状
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    supports_dynamic: bool,
}

impl Amax {
    pub(crate) fn new(parents: &[&NodeHandle], axis: usize) -> Result<Self, GraphError> {
        // 1. 验证父节点数量（需要 1 个）
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "Amax 节点需要 1 个父节点".to_string(),
            ));
        }

        let input_shape = parents[0].value_expected_shape();

        // 2. 验证 axis 有效
        if axis >= input_shape.len() {
            return Err(GraphError::InvalidOperation(format!(
                "Amax: axis {} 超出输入维度 {}",
                axis,
                input_shape.len()
            )));
        }

        // 3. 计算 reduction 后的形状（移除指定轴）
        let mut fixed_shape: Vec<usize> = input_shape.to_vec();
        fixed_shape.remove(axis);

        // 如果结果是标量，至少保留一个维度
        if fixed_shape.is_empty() {
            fixed_shape.push(1);
        }

        // 4. 动态形状（也需要移除指定轴）
        let parent_dynamic = parents[0].dynamic_expected_shape();
        let mut dynamic_dims: Vec<_> = parent_dynamic.dims().to_vec();
        if axis < dynamic_dims.len() {
            dynamic_dims.remove(axis);
        }
        if dynamic_dims.is_empty() {
            dynamic_dims.push(Some(1)); // Dim = Option<usize>，Some(n) 表示固定值
        }
        let dynamic_shape = DynamicShape::new(&dynamic_dims);

        // 5. 是否支持动态 batch
        let supports_dynamic = parents[0].supports_dynamic_batch();

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            axis,
            fixed_shape,
            dynamic_shape,
            supports_dynamic,
        })
    }

    /// 获取 axis
    pub(crate) fn axis(&self) -> usize {
        self.axis
    }
}

impl TraitNode for Amax {
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

        // 使用 Tensor::amax(axis)
        self.value = Some(input_value.amax(self.axis));
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
        // Amax 的反向传播：
        // - 梯度只流向产生最大值的位置
        // - 如果有多个并列最大值，梯度平分

        let input_value = target_parent.value().ok_or_else(|| {
            GraphError::ComputationError("Amax 梯度计算时 target_parent 没有值".to_string())
        })?;

        let max_value = self.value.as_ref().ok_or_else(|| {
            GraphError::ComputationError("Amax 梯度计算时 value 为空".to_string())
        })?;

        let input_shape = input_value.shape();
        let axis = self.axis;

        // 创建与输入形状相同的零张量
        let mut grad_data = vec![0.0f32; input_value.size()];

        // 计算沿 axis 的步长和其他维度的信息
        let axis_size = input_shape[axis];
        let outer_size: usize = input_shape[..axis].iter().product();
        let inner_size: usize = input_shape[axis + 1..].iter().product();

        let input_slice = input_value.data_as_slice();
        let max_slice = max_value.data_as_slice();
        let upstream_slice = upstream_grad.data_as_slice();

        // 遍历每个 reduction 位置
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                // 计算当前位置在 max_value 和 upstream_grad 中的索引
                let reduced_idx = outer * inner_size + inner;
                let max_val = max_slice[reduced_idx];
                let upstream_val = upstream_slice[reduced_idx];

                // 统计有多少个位置等于最大值（用于平分梯度）
                let mut count = 0;
                for k in 0..axis_size {
                    let input_idx = outer * axis_size * inner_size + k * inner_size + inner;
                    if (input_slice[input_idx] - max_val).abs() < 1e-7 {
                        count += 1;
                    }
                }

                // 将梯度分配到等于最大值的位置
                let grad_per_max = if count > 0 {
                    upstream_val / count as f32
                } else {
                    0.0
                };

                for k in 0..axis_size {
                    let input_idx = outer * axis_size * inner_size + k * inner_size + inner;
                    if (input_slice[input_idx] - max_val).abs() < 1e-7 {
                        grad_data[input_idx] = grad_per_max;
                    }
                }
            }
        }

        Ok(Tensor::new(&grad_data, input_shape))
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
