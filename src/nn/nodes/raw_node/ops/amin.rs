/*
 * @Author       : 老董
 * @Date         : 2026-01-31
 * @Description  : Amin 节点 - 沿指定轴取最小值（只返回值，不返回索引）
 *
 * 用于强化学习（如 Double DQN 选保守 Q 值）、特征池化等场景
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Amin 节点：沿指定轴取最小值（只返回值，不返回索引）
///
/// 对应 `PyTorch` 的 `tensor.amin(dim=axis)`。
///
/// 前向：`output = amin(input, axis=dim)`
/// 反向：梯度只流向产生最小值的位置，如果有多个并列最小值则平分
///
/// # 主要用途
/// - Double DQN: 选保守 Q 值
/// - 特征池化：沿特征维度取最小
///
/// # 父节点顺序
/// - parents[0]: 输入张量
#[derive(Clone)]
pub(crate) struct Amin {
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
    #[allow(dead_code)]
    supports_dynamic: bool,
}

impl Amin {
    /// 从父节点形状信息创建 Amin 节点（核心实现）
    pub(in crate::nn) fn new(
        input_shape: &[usize],
        input_dynamic_shape: &DynamicShape,
        axis: usize,
    ) -> Result<Self, GraphError> {
        // 验证 axis 有效
        if axis >= input_shape.len() {
            return Err(GraphError::InvalidOperation(format!(
                "Amin: axis {} 超出输入维度 {}",
                axis,
                input_shape.len()
            )));
        }

        // 计算 reduction 后的形状（移除指定轴）
        let mut fixed_shape: Vec<usize> = input_shape.to_vec();
        fixed_shape.remove(axis);
        if fixed_shape.is_empty() {
            fixed_shape.push(1);
        }

        // 动态形状（也需要移除指定轴）
        let mut dynamic_dims: Vec<_> = input_dynamic_shape.dims().to_vec();
        if axis < dynamic_dims.len() {
            dynamic_dims.remove(axis);
        }
        if dynamic_dims.is_empty() {
            dynamic_dims.push(Some(1));
        }
        let dynamic_shape = DynamicShape::new(&dynamic_dims);

        // 是否支持动态 batch
        let supports_dynamic = input_dynamic_shape.dims().first() == Some(&None);

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
    #[allow(dead_code)]
    pub(crate) fn axis(&self) -> usize {
        self.axis
    }
}

impl TraitNode for Amin {
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

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        // 使用 Tensor::amin(axis)
        self.value = Some(parent_values[0].amin(self.axis));
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
        // Amin 的反向传播：
        // - 梯度只流向产生最小值的位置
        // - 如果有多个并列最小值，梯度平分

        let input_value = parent_values.get(target_parent_index).ok_or_else(|| {
            GraphError::ComputationError("Amin 梯度计算时父节点没有值".to_string())
        })?;

        let min_value = self.value.as_ref().ok_or_else(|| {
            GraphError::ComputationError("Amin 梯度计算时 value 为空".to_string())
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
        let min_slice = min_value.data_as_slice();
        let upstream_slice = upstream_grad.data_as_slice();

        // 遍历每个 reduction 位置
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                // 计算当前位置在 min_value 和 upstream_grad 中的索引
                let reduced_idx = outer * inner_size + inner;
                let min_val = min_slice[reduced_idx];
                let upstream_val = upstream_slice[reduced_idx];

                // 统计有多少个位置等于最小值（用于平分梯度）
                let mut count = 0;
                for k in 0..axis_size {
                    let input_idx = outer * axis_size * inner_size + k * inner_size + inner;
                    if (input_slice[input_idx] - min_val).abs() < 1e-7 {
                        count += 1;
                    }
                }

                // 将梯度分配到等于最小值的位置
                let grad_per_min = if count > 0 {
                    upstream_val / count as f32
                } else {
                    0.0
                };

                for k in 0..axis_size {
                    let input_idx = outer * axis_size * inner_size + k * inner_size + inner;
                    if (input_slice[input_idx] - min_val).abs() < 1e-7 {
                        grad_data[input_idx] = grad_per_min;
                    }
                }
            }
        }

        Ok(GradResult::Computed(Tensor::new(&grad_data, input_shape)))
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
