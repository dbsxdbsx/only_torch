/*
 * @Author       : 老董
 * @Date         : 2026-05-01
 * @Description  : Atan2 节点 - 逐元素计算 atan2(y, x)
 *
 * CIoU loss 中的角度差项 v = (4/π²) · (atan(w_t/h_t) − atan(w_p/h_p))²
 * 在 only_torch 中需要这个基础积木才能用拼接式实现。
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::{Tensor, broadcast_shape};

/// Atan2 节点：逐元素计算 `atan2(y, x)`，返回值范围 `(-π, π]`
///
/// # 父节点顺序
/// - parents[0]: y（沿轴的"sin 边"，对齐 PyTorch `torch.atan2(y, x)`）
/// - parents[1]: x（沿轴的"cos 边"）
///
/// 支持 NumPy 风格广播。
///
/// # 反向传播解析公式
/// 对 `out = atan2(y, x)`：
/// - `∂out/∂y =  x / (x² + y²)`
/// - `∂out/∂x = -y / (x² + y²)`
///
/// # `(y, x) = (0, 0)` 处的 fallback（项目选择，与 `PyTorch` 不一致）
/// PyTorch 实测在该点 `backward` 返回 `NaN`（`x² + y² = 0` 形成 `0/0`）。
/// only_torch 选择更安全的 `0` fallback，避免 `NaN` 污染下游训练（特别是
/// CIoU 在退化样本上常踩到 `(0, 0)`）。**这是项目意图，不是与 PyTorch 对齐**——
/// 测试和注释里需要明确说明。
#[derive(Clone)]
pub(crate) struct Atan2 {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 固定形状（广播后的形状）
    fixed_shape: Vec<usize>,
    /// 动态形状（支持动态 batch）
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    #[allow(dead_code)]
    supports_dynamic: bool,
}

impl Atan2 {
    /// 从父节点形状信息创建 Atan2 节点
    pub(in crate::nn) fn new(
        y_shape: &[usize],
        x_shape: &[usize],
        y_dynamic_shape: &DynamicShape,
        x_dynamic_shape: &DynamicShape,
    ) -> Result<Self, GraphError> {
        let fixed_shape =
            broadcast_shape(y_shape, x_shape).ok_or_else(|| GraphError::ShapeMismatch {
                expected: y_shape.to_vec(),
                got: x_shape.to_vec(),
                message: "Atan2 节点的父节点形状无法广播".to_string(),
            })?;

        let dynamic_shape = y_dynamic_shape.broadcast_with(x_dynamic_shape);

        let supports_dynamic = y_dynamic_shape.dims().first() == Some(&None)
            || x_dynamic_shape.dims().first() == Some(&None);

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape,
            dynamic_shape,
            supports_dynamic,
        })
    }
}

impl TraitNode for Atan2 {
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
        self.value = Some(parent_values[0].atan2(parent_values[1]));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// Atan2 反向传播 VJP
    ///
    /// `∂atan2(y, x)/∂y =  x / (x² + y²)`
    /// `∂atan2(y, x)/∂x = -y / (x² + y²)`
    ///
    /// `(y, x) = (0, 0)` 时 fallback 为 0（项目意图，不同于 PyTorch NaN）。
    fn calc_grad_to_parent(
        &self,
        target_parent_index: usize,
        parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        let y_value = parent_values.first().ok_or_else(|| {
            GraphError::ComputationError("Atan2 梯度计算时 y 父节点没有值".to_string())
        })?;
        let x_value = parent_values.get(1).ok_or_else(|| {
            GraphError::ComputationError("Atan2 梯度计算时 x 父节点没有值".to_string())
        })?;

        let target_shape = parent_values
            .get(target_parent_index)
            .ok_or_else(|| {
                GraphError::ComputationError(format!(
                    "Atan2 梯度计算时 target_parent_index {target_parent_index} 越界"
                ))
            })?
            .shape()
            .to_vec();
        let output_shape = upstream_grad.shape();

        // broadcast_to 已产出连续张量；upstream 用 Cow 守卫：连续时零拷贝借用。
        let y_broadcast = y_value.broadcast_to(output_shape).into_contiguous();
        let x_broadcast = x_value.broadcast_to(output_shape).into_contiguous();
        let upstream_contiguous = upstream_grad.contiguous();

        let y_slice = y_broadcast.data_as_slice();
        let x_slice = x_broadcast.data_as_slice();
        let upstream_slice = upstream_contiguous.data_as_slice();

        let mut grad_data = Vec::with_capacity(upstream_slice.len());

        match target_parent_index {
            0 => {
                for i in 0..upstream_slice.len() {
                    let y = y_slice[i];
                    let x = x_slice[i];
                    let denom = x * x + y * y;
                    let local = if denom == 0.0 { 0.0 } else { x / denom };
                    grad_data.push(upstream_slice[i] * local);
                }
            }
            1 => {
                for i in 0..upstream_slice.len() {
                    let y = y_slice[i];
                    let x = x_slice[i];
                    let denom = x * x + y * y;
                    let local = if denom == 0.0 { 0.0 } else { -y / denom };
                    grad_data.push(upstream_slice[i] * local);
                }
            }
            other => {
                return Err(GraphError::ComputationError(format!(
                    "Atan2 只有 2 个父节点（y, x），target_parent_index = {other} 越界"
                )));
            }
        }

        let grad = Tensor::new(&grad_data, output_shape);

        if output_shape != target_shape.as_slice() {
            Ok(GradResult::Computed(grad.sum_to_shape(&target_shape)))
        } else {
            Ok(GradResult::Computed(grad))
        }
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
