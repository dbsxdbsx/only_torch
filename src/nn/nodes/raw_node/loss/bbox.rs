use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::raw_node::hash_dedup_params;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;
use crate::vision::detection::{BBox, BoxFormat};

use super::Reduction;

const FINITE_DIFF_EPSILON: f32 = 1e-3;

/// bbox IoU-family 损失类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BBoxLossKind {
    /// `1 - IoU`
    IoU,
    /// `1 - GIoU`
    GIoU,
    /// `1 - DIoU`
    DIoU,
    /// `1 - CIoU`
    CIoU,
}

/// BBox IoU-family 损失节点。
///
/// 输入与 target 均为 `[N, 4]`，坐标格式由 `format` 显式指定。该节点面向
/// 已完成正负样本匹配后的 bbox 回归，不包含 anchor matching / obj / cls 等
/// 具体检测器逻辑。
///
/// 反向传播当前使用有限差分近似梯度，适合小规模验证和 adapter 原型；若进入真实
/// detection fine-tune，需要优先替换为解析梯度以降低训练成本并提升数值稳定性。
#[derive(Clone)]
pub(crate) struct BBoxLoss {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    shape: Vec<usize>,
    kind: BBoxLossKind,
    format: BoxFormat,
    reduction: Reduction,
    num_boxes_cache: usize,
    #[allow(dead_code)]
    parents_ids: Vec<NodeId>,
}

impl BBoxLoss {
    pub(in crate::nn) fn new(
        input_shape: &[usize],
        target_shape: &[usize],
        input_dynamic_shape: &DynamicShape,
        target_dynamic_shape: &DynamicShape,
        parent_ids: Vec<NodeId>,
        kind: BBoxLossKind,
        format: BoxFormat,
        reduction: Reduction,
    ) -> Result<Self, GraphError> {
        validate_bbox_dynamic_shape("input", input_shape, input_dynamic_shape)?;
        validate_bbox_dynamic_shape("target", target_shape, target_dynamic_shape)?;
        if !input_dynamic_shape.is_compatible(target_dynamic_shape) {
            return Err(GraphError::ShapeMismatch {
                expected: input_shape.to_vec(),
                got: target_shape.to_vec(),
                message: "bbox_loss: input 和 target 动态形状必须兼容".to_string(),
            });
        }

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            shape: vec![1, 1],
            kind,
            format,
            reduction,
            num_boxes_cache: input_shape.first().copied().unwrap_or(0),
            parents_ids: parent_ids,
        })
    }

    pub(crate) const fn kind(&self) -> BBoxLossKind {
        self.kind
    }

    pub(crate) const fn format(&self) -> BoxFormat {
        self.format
    }

    pub(crate) const fn reduction(&self) -> Reduction {
        self.reduction
    }

    fn loss_sum(input: &Tensor, target: &Tensor, kind: BBoxLossKind, format: BoxFormat) -> f32 {
        let num_boxes = input.shape()[0];
        (0..num_boxes)
            .map(|row| {
                let pred = bbox_from_row(input, row, format);
                let gt = bbox_from_row(target, row, format);
                1.0 - similarity(pred, gt, kind)
            })
            .sum()
    }

    fn finite_diff_grad(
        input: &Tensor,
        target: &Tensor,
        kind: BBoxLossKind,
        format: BoxFormat,
        reduction: Reduction,
        upstream_scale: f32,
    ) -> Tensor {
        let mut grad_data = vec![0.0; input.size()];
        let mut plus_data = input.data_as_slice().to_vec();
        let mut minus_data = input.data_as_slice().to_vec();
        let num_boxes = input.shape()[0];
        let reduction_scale = match reduction {
            Reduction::Mean => {
                if num_boxes == 0 {
                    0.0
                } else {
                    1.0 / num_boxes as f32
                }
            }
            Reduction::Sum => 1.0,
        };

        for idx in 0..input.size() {
            let original = input.data_as_slice()[idx];
            plus_data[idx] = original + FINITE_DIFF_EPSILON;
            minus_data[idx] = original - FINITE_DIFF_EPSILON;

            let plus = Tensor::new(&plus_data, input.shape());
            let minus = Tensor::new(&minus_data, input.shape());
            let plus_loss = Self::loss_sum(&plus, target, kind, format);
            let minus_loss = Self::loss_sum(&minus, target, kind, format);
            grad_data[idx] = (plus_loss - minus_loss) / (2.0 * FINITE_DIFF_EPSILON)
                * reduction_scale
                * upstream_scale;

            plus_data[idx] = original;
            minus_data[idx] = original;
        }

        Tensor::new(&grad_data, input.shape())
    }
}

impl TraitNode for BBoxLoss {
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
        &self.shape
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        let input = parent_values[0];
        let target = parent_values[1];
        validate_bbox_runtime_shape(input, target)?;

        self.num_boxes_cache = input.shape()[0];
        let loss_sum = Self::loss_sum(input, target, self.kind, self.format);
        let loss_value = match self.reduction {
            Reduction::Mean => {
                if self.num_boxes_cache == 0 {
                    0.0
                } else {
                    loss_sum / self.num_boxes_cache as f32
                }
            }
            Reduction::Sum => loss_sum,
        };
        self.value = Some(Tensor::new(&[loss_value], &[1, 1]));
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
        if target_parent_index != 0 {
            return Err(GraphError::InvalidOperation(
                "不应该对 bbox_loss target 计算梯度".to_string(),
            ));
        }

        let input = parent_values[0];
        let target = parent_values[1];
        validate_bbox_runtime_shape(input, target)?;
        let upstream_scale =
            upstream_grad
                .get_data_number()
                .ok_or_else(|| GraphError::ShapeMismatch {
                    expected: vec![1, 1],
                    got: upstream_grad.shape().to_vec(),
                    message: "bbox_loss: upstream_grad 必须是标量 [1, 1]".to_string(),
                })?;

        let grad = Self::finite_diff_grad(
            input,
            target,
            self.kind,
            self.format,
            self.reduction,
            upstream_scale,
        );
        Ok(GradResult::Computed(grad))
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

    fn dedup_fingerprint(&self) -> Option<u64> {
        Some(hash_dedup_params(&[
            self.kind as u64,
            self.format as u64,
            self.reduction as u64,
        ]))
    }
}

fn validate_bbox_dynamic_shape(
    role: &str,
    shape: &[usize],
    dynamic_shape: &DynamicShape,
) -> Result<(), GraphError> {
    if dynamic_shape.ndim() != 2 || dynamic_shape.dim(1) != Some(4) {
        return Err(GraphError::ShapeMismatch {
            expected: vec![shape.first().copied().unwrap_or(0), 4],
            got: shape.to_vec(),
            message: format!("bbox_loss: {role} 必须是 [N, 4] bbox Tensor"),
        });
    }
    Ok(())
}

fn validate_bbox_runtime_shape(input: &Tensor, target: &Tensor) -> Result<(), GraphError> {
    if input.shape() != target.shape() {
        return Err(GraphError::ShapeMismatch {
            expected: input.shape().to_vec(),
            got: target.shape().to_vec(),
            message: "bbox_loss: input 和 target 形状必须一致".to_string(),
        });
    }
    if input.shape().len() != 2 || input.shape()[1] != 4 {
        return Err(GraphError::ShapeMismatch {
            expected: vec![input.shape().first().copied().unwrap_or(0), 4],
            got: input.shape().to_vec(),
            message: "bbox_loss: input 和 target 必须是 [N, 4] bbox Tensor".to_string(),
        });
    }
    Ok(())
}

fn bbox_from_row(tensor: &Tensor, row: usize, format: BoxFormat) -> BBox {
    BBox::from_array(
        [
            tensor[[row, 0]],
            tensor[[row, 1]],
            tensor[[row, 2]],
            tensor[[row, 3]],
        ],
        format,
    )
}

fn similarity(a: BBox, b: BBox, kind: BBoxLossKind) -> f32 {
    match kind {
        BBoxLossKind::IoU => a.iou(b),
        BBoxLossKind::GIoU => a.giou(b),
        BBoxLossKind::DIoU => a.diou(b),
        BBoxLossKind::CIoU => a.ciou(b),
    }
}
