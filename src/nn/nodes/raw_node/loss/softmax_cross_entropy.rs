use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;
use rayon::prelude::*;

/// Softmax + `CrossEntropy` 融合损失节点
///
/// 将 Softmax 激活和交叉熵损失合并为单一节点，具有以下优势：
/// 1. 数值稳定性：使用 log-sum-exp 技巧避免溢出
/// 2. 梯度简洁：∂L/∂x = softmax(x) - y
///
/// ## 输入
/// - 父节点 0: logits（预测值，未经 softmax 的原始分数）
/// - 父节点 1: labels（one-hot 编码的真实标签）
///
/// ## 输出
/// - 标量损失值 L = -Σ `y_i` * log(softmax(x)_i)
///
/// ## 数值稳定计算
/// ```text
/// softmax(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))
/// L = -Σ y_i * (x_i - max(x) - log(Σ exp(x_j - max(x))))
/// ```
#[derive(Clone)]
pub(crate) struct SoftmaxCrossEntropy {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>, // 梯度
    /// 输出形状固定为 [1, 1]（标量损失）
    shape: Vec<usize>,
    /// 缓存 softmax 结果，用于反向传播（支持 batch）
    softmax_cache: Option<Tensor>,
    /// 父节点 ID，用于区分 logits 和 labels
    #[allow(dead_code)]
    parents_ids: Vec<NodeId>,
}

impl SoftmaxCrossEntropy {
    /// 从父节点形状信息创建 SoftmaxCrossEntropy 节点（核心实现）
    pub(in crate::nn) fn new(
        logits_shape: &[usize],
        labels_shape: &[usize],
        logits_dynamic_shape: &DynamicShape,
        labels_dynamic_shape: &DynamicShape,
        parent_ids: Vec<NodeId>,
    ) -> Result<Self, GraphError> {
        // 验证形状兼容性
        if !logits_dynamic_shape.is_compatible(labels_dynamic_shape) {
            return Err(GraphError::ShapeMismatch {
                expected: logits_shape.to_vec(),
                got: labels_shape.to_vec(),
                message: "logits 和 labels 动态形状必须兼容".to_string(),
            });
        }

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            shape: vec![1, 1],
            softmax_cache: None,
            parents_ids: parent_ids,
        })
    }

    /// 计算数值稳定的 softmax 与交叉熵损失（支持 batch，Rayon 并行）。
    ///
    /// softmax 和 loss 共享同一轮 max / exp 计算，避免分类训练主路径重复扫描 logits。
    /// 输入: [batch, `num_classes`] 或 [1, `num_classes`]
    /// 输出: (softmax, 平均 loss)
    fn stable_softmax_and_loss_batch(logits: &Tensor, labels: &Tensor) -> (Tensor, f32) {
        let shape = logits.shape();
        let batch_size = shape[0];
        let num_classes = shape[1];
        let mut softmax_data = vec![0.0f32; batch_size * num_classes];

        let total_loss: f32 = softmax_data
            .par_chunks_mut(num_classes)
            .enumerate()
            .map(|(b, sample_result)| {
                // 找到该样本的最大值
                let mut max_val = logits[[b, 0]];
                for c in 1..num_classes {
                    if logits[[b, c]] > max_val {
                        max_val = logits[[b, c]];
                    }
                }

                // 计算 exp(x - max) 和 sum
                let mut sum_exp = 0.0f32;
                for c in 0..num_classes {
                    let exp_val = (logits[[b, c]] - max_val).exp();
                    sample_result[c] = exp_val;
                    sum_exp += exp_val;
                }

                // 归一化
                for c in 0..num_classes {
                    sample_result[c] /= sum_exp;
                }

                let log_sum_exp = sum_exp.ln();

                // 计算该样本的损失
                // L = -Σ y_i * (x_i - max - log_sum_exp)
                let mut dot_product = 0.0f32;
                for c in 0..num_classes {
                    dot_product += logits[[b, c]] * labels[[b, c]];
                }

                -dot_product + max_val + log_sum_exp
            })
            .sum();

        (
            Tensor::new(&softmax_data, shape),
            total_loss / batch_size as f32,
        )
    }
}

impl TraitNode for SoftmaxCrossEntropy {
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
        let logits = parent_values[0];
        let labels = parent_values[1];
        let (softmax, loss) = Self::stable_softmax_and_loss_batch(logits, labels);
        self.softmax_cache = Some(softmax);
        self.value = Some(Tensor::new(&[loss], &[1, 1]));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    // ========== VJP 模式 ==========

    /// `SoftmaxCrossEntropy` 的 batch 梯度计算
    ///
    /// 对于 logits: [batch, `num_classes`]，labels: [batch, `num_classes`]：
    /// `dL/d_logits` = (softmax - labels) / `batch_size`
    fn calc_grad_to_parent(
        &self,
        target_parent_index: usize,
        parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        if target_parent_index == 0 {
            // 对 logits 的梯度
            let softmax = self.softmax_cache.as_ref().ok_or_else(|| {
                GraphError::ComputationError("softmax 缓存为空，需先执行前向传播".to_string())
            })?;
            let labels = parent_values[1];

            // dL/d_logits = (softmax - labels) / batch_size
            let batch_size = softmax.shape()[0] as f32;
            let upstream = upstream_grad[[0, 0]];
            let grad = ((softmax - labels) / batch_size) * upstream;
            Ok(GradResult::Computed(grad))
        } else {
            // 对 labels 的梯度（通常不需要，labels 是常量）
            Err(GraphError::InvalidOperation(
                "不应该对 labels 计算梯度".to_string(),
            ))
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
