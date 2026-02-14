use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

use super::Reduction;

/// BCE（Binary Cross Entropy，二元交叉熵）损失节点
///
/// 计算 logits 和二值标签之间的二元交叉熵损失。
/// 采用 `BCEWithLogitsLoss` 形式，内置 Sigmoid 激活，数值稳定。
///
/// ## 使用场景
/// - **二分类**：输出单个 logit，预测 0/1
/// - **多标签分类**：输出多个 logits，每个独立预测 0/1（一个样本可同时属于多个类别）
///
/// ## 公式（数值稳定形式）
/// ```text
/// BCE = mean(max(logits, 0) - logits * target + log(1 + exp(-|logits|)))
/// ```
///
/// 等价于：
/// ```text
/// p = sigmoid(logits)
/// BCE = -mean(target * log(p) + (1 - target) * log(1 - p))
/// ```
///
/// ## 梯度
/// - Mean: `∂L/∂logits = (sigmoid(logits) - target) / N`
/// - Sum: `∂L/∂logits = sigmoid(logits) - target`
///
/// ## 输入
/// - 父节点 0: logits（未激活的原始输出）
/// - 父节点 1: target（二值标签，0 或 1）
///
/// ## 输出
/// - 标量损失值 [1, 1]
///
/// ## 参考
/// - `PyTorch`: `torch.nn.BCEWithLogitsLoss`
#[derive(Clone)]
pub(crate) struct BCE {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 输出形状固定为 [1, 1]（标量损失）
    shape: Vec<usize>,
    /// Reduction 模式: Mean 或 Sum
    reduction: Reduction,
    /// 缓存 sigmoid(logits)，用于反向传播
    sigmoid_cache: Option<Tensor>,
    /// 缓存 target，用于反向传播
    target_cache: Option<Tensor>,
    /// 缓存元素总数，用于 mean reduction
    numel_cache: usize,
    /// 父节点 ID，用于区分 logits 和 target
    #[allow(dead_code)]
    parents_ids: Vec<NodeId>,
}

impl BCE {
    /// 从父节点形状信息创建 BCE 节点（核心实现）
    pub(in crate::nn) fn new(
        logits_shape: &[usize],
        target_shape: &[usize],
        logits_dynamic_shape: &DynamicShape,
        target_dynamic_shape: &DynamicShape,
        parent_ids: Vec<NodeId>,
        reduction: Reduction,
    ) -> Result<Self, GraphError> {
        // 验证形状兼容性
        if !logits_dynamic_shape.is_compatible(target_dynamic_shape) {
            return Err(GraphError::ShapeMismatch {
                expected: logits_shape.to_vec(),
                got: target_shape.to_vec(),
                message: "logits 和 target 动态形状必须兼容".to_string(),
            });
        }

        let numel: usize = logits_shape.iter().product();

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            shape: vec![1, 1],
            reduction,
            sigmoid_cache: None,
            target_cache: None,
            numel_cache: numel,
            parents_ids: parent_ids,
        })
    }

    /// 计算数值稳定的 BCE 损失总和
    ///
    /// BCE(logit, target) = max(logit, 0) - logit * target + log(1 + exp(-|logit|))
    fn stable_bce_sum(logits: &Tensor, target: &Tensor) -> f32 {
        let logits_view = logits.flatten_view();
        let target_view = target.flatten_view();

        logits_view
            .iter()
            .zip(target_view.iter())
            .map(|(&l, &t)| {
                // 数值稳定的 BCE 计算
                let max_val = l.max(0.0);
                let abs_logit = l.abs();
                l.mul_add(-t, max_val) + (-abs_logit).exp().ln_1p()
            })
            .sum()
    }
}

impl TraitNode for BCE {
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
        let target = parent_values[1];
        // 更新 numel（支持动态 batch size）
        self.numel_cache = logits.size();
        // 计算 sigmoid(logits) 并缓存，用于反向传播
        let sigmoid = logits.sigmoid();
        self.sigmoid_cache = Some(sigmoid);
        self.target_cache = Some(target.clone());
        // 计算数值稳定的 BCE
        let bce_sum = Self::stable_bce_sum(logits, target);
        // 根据 reduction 模式计算损失
        let loss_value = match self.reduction {
            Reduction::Mean => bce_sum / (self.numel_cache as f32),
            Reduction::Sum => bce_sum,
        };
        self.value = Some(Tensor::new(&[loss_value], &[1, 1]));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// BCE 的 VJP 梯度计算
    ///
    /// 对于 logits: [batch, features]，target: [batch, features]：
    /// - Mean: `dL/d_logits = (sigmoid(logits) - target) / N`
    /// - Sum: `dL/d_logits = sigmoid(logits) - target`
    fn calc_grad_to_parent(
        &self,
        target_parent_index: usize,
        _parent_values: &[&Tensor],
        _upstream_grad: &Tensor,
    ) -> Result<Tensor, GraphError> {
        let sigmoid = self.sigmoid_cache.as_ref().ok_or_else(|| {
            GraphError::ComputationError("sigmoid 缓存为空，需先执行前向传播".to_string())
        })?;
        let target = self.target_cache.as_ref().ok_or_else(|| {
            GraphError::ComputationError("target 缓存为空，需先执行前向传播".to_string())
        })?;

        if target_parent_index == 0 {
            // 对 logits 的梯度: sigmoid - target
            let diff = sigmoid - target;
            let grad = match self.reduction {
                Reduction::Mean => &diff * (1.0 / self.numel_cache as f32),
                Reduction::Sum => diff,
            };
            Ok(grad)
        } else {
            // 对 target 的梯度（通常不需要）
            Err(GraphError::InvalidOperation(
                "不应该对 target 计算梯度".to_string(),
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
