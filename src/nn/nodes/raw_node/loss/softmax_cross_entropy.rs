use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::tensor::Tensor;

/// Softmax + CrossEntropy 融合损失节点
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
/// - 标量损失值 L = -Σ y_i * log(softmax(x)_i)
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
    jacobi: Option<Tensor>,
    grad: Option<Tensor>, // Batch 模式的梯度
    /// 输出形状固定为 [1, 1]（标量损失）
    shape: Vec<usize>,
    /// 缓存 softmax 结果，用于反向传播（支持 batch）
    softmax_cache: Option<Tensor>,
    /// 缓存 labels，用于反向传播
    labels_cache: Option<Tensor>,
    /// 父节点 ID，用于区分 logits 和 labels
    parents_ids: Vec<NodeId>,
}

impl SoftmaxCrossEntropy {
    pub(crate) fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parents.len() != 2 {
            return Err(GraphError::InvalidOperation(
                "SoftmaxCrossEntropy 节点需要 2 个父节点（logits 和 labels）".to_string(),
            ));
        }

        // 2. 验证形状兼容性
        let logits_shape = parents[0].value_expected_shape();
        let labels_shape = parents[1].value_expected_shape();
        if logits_shape != labels_shape {
            return Err(GraphError::ShapeMismatch {
                expected: logits_shape.to_vec(),
                got: labels_shape.to_vec(),
                message: "logits 和 labels 形状必须相同".to_string(),
            });
        }

        Ok(Self {
            id: None,
            name: None,
            value: None,
            jacobi: None,
            grad: None,
            shape: vec![1, 1], // 损失是标量
            softmax_cache: None,
            labels_cache: None,
            parents_ids: vec![parents[0].id(), parents[1].id()],
        })
    }

    /// 计算数值稳定的 softmax（支持 batch）
    /// 输入: [batch, num_classes] 或 [1, num_classes]
    /// 输出: [batch, num_classes] 或 [1, num_classes]
    fn stable_softmax_batch(logits: &Tensor) -> Tensor {
        let shape = logits.shape();
        let batch_size = shape[0];
        let num_classes = shape[1];

        let mut result = Tensor::zeros(shape);
        for b in 0..batch_size {
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
                result[[b, c]] = exp_val;
                sum_exp += exp_val;
            }

            // 归一化
            for c in 0..num_classes {
                result[[b, c]] /= sum_exp;
            }
        }
        result
    }

    /// 计算数值稳定的交叉熵损失（支持 batch，返回平均损失）
    fn stable_cross_entropy_batch(logits: &Tensor, labels: &Tensor) -> f32 {
        let shape = logits.shape();
        let batch_size = shape[0];
        let num_classes = shape[1];

        let mut total_loss = 0.0f32;
        for b in 0..batch_size {
            // 找到该样本的最大值
            let mut max_val = logits[[b, 0]];
            for c in 1..num_classes {
                if logits[[b, c]] > max_val {
                    max_val = logits[[b, c]];
                }
            }

            // 计算 log_sum_exp
            let mut sum_exp = 0.0f32;
            for c in 0..num_classes {
                sum_exp += (logits[[b, c]] - max_val).exp();
            }
            let log_sum_exp = sum_exp.ln();

            // 计算该样本的损失
            // L = -Σ y_i * (x_i - max - log_sum_exp)
            let mut dot_product = 0.0f32;
            for c in 0..num_classes {
                dot_product += logits[[b, c]] * labels[[b, c]];
            }
            total_loss += -dot_product + max_val + log_sum_exp;
        }

        // 返回平均损失
        total_loss / batch_size as f32
    }

    /// 兼容旧代码的单样本 softmax
    fn stable_softmax(logits: &Tensor) -> Tensor {
        Self::stable_softmax_batch(logits)
    }

    /// 兼容旧代码的单样本交叉熵
    fn stable_cross_entropy(logits: &Tensor, labels: &Tensor) -> f32 {
        Self::stable_cross_entropy_batch(logits, labels)
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

    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
        // 获取 logits 和 labels
        let logits = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的 logits 父节点{}没有值",
                self.display_node(),
                parents[0]
            ))
        })?;
        let labels = parents[1].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的 labels 父节点{}没有值",
                self.display_node(),
                parents[1]
            ))
        })?;

        // 缓存 softmax 和 labels 用于反向传播
        self.softmax_cache = Some(Self::stable_softmax_batch(logits));
        self.labels_cache = Some(labels.clone());

        // 计算损失（batch 平均）
        let loss = Self::stable_cross_entropy_batch(logits, labels);
        self.value = Some(Tensor::new(&[loss], &[1, 1]));

        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_jacobi_to_a_parent(
        &self,
        target_parent: &NodeHandle,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // 梯度公式：∂L/∂logits = softmax - labels
        // 注意：对 labels 的梯度通常不需要（labels 是常量），但为完整性仍实现

        if target_parent.id() == self.parents_ids[0] {
            // 对 logits 的梯度
            let softmax = self.softmax_cache.as_ref().ok_or_else(|| {
                GraphError::ComputationError(
                    "softmax 缓存为空，需先执行前向传播".to_string(),
                )
            })?;
            let labels = _assistant_parent
                .ok_or_else(|| {
                    GraphError::ComputationError("缺少 labels 辅助父节点".to_string())
                })?
                .value()
                .ok_or_else(|| GraphError::ComputationError("labels 没有值".to_string()))?;

            // ∂L/∂logits = softmax - labels
            // 由于损失是标量 [1,1]，logits 是 [n]，雅可比是 [1, n]
            let grad = softmax - labels;
            let n = grad.size();
            Ok(grad.reshape(&[1, n]))
        } else {
            // 对 labels 的梯度（通常不需要，但为完整性实现）
            // ∂L/∂labels = -log(softmax)
            let softmax = self.softmax_cache.as_ref().ok_or_else(|| {
                GraphError::ComputationError(
                    "softmax 缓存为空，需先执行前向传播".to_string(),
                )
            })?;
            let grad = softmax.ln() * (-1.0);
            let n = grad.size();
            Ok(grad.reshape(&[1, n]))
        }
    }

    fn jacobi(&self) -> Option<&Tensor> {
        self.jacobi.as_ref()
    }

    fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError> {
        self.jacobi = jacobi.cloned();
        Ok(())
    }

    // ========== Batch 模式 ==========

    /// SoftmaxCrossEntropy 的 batch 梯度计算
    ///
    /// 对于 logits: [batch, num_classes]，labels: [batch, num_classes]：
    /// dL/d_logits = (softmax - labels) / batch_size
    fn calc_grad_to_parent(
        &self,
        target_parent: &NodeHandle,
        _upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        if target_parent.id() == self.parents_ids[0] {
            // 对 logits 的梯度
            let softmax = self.softmax_cache.as_ref().ok_or_else(|| {
                GraphError::ComputationError(
                    "softmax 缓存为空，需先执行前向传播".to_string(),
                )
            })?;
            let labels = self.labels_cache.as_ref().ok_or_else(|| {
                GraphError::ComputationError("labels 缓存为空，需先执行前向传播".to_string())
            })?;

            // dL/d_logits = (softmax - labels) / batch_size
            let batch_size = softmax.shape()[0] as f32;
            let grad = (softmax - labels) / batch_size;
            Ok(grad)
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

    fn set_grad(&mut self, grad: Option<&Tensor>) -> Result<(), GraphError> {
        self.grad = grad.cloned();
        Ok(())
    }
}

