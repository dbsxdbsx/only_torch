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
    /// 输出形状固定为 [1, 1]（标量损失）
    shape: Vec<usize>,
    /// 缓存 softmax 结果，用于反向传播
    softmax_cache: Option<Tensor>,
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
            shape: vec![1, 1], // 损失是标量
            softmax_cache: None,
            parents_ids: vec![parents[0].id(), parents[1].id()],
        })
    }

    /// 计算数值稳定的 softmax
    /// softmax(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))
    fn stable_softmax(logits: &Tensor) -> Tensor {
        let max_val = logits.max_value();
        let shifted = logits - max_val;
        let exp_shifted = shifted.exp();
        let sum_exp: f32 = exp_shifted.flatten_view().iter().sum();
        exp_shifted / sum_exp
    }

    /// 计算数值稳定的交叉熵损失
    /// L = -Σ y_i * log(softmax(x)_i)
    ///   = -Σ y_i * (x_i - max(x) - log(Σ exp(x_j - max(x))))
    fn stable_cross_entropy(logits: &Tensor, labels: &Tensor) -> f32 {
        let max_val = logits.max_value();
        let shifted = logits - max_val;
        let exp_shifted = shifted.exp();
        let sum_exp: f32 = exp_shifted.flatten_view().iter().sum();
        let log_sum_exp = sum_exp.ln();

        // L = -Σ y_i * (x_i - max - log_sum_exp)
        //   = -Σ y_i * x_i + max * Σ y_i + log_sum_exp * Σ y_i
        // 由于 y 是 one-hot，Σ y_i = 1
        let logits_flat = logits.flatten_view();
        let labels_flat = labels.flatten_view();
        let dot_product: f32 = logits_flat
            .iter()
            .zip(labels_flat.iter())
            .map(|(x, y)| x * y)
            .sum();

        -dot_product + max_val + log_sum_exp
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

        // 缓存 softmax 用于反向传播
        self.softmax_cache = Some(Self::stable_softmax(logits));

        // 计算损失
        let loss = Self::stable_cross_entropy(logits, labels);
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
}

