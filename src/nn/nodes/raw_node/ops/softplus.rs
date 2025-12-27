use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::tensor::Tensor;

/// SoftPlus 激活函数节点
///
/// forward: f(x) = ln(1 + e^x)
/// backward: f'(x) = sigmoid(x) = 1 / (1 + e^(-x))
///
/// SoftPlus 是 ReLU 的平滑近似，处处可微，适用于：
/// - 需要正值输出的场景（如方差/标准差预测）
/// - 需要平滑梯度的优化场景
/// - 概率模型（VAE）、连续动作空间强化学习（SAC/PPO）
#[derive(Clone)]
pub(crate) struct SoftPlus {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    jacobi: Option<Tensor>,
    grad: Option<Tensor>, // Batch 模式的梯度
    shape: Vec<usize>,
    /// 缓存父节点的值（用于反向传播）
    parent_value: Option<Tensor>,
}

impl SoftPlus {
    pub(crate) fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        // 1. 必要的验证
        // 1.1 父节点数量验证
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "SoftPlus节点只需要1个父节点".to_string(),
            ));
        }

        // 2. 返回
        Ok(Self {
            id: None,
            name: None,
            value: None,
            jacobi: None,
            grad: None,
            shape: parents[0].value_expected_shape().to_vec(),
            parent_value: None,
        })
    }

    /// 数值稳定的 SoftPlus 计算
    ///
    /// 对于大正数 x，直接使用 ln(1 + e^x) 会导致 e^x 溢出
    /// 使用恒等变换: softplus(x) = x + ln(1 + e^(-x)) 当 x > 0
    ///              softplus(x) = ln(1 + e^x) 当 x <= 0
    fn stable_softplus(x: &Tensor) -> Tensor {
        // 阈值：超过此值时使用稳定公式避免溢出
        const THRESHOLD: f32 = 20.0;

        x.where_with_f32(
            |val| val > THRESHOLD,
            |val| val, // x > threshold: softplus(x) ≈ x
            |val| {
                if val > 0.0 {
                    // 0 < x <= threshold: x + ln(1 + e^(-x))
                    val + (1.0 + (-val).exp()).ln()
                } else {
                    // x <= 0: ln(1 + e^x)
                    (1.0 + val.exp()).ln()
                }
            },
        )
    }
}

impl TraitNode for SoftPlus {
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
        // 1. 获取父节点的值
        let parent_value = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的父节点{}没有值。不该触及本错误，否则说明crate代码有问题",
                self.display_node(),
                parents[0]
            ))
        })?;

        // 2. 缓存父节点的值（用于反向传播）
        self.parent_value = Some(parent_value.clone());

        // 3. 计算 softplus(x) = ln(1 + e^x)（数值稳定版本）
        self.value = Some(Self::stable_softplus(parent_value));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_jacobi_to_a_parent(
        &self,
        _target_parent: &NodeHandle,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // SoftPlus 的导数: d(softplus(x))/dx = sigmoid(x)
        // 由于是逐元素操作，雅可比矩阵是对角矩阵
        let parent_value = self.parent_value.as_ref().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}没有缓存的父节点值。不该触及本错误，否则说明crate代码有问题",
                self.display_node()
            ))
        })?;

        // 计算 sigmoid(x)，并转换为 Jacobian 对角矩阵
        let derivative = parent_value.sigmoid();
        Ok(derivative.jacobi_diag())
    }

    fn jacobi(&self) -> Option<&Tensor> {
        self.jacobi.as_ref()
    }

    fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError> {
        self.jacobi = jacobi.cloned();
        Ok(())
    }

    // ========== Batch 模式 ==========

    fn calc_grad_to_parent(
        &self,
        _target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // SoftPlus 的梯度: upstream_grad * sigmoid(x)
        let parent_value = self.parent_value.as_ref().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}没有缓存的父节点值，无法计算梯度",
                self.display_node()
            ))
        })?;

        // 计算 sigmoid(x)（逐元素）
        let local_grad = parent_value.sigmoid();

        // 逐元素乘以上游梯度
        Ok(upstream_grad * &local_grad)
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
}

