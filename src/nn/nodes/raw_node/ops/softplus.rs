use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// `SoftPlus` 激活函数节点
///
/// forward: f(x) = ln(1 + e^x)
/// backward: f'(x) = sigmoid(x) = 1 / (1 + e^(-x)) = 1 - exp(-softplus(x))
///
/// `SoftPlus` 是 `ReLU` 的平滑近似，处处可微，适用于：
/// - 需要正值输出的场景（如方差/标准差预测）
/// - 需要平滑梯度的优化场景
/// - 概率模型（VAE）、连续动作空间强化学习（SAC/PPO）
#[derive(Clone)]
pub(crate) struct SoftPlus {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 固定形状（用于 `value_expected_shape`）
    fixed_shape: Vec<usize>,
    /// 动态形状（支持动态 batch）
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    #[allow(dead_code)]
    supports_dynamic: bool,
}

impl SoftPlus {
    /// 从父节点形状信息创建 SoftPlus 节点（核心实现）
    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
    ) -> Result<Self, GraphError> {
        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape: parent_shape.to_vec(),
            dynamic_shape: parent_dynamic_shape.clone(),
            supports_dynamic: parent_dynamic_shape.has_dynamic_dims(),
        })
    }

    /// 数值稳定的 `SoftPlus` 计算
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
                    val + (-val).exp().ln_1p()
                } else {
                    // x <= 0: ln(1 + e^x)
                    val.exp().ln_1p()
                }
            },
        )
    }

    /// 从 softplus(x) 计算 sigmoid(x)
    ///
    /// 数学推导:
    ///   y = softplus(x) = ln(1 + e^x)
    ///   e^y = 1 + e^x
    ///   sigmoid(x) = e^x / (1 + e^x) = (e^y - 1) / e^y = 1 - e^(-y)
    ///
    /// 这允许我们从输出计算梯度，对 BPTT 很关键
    fn sigmoid_from_softplus(softplus_output: &Tensor) -> Tensor {
        // sigmoid(x) = 1 - exp(-softplus(x))
        // 对于大 y（对应大正数 x），sigmoid ≈ 1，exp(-y) 会下溢到 0，公式仍正确
        softplus_output.where_with_f32(
            |y| y > 20.0,
            |_| 1.0, // 大 y 时 sigmoid ≈ 1
            |y| 1.0 - (-y).exp(),
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
        &self.fixed_shape
    }

    fn dynamic_expected_shape(&self) -> DynamicShape {
        self.dynamic_shape.clone()
    }

    fn supports_dynamic_batch(&self) -> bool {
        self.supports_dynamic
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        // 计算 softplus(x) = ln(1 + e^x)（数值稳定版本）
        self.value = Some(Self::stable_softplus(parent_values[0]));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<Tensor, GraphError> {
        // SoftPlus 的梯度: upstream_grad * sigmoid(x) = upstream_grad * (1 - exp(-y))
        //
        // 重要：使用 value（输出）计算 sigmoid，而非 parent_value（输入）
        // 这对 BPTT 很关键，因为 BPTT 只恢复 value，不恢复 parent_value
        let value = self.value().ok_or_else(|| {
            GraphError::ComputationError(format!("{}没有值，无法计算梯度", self.display_node()))
        })?;

        // 计算 sigmoid(x) = 1 - exp(-softplus(x))（逐元素）
        let local_grad = Self::sigmoid_from_softplus(value);

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

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
