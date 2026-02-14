use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Leaky `ReLU` 激活函数节点
///
/// forward: f(x) = x if x > 0, else `negative_slope` * x
/// backward: d(f)/dx = 1 if x > 0, else `negative_slope`
///
/// 当 `negative_slope` = 0 时，等价于标准 `ReLU`
/// `MatrixSlow` 使用 `negative_slope` = 0.1
#[derive(Clone)]
pub(crate) struct LeakyReLU {
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
    /// 负半轴斜率，默认 0.0（标准 `ReLU`）
    negative_slope: f32,
}

impl LeakyReLU {
    /// 获取 `negative_slope（alpha）值`
    pub(crate) const fn alpha(&self) -> f32 {
        self.negative_slope
    }

    /// 从父节点形状信息创建 LeakyReLU 节点（核心实现）
    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
        negative_slope: f32,
    ) -> Result<Self, GraphError> {
        // negative_slope 验证（通常应该是非负小数）
        if negative_slope < 0.0 {
            return Err(GraphError::InvalidOperation(format!(
                "LeakyReLU的negative_slope应为非负数，但得到: {negative_slope}"
            )));
        }

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape: parent_shape.to_vec(),
            dynamic_shape: parent_dynamic_shape.clone(),
            supports_dynamic: parent_dynamic_shape.has_dynamic_dims(),
            negative_slope,
        })
    }
}

impl TraitNode for LeakyReLU {
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

    fn dedup_fingerprint(&self) -> Option<u64> {
        Some(self.negative_slope.to_bits() as u64)
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        self.value = Some(parent_values[0].leaky_relu(self.negative_slope));
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
    ) -> Result<GradResult, GraphError> {
        // LeakyReLU 的梯度: upstream_grad * (1 if x > 0 else negative_slope)
        //
        // 重要：使用 value（输出）而非 parent_value（输入）判断区域
        // 这对 BPTT 很关键，因为 BPTT 只恢复 value，不恢复 parent_value
        // 数学上等价：output > 0 ⟺ input > 0（当 slope >= 0 时）
        let value = self.value().ok_or_else(|| {
            GraphError::ComputationError(format!("{}没有值，无法计算梯度", self.display_node()))
        })?;

        // 计算局部梯度（逐元素）
        let slope = self.negative_slope;
        let local_grad = value.where_with_f32(
            |y| y > 0.0,
            |_| 1.0,   // y > 0 时导数为 1
            |_| slope, // y <= 0 时导数为 slope
        );

        // 逐元素乘以上游梯度
        Ok(GradResult::Computed(upstream_grad * &local_grad))
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
