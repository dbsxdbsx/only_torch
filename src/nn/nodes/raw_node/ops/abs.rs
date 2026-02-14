use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Abs 绝对值节点
///
/// forward: abs(x) = |x|
/// backward: d(abs)/dx = sign(x)，在 x=0 处定义为 0
///
/// 与 `ReLU` 类似，在 x=0 处严格意义上不可微，但实践中定义为 0 不影响训练。
#[derive(Clone)]
pub(crate) struct Abs {
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
    /// 缓存父节点的值，用于反向传播计算 sign(x)
    parent_value_cache: Option<Tensor>,
}

impl Abs {
    /// 从父节点形状信息创建 Abs 节点（核心实现）
    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
    ) -> Result<Self, GraphError> {
        let supports_dynamic = parent_dynamic_shape.dims().first() == Some(&None);

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape: parent_shape.to_vec(),
            dynamic_shape: parent_dynamic_shape.clone(),
            supports_dynamic,
            parent_value_cache: None,
        })
    }
}

impl TraitNode for Abs {
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
        // 缓存父节点的值用于反向传播
        self.parent_value_cache = Some(parent_values[0].clone());
        // 计算 abs(x) = |x|
        self.value = Some(parent_values[0].abs());
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// Abs 的梯度: `upstream_grad` * sign(x)
    ///
    /// 在 x=0 处，sign(0) = 0，所以梯度也为 0（与 `PyTorch` 行为一致）
    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        // 获取缓存的父节点值
        let parent_value = self.parent_value_cache.as_ref().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的父节点值缓存为空，需先执行前向传播",
                self.display_node()
            ))
        })?;

        // 计算局部梯度 sign(x)
        let local_grad = parent_value.sign();

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
        self.parent_value_cache = None;
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
