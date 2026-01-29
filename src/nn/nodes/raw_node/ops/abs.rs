use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
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
    supports_dynamic: bool,
    /// 缓存父节点的值，用于反向传播计算 sign(x)
    parent_value_cache: Option<Tensor>,
}

impl Abs {
    pub(crate) fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        // 1. 必要的验证
        // 1.1 父节点数量验证
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "Abs节点只需要1个父节点".to_string(),
            ));
        }

        // 2. 从父节点继承动态形状信息
        let parent = &parents[0];
        let fixed_shape = parent.value_expected_shape().to_vec();
        let dynamic_shape = parent.dynamic_expected_shape();
        let supports_dynamic = parent.supports_dynamic_batch();

        // 3. 返回
        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape,
            dynamic_shape,
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

    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
        // 1. 获取父节点的值
        let parent_value = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的父{}没有值。不该触及本错误，否则说明 crate 代码有问题",
                self.display_node(),
                parents[0]
            ))
        })?;

        // 2. 缓存父节点的值用于反向传播
        self.parent_value_cache = Some(parent_value.clone());

        // 3. 计算 abs(x) = |x|
        self.value = Some(parent_value.abs());
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
        _target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
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
        self.parent_value_cache = None;
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
