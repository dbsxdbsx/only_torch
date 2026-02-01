use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Sigmoid 激活函数节点
///
/// forward: sigmoid(x) = 1 / (1 + e^(-x))
/// backward: d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
#[derive(Clone)]
pub(crate) struct Sigmoid {
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
}

impl Sigmoid {
    /// 从父节点形状信息创建 Sigmoid 节点（核心实现）
    pub(in crate::nn) fn new_from_shapes(
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

    /// 从 NodeHandle 创建（过渡期 API，委托给 new_from_shapes）
    pub(crate) fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "Sigmoid节点只需要1个父节点".to_string(),
            ));
        }

        let parent = &parents[0];
        Self::new_from_shapes(&parent.value_expected_shape(), &parent.dynamic_expected_shape())
    }
}

impl TraitNode for Sigmoid {
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
        // 计算 sigmoid(x) = 1 / (1 + e^(-x))
        self.value = Some(parent_values[0].sigmoid());
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
        // Sigmoid 的梯度: upstream_grad * sigmoid(x) * (1 - sigmoid(x))
        let value = self.value().ok_or_else(|| {
            GraphError::ComputationError(format!("{}没有值，无法计算梯度", self.display_node()))
        })?;

        // 计算 sigmoid(x) * (1 - sigmoid(x))（逐元素）
        let one_minus_sigmoid = Tensor::ones(value.shape()) - value;
        let local_grad = value * &one_minus_sigmoid;

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
