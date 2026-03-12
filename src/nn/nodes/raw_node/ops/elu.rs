use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// ELU 激活函数节点
///
/// forward: elu(x, alpha) = x if x > 0, else alpha * (exp(x) - 1)
/// backward: elu'(x) = 1 if x > 0, else value + alpha（可从 value 反推）
#[derive(Clone)]
pub(crate) struct Elu {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    fixed_shape: Vec<usize>,
    dynamic_shape: DynamicShape,
    #[allow(dead_code)]
    supports_dynamic: bool,
    alpha: f32,
}

impl Elu {
    pub(crate) const fn alpha(&self) -> f32 { self.alpha }

    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
        alpha: f32,
    ) -> Result<Self, GraphError> {
        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape: parent_shape.to_vec(),
            dynamic_shape: parent_dynamic_shape.clone(),
            supports_dynamic: parent_dynamic_shape.has_dynamic_dims(),
            alpha,
        })
    }
}

impl TraitNode for Elu {
    fn id(&self) -> NodeId { self.id.unwrap() }
    fn set_id(&mut self, id: NodeId) { self.id = Some(id); }
    fn name(&self) -> &str { self.name.as_ref().unwrap() }
    fn set_name(&mut self, name: &str) { self.name = Some(name.to_string()); }
    fn value_expected_shape(&self) -> &[usize] { &self.fixed_shape }
    fn dynamic_expected_shape(&self) -> DynamicShape { self.dynamic_shape.clone() }
    fn supports_dynamic_batch(&self) -> bool { self.supports_dynamic }

    fn dedup_fingerprint(&self) -> Option<u64> {
        Some(self.alpha.to_bits() as u64)
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        self.value = Some(parent_values[0].elu(self.alpha));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> { self.value.as_ref() }

    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        // elu'(x) = 1 if x > 0, else value + alpha（从 value 反推）
        let value = self.value().ok_or_else(|| {
            GraphError::ComputationError(format!("{}没有值，无法计算梯度", self.display_node()))
        })?;
        let alpha = self.alpha;
        let local_grad = value.where_with_f32(
            |y| y > 0.0,
            |_| 1.0,
            |y| y + alpha, // elu'(x) = elu(x) + alpha = value + alpha 当 x <= 0
        );
        Ok(GradResult::Computed(upstream_grad * &local_grad))
    }

    fn grad(&self) -> Option<&Tensor> { self.grad.as_ref() }
    fn set_grad(&mut self, grad: Option<&Tensor>) -> Result<(), GraphError> {
        self.grad = grad.cloned();
        Ok(())
    }

    fn grad_mut(&mut self) -> Option<&mut Tensor> {
        self.grad.as_mut()
    }
    fn clear_value(&mut self) -> Result<(), GraphError> { self.value = None; Ok(()) }
    fn set_value_unchecked(&mut self, value: Option<&Tensor>) { self.value = value.cloned(); }
}
