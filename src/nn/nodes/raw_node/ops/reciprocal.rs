/*
 * @Author       : 老董
 * @Description  : Reciprocal（倒数）节点
 *                 实现逐元素倒数: y = 1/x
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// 倒数节点
///
/// forward: y = 1/x
/// backward: dy/dx = -1/x²
#[derive(Clone)]
pub(crate) struct Reciprocal {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    fixed_shape: Vec<usize>,
    dynamic_shape: DynamicShape,
    #[allow(dead_code)]
    supports_dynamic: bool,
    input_cache: Option<Tensor>,
}

impl Reciprocal {
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
            input_cache: None,
        })
    }
}

impl TraitNode for Reciprocal {
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
        self.input_cache = Some(parent_values[0].clone());
        self.value = Some(parent_values[0].reciprocal());
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// VJP: grad = upstream * (-1/x²)
    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        let input = self
            .input_cache
            .as_ref()
            .ok_or_else(|| GraphError::ComputationError("Reciprocal 输入缓存为空".to_string()))?;
        // -1/x² = -(1/x)²，用 input.square() 更高效
        let neg_inv_sq = &input.square().reciprocal() * (-1.0);
        Ok(GradResult::Computed(upstream_grad * &neg_inv_sq))
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
        self.input_cache = None;
        Ok(())
    }
    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
