/*
 * @Author       : 老董
 * @Description  : HardTanh 激活节点
 *                 实现 min(max(min_val, x), max_val)
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// HardTanh 激活节点
///
/// forward: y = min(max(min_val, x), max_val)
/// backward: dy/dx = 1 if min_val < x < max_val, else 0
#[derive(Clone)]
pub(crate) struct HardTanh {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    fixed_shape: Vec<usize>,
    dynamic_shape: DynamicShape,
    #[allow(dead_code)]
    supports_dynamic: bool,
    min_val: f32,
    max_val: f32,
}

impl HardTanh {
    pub(crate) const fn min_val(&self) -> f32 {
        self.min_val
    }
    pub(crate) const fn max_val(&self) -> f32 {
        self.max_val
    }

    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
        min_val: f32,
        max_val: f32,
    ) -> Result<Self, GraphError> {
        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape: parent_shape.to_vec(),
            dynamic_shape: parent_dynamic_shape.clone(),
            supports_dynamic: parent_dynamic_shape.has_dynamic_dims(),
            min_val,
            max_val,
        })
    }
}

impl TraitNode for HardTanh {
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
        self.value = Some(parent_values[0].hard_tanh(self.min_val, self.max_val));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// VJP: grad = upstream * (1 if min_val < x < max_val else 0)
    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        let value = self.value().ok_or_else(|| {
            GraphError::ComputationError(format!("{}没有值", self.display_node()))
        })?;
        let min_val = self.min_val;
        let max_val = self.max_val;
        Ok(GradResult::Computed(upstream_grad.where_with_tensor(
            value,
            |_, y| y > min_val && y < max_val,
            |g, _| g,
            |_, _| 0.0,
        )))
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
