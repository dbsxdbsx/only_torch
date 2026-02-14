use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Mish 激活函数节点
///
/// forward: mish(x) = x * tanh(softplus(x))
/// backward: mish'(x) = tanh(sp) + x * sech^2(sp) * sigmoid(x)
///   其中 sp = softplus(x)
#[derive(Clone)]
pub(crate) struct Mish {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    fixed_shape: Vec<usize>,
    dynamic_shape: DynamicShape,
    #[allow(dead_code)]
    supports_dynamic: bool,
}

impl Mish {
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
}

impl TraitNode for Mish {
    fn id(&self) -> NodeId { self.id.unwrap() }
    fn set_id(&mut self, id: NodeId) { self.id = Some(id); }
    fn name(&self) -> &str { self.name.as_ref().unwrap() }
    fn set_name(&mut self, name: &str) { self.name = Some(name.to_string()); }
    fn value_expected_shape(&self) -> &[usize] { &self.fixed_shape }
    fn dynamic_expected_shape(&self) -> DynamicShape { self.dynamic_shape.clone() }
    fn supports_dynamic_batch(&self) -> bool { self.supports_dynamic }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        self.value = Some(parent_values[0].mish());
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> { self.value.as_ref() }

    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<Tensor, GraphError> {
        // mish'(x) = tanh(sp) + x * sech^2(sp) * sigmoid(x)
        // sp = softplus(x)
        const THRESHOLD: f32 = 20.0;
        let x = parent_values[0];
        let local_grad = x.where_with_f32(
            |_| true,
            |x_val| {
                // 数值稳定的 softplus
                let sp = if x_val > THRESHOLD {
                    x_val
                } else if x_val > 0.0 {
                    x_val + (-x_val).exp().ln_1p()
                } else {
                    x_val.exp().ln_1p()
                };
                let tanh_sp = sp.tanh();
                let sech2_sp = 1.0 - tanh_sp * tanh_sp;
                let sig = 1.0 / (1.0 + (-x_val).exp());
                tanh_sp + x_val * sech2_sp * sig
            },
            |_| unreachable!(),
        );
        Ok(upstream_grad * &local_grad)
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
