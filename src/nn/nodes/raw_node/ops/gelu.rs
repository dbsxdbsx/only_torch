use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// GELU 激活函数节点（tanh 近似版，GPT-2 风格）
///
/// forward: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// backward: 从 parent_values 计算（公式无法从 value 反推）
#[derive(Clone)]
pub(crate) struct Gelu {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    fixed_shape: Vec<usize>,
    dynamic_shape: DynamicShape,
    #[allow(dead_code)]
    supports_dynamic: bool,
}

impl Gelu {
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

impl TraitNode for Gelu {
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
        self.value = Some(parent_values[0].gelu());
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        // GELU 梯度必须从 parent_values（输入 x）计算，无法从 value 反推
        // gelu'(x) = 0.5*(1+tanh(z)) + x*0.5*(1-tanh(z)^2)*sqrt(2/pi)*(1+3*0.044715*x^2)
        // 其中 z = sqrt(2/pi)*(x + 0.044715*x^3)
        const SQRT_2_OVER_PI: f32 = 0.7978845608;
        const COEFF: f32 = 0.044715;

        let x = parent_values[0];
        let local_grad = x.where_with_f32(
            |_| true, // 对所有元素统一计算
            |x_val| {
                let z = SQRT_2_OVER_PI * (x_val + COEFF * x_val * x_val * x_val);
                let tanh_z = z.tanh();
                let sech2_z = 1.0 - tanh_z * tanh_z;
                let dz_dx = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x_val * x_val);
                0.5 * (1.0 + tanh_z) + x_val * 0.5 * sech2_z * dz_dx
            },
            |_| unreachable!(),
        );

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
