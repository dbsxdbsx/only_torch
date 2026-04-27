/*
 * @Author       : 老董
 * @Description  : ReLU6 激活节点
 *                 实现 min(max(0, x), 6)
 *                 移动端和量化网络常用
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// ReLU6 激活节点
///
/// forward: y = min(max(0, x), 6)
/// backward: dy/dx = 1 if 0 < x < 6, else 0
#[derive(Clone)]
pub(crate) struct ReLU6 {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    fixed_shape: Vec<usize>,
    dynamic_shape: DynamicShape,
    #[allow(dead_code)]
    supports_dynamic: bool,
}

impl ReLU6 {
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

impl TraitNode for ReLU6 {
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
        self.value = Some(parent_values[0].relu6());
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// VJP: grad = upstream * (1 if 0 < x < 6 else 0)
    /// 使用输出值判断区域（与 ReLU 一致的优化策略）
    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        let value = self.value().ok_or_else(|| {
            GraphError::ComputationError(format!("{}没有值", self.display_node()))
        })?;
        // y > 0 && y < 6 → 梯度通过；否则为 0
        Ok(GradResult::Computed(upstream_grad.where_with_tensor(
            value,
            |_, y| y > 0.0 && y < 6.0,
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
