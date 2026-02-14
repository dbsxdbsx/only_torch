/*
 * @Author       : 老董
 * @Date         : 2026-02-15
 * @Description  : Pow（幂运算）节点
 *                 实现逐元素幂运算: y = x^p
 *
 * 指数 p 为常量（非计算图变量），在创建节点时确定。
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// 幂运算节点
///
/// forward: y = x^p
/// backward: dy/dx = p * x^(p-1)
///
/// ## 输入
/// - 父节点：任意形状的张量
/// - 指数 p：f32 常量
///
/// ## 输出
/// - 与输入形状相同
#[derive(Clone)]
pub(crate) struct Pow {
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
    /// 常量指数
    exponent: f32,
    /// 缓存输入值，用于反向传播
    input_cache: Option<Tensor>,
}

impl Pow {
    /// 从父节点形状信息创建 Pow 节点
    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
        exponent: f32,
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
            exponent,
            input_cache: None,
        })
    }
}

impl TraitNode for Pow {
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
        // 缓存输入用于反向传播
        self.input_cache = Some(parent_values[0].clone());
        // 计算 x^p
        self.value = Some(parent_values[0].powf(self.exponent));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// Pow 反向传播的 VJP 计算
    ///
    /// 对于 y = x^p，有：
    /// dy/dx = p * x^(p-1)
    ///
    /// VJP: grad_to_parent = upstream_grad * p * x^(p-1)
    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        let input = self.input_cache.as_ref().ok_or_else(|| {
            GraphError::ComputationError("Pow 输入缓存为空，需先执行前向传播".to_string())
        })?;

        // grad = upstream_grad * p * x^(p-1)
        let grad = upstream_grad * &(input.powf(self.exponent - 1.0) * self.exponent);
        Ok(GradResult::Computed(grad))
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
