/*
 * @Author       : 老董
 * @Date         : 2026-01-31
 * @Description  : Ln（自然对数）节点
 *                 实现逐元素自然对数: y = ln(x)
 *
 * 注意：输入 x 必须为正数，否则结果为 NaN 或 -Inf
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// 自然对数节点
///
/// forward: y = ln(x)
/// backward: dy/dx = 1/x
///
/// ## 输入
/// - 父节点：任意形状的张量（元素应为正数）
///
/// ## 输出
/// - 与输入形状相同
#[derive(Clone)]
pub(crate) struct Ln {
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
    /// 缓存输入值，用于反向传播
    input_cache: Option<Tensor>,
}

impl Ln {
    /// 从父节点形状信息创建 Ln 节点（核心实现）
    pub(in crate::nn) fn new_from_shapes(
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
            input_cache: None,
        })
    }

    /// 从 NodeHandle 创建（过渡期 API，委托给 new_from_shapes）
    pub(crate) fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "Ln 节点只需要 1 个父节点".to_string(),
            ));
        }

        Self::new_from_shapes(
            &parents[0].value_expected_shape(),
            &parents[0].dynamic_expected_shape(),
        )
    }
}

impl TraitNode for Ln {
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
        let input = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的父{}没有值",
                self.display_node(),
                parents[0]
            ))
        })?;

        // 缓存输入用于反向传播
        self.input_cache = Some(input.clone());

        // 计算 ln(x)
        self.value = Some(input.ln());
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// Ln 反向传播的 VJP 计算
    ///
    /// 对于 y = ln(x)，有：
    /// dy/dx = 1/x
    ///
    /// VJP: grad_to_parent = upstream_grad / x
    fn calc_grad_to_parent(
        &self,
        _target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        let input = self.input_cache.as_ref().ok_or_else(|| {
            GraphError::ComputationError("Ln 输入缓存为空，需先执行前向传播".to_string())
        })?;

        // grad = upstream_grad / x
        Ok(upstream_grad / input)
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
        self.input_cache = None;
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
