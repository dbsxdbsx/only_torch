/*
 * @Author       : 老董
 * @Date         : 2026-02-12
 * @Description  : Sqrt（平方根）节点
 *                 实现逐元素平方根: y = √x
 *
 * 注意：输入 x 应为非负数，否则结果为 NaN。
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// 平方根节点
///
/// forward: y = √x
/// backward: dy/dx = 0.5 / √x = 0.5 / y
///
/// ## 输入
/// - 父节点：任意形状的张量（元素应为非负数）
///
/// ## 输出
/// - 与输入形状相同
#[derive(Clone)]
pub(crate) struct Sqrt {
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
}

impl Sqrt {
    /// 从父节点形状信息创建 Sqrt 节点
    pub(in crate::nn) fn new(
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
        })
    }
}

impl TraitNode for Sqrt {
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
        // 计算 √x（委托给 Tensor::sqrt()）
        self.value = Some(parent_values[0].sqrt());
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// Sqrt 反向传播的 VJP 计算
    ///
    /// 对于 y = √x，有：
    /// dy/dx = 1 / (2√x) = 0.5 / y
    ///
    /// VJP: grad_to_parent = upstream_grad * 0.5 / √x
    ///
    /// 注意：利用前向输出 y=√x 避免重复计算
    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<Tensor, GraphError> {
        let output = self.value.as_ref().ok_or_else(|| {
            GraphError::ComputationError("Sqrt 前向值为空，需先执行前向传播".to_string())
        })?;

        // grad = upstream_grad * 0.5 / y，其中 y = √x
        let half_recip = Tensor::new(
            &output
                .data_as_slice()
                .iter()
                .map(|&y| 0.5 / y)
                .collect::<Vec<_>>(),
            output.shape(),
        );

        Ok(upstream_grad * &half_recip)
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
