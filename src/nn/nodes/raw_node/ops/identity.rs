/*
 * Identity 节点（纯恒等映射）
 *
 * forward: y = x（直接传递父节点的值）
 * backward: 直接传递上游梯度（局部梯度 = 1）
 *
 * # 与 Detach 的区别
 *
 * | 节点 | forward | backward | 用途 |
 * |------|---------|----------|------|
 * | Identity | y = x | 透传梯度 | pass-through / NEAT 占位 / skip connection |
 * | Detach   | y = x | 阻断梯度 | 显式梯度截断边界（通过 `Var::detach()` 创建） |
 *
 * # 用途
 *
 * 1. **NEAT 演化**：作为 pass-through 空节点，后续可被变异为其他类型
 * 2. **Skip connection**：ResNet 残差连接中的恒等分支
 * 3. **Sequential 占位**：模块化构建中的 no-op 占位符
 *
 * # 可视化
 *
 * Identity 节点使用样式：椭圆形、虚线边框、浅灰色背景。
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Identity 节点（纯恒等映射）
///
/// 直接传递父节点的值，不做任何变换。
/// 反向传播时透传梯度（局部梯度 = 1）。
///
/// # 可视化
///
/// 椭圆形、虚线边框、浅灰色背景（`#E0E0E0`）
#[derive(Clone)]
pub(crate) struct Identity {
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

impl Identity {
    /// 从父节点形状信息创建 Identity 节点（核心实现）
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

impl TraitNode for Identity {
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
        // 直接复制父节点的值
        self.value = Some(parent_values[0].clone());
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
        // 直接传递上游梯度（Identity 的局部梯度是 1）
        Ok(upstream_grad.clone())
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
