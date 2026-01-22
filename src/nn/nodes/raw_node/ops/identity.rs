/*
 * Identity 节点（恒等映射）
 *
 * forward: y = x（直接传递父节点的值）
 * backward: 直接传递上游梯度（局部梯度 = 1）
 *
 * # 用途
 *
 * Identity 节点通过 `Var::detach_node()` 创建，用于在计算图中建立**显式的梯度截断边界**。
 *
 * ## 典型使用场景
 *
 * 1. **需要在 detach 后继续构建图**
 *    ```ignore
 *    let x_detached = x.detach_node();  // 创建 Identity 节点
 *    let y = x_detached.sigmoid();      // 可以继续构建图
 *    let z = y.matmul(&w);              // 继续链式操作
 *    ```
 *
 * 2. **调试/可视化时需要看到明确的 detach 边界**
 *    Identity 节点在 Graphviz 中显示为独立节点（椭圆形，虚线，浅紫色），
 *    方便定位梯度截断位置。
 *
 * 3. **迁移学习/多任务学习**
 *    冻结部分网络时，可以在共享特征提取器后添加 Identity 节点，
 *    使不同任务头有独立的梯度流控制。
 *
 * ## 与 DetachedVar 的区别
 *
 * | 方法 | 返回类型 | 是否创建图节点 | 适用场景 |
 * |------|---------|--------------|---------|
 * | `var.detach()` | `DetachedVar` | ❌ | ModelState.forward() |
 * | `var.detach_node()` | `Var` | ✅ Identity | 直接图操作、可视化 |
 *
 * 对于绝大多数场景（包括 GAN 训练），直接使用 `detach()` 即可，无需使用本节点。
 * `detach_node()` 仅用于需要在 detach 后继续构建图的高级场景。
 *
 * # 可视化
 *
 * Identity 节点使用特殊样式：椭圆形、虚线边框、浅紫色背景。
 * 这表明它是用户有意识创建的 detach 边界，区别于内部使用的 GradientRouter（灰色）。
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Identity 节点（恒等映射）
///
/// 直接传递父节点的值，不做任何变换。
/// 通过 `Var::detach_node()` 创建，用于在图中建立显式的梯度截断边界。
///
/// # 何时使用
///
/// - 需要在 detach 后继续对结果进行图操作（如 `x.detach_node().sigmoid()`）
/// - 需要在可视化中看到明确的 detach 边界
/// - 迁移学习/多任务学习中的梯度流控制
///
/// # 何时不使用
///
/// - 仅需要将 detached 值传入 `ModelState::forward()` → 使用 `var.detach()` 返回的 `DetachedVar`
///
/// # 可视化
///
/// 椭圆形、虚线边框、浅紫色背景（`#E1BEE7`）
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
    supports_dynamic: bool,
}

impl Identity {
    pub(crate) fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        // 验证：必须有且仅有 1 个父节点
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "Identity 节点只需要 1 个父节点".to_string(),
            ));
        }

        // 从父节点继承动态形状信息
        let parent = &parents[0];
        let fixed_shape = parent.value_expected_shape().to_vec();
        let dynamic_shape = parent.dynamic_expected_shape();
        let supports_dynamic = parent.supports_dynamic_batch();

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape,
            dynamic_shape,
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

    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
        // 直接复制父节点的值
        let parent_value = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{} 的父节点 {} 没有值",
                self.display_node(),
                parents[0]
            ))
        })?;

        self.value = Some(parent_value.clone());
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_grad_to_parent(
        &self,
        _target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // 直接传递上游梯度（Identity 的局部梯度是 1）
        Ok(upstream_grad.clone())
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
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
