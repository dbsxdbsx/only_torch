/*
 * GradientRouter 节点（梯度路由器）
 *
 * 这是一个特殊的输入节点，用于 ModelState 的智能缓存机制：
 * - 像 Input 节点一样存储值（通过 set_value 设置）
 * - 支持动态设置 is_detached 标志（控制梯度是否传播）
 * - 支持梯度路由：将自身梯度累加到指定的目标节点
 * - **支持动态 batch**：不同 batch_size 的值可以复用同一个 GradientRouter
 *
 * # 用途
 * ModelState 为每种特征形状创建一个 GradientRouter 作为模型的入口点。
 * 无论用户传入 Tensor 还是 Var，都复用同一个 GradientRouter：
 * - Tensor 输入：复制值，无梯度路由
 * - detached Var 输入：复制值，无梯度路由
 * - 非 detached Var 输入：复制值，设置梯度路由目标
 *
 * # 动态 Batch（类似 Keras）
 * GradientRouter 只验证特征维度匹配，忽略 batch 维度（第一维）：
 * - `[256, 64]` 和 `[1, 64]` 可以复用同一个 GradientRouter
 * - 可视化时 batch 维度显示为 `?`
 *
 * # 可视化
 * GradientRouter 是内部实现节点，在可视化时使用特殊样式（虚线、灰色）。
 */

use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::NodeId;
use crate::nn::shape::DynamicShape;
use crate::nn::GraphError;
use crate::tensor::Tensor;
use std::cell::RefCell;

/// GradientRouter 节点（梯度路由器）
///
/// 作为 ModelState 缓存结构的入口点，支持：
/// - 动态值更新（通过 set_value）
/// - 动态 detached 状态切换
/// - 梯度路由到外部目标节点
/// - **动态 batch**：第一维可以是任意值
#[derive(Clone)]
pub(crate) struct GradientRouter {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 动态形状：第一维是 None（动态 batch）
    dynamic_shape: DynamicShape,
    /// 用于 value_expected_shape 的固定形状缓存（首次调用时的形状）
    fixed_shape: Vec<usize>,
    /// 是否处于 detached 状态（阻止梯度传播）
    is_detached: RefCell<bool>,
    /// 梯度路由目标（backward 后将梯度累加到此节点）
    gradient_target: RefCell<Option<NodeId>>,
}

impl GradientRouter {
    /// 创建一个支持动态 batch 的 GradientRouter
    ///
    /// # 参数
    /// - `initial_shape`: 首次调用时的完整形状（如 `[256, 64]`）
    ///
    /// 内部会自动将第一维标记为动态，生成 `[?, 64]`。
    pub(crate) fn new(initial_shape: &[usize]) -> Self {
        // 创建动态形状：第一维是 None
        let dynamic_shape = if initial_shape.len() > 1 {
            DynamicShape::with_dynamic_batch(&initial_shape[1..])
        } else {
            DynamicShape::fixed(initial_shape)
        };

        Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            dynamic_shape,
            fixed_shape: initial_shape.to_vec(),
            is_detached: RefCell::new(false),
            gradient_target: RefCell::new(None),
        }
    }

    /// 从 DynamicShape 创建（预留 API，暂未使用）
    #[allow(dead_code)]
    pub(crate) fn with_dynamic_shape(shape: DynamicShape, initial_fixed: &[usize]) -> Self {
        Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            dynamic_shape: shape,
            fixed_shape: initial_fixed.to_vec(),
            is_detached: RefCell::new(false),
            gradient_target: RefCell::new(None),
        }
    }

    /// 设置 detached 状态
    pub(crate) fn set_detached(&self, detached: bool) {
        *self.is_detached.borrow_mut() = detached;
    }

    /// 获取 detached 状态
    pub(crate) fn is_detached(&self) -> bool {
        *self.is_detached.borrow()
    }

    /// 设置梯度路由目标
    ///
    /// 当 backward 计算出此节点的梯度后，会将梯度累加到目标节点。
    /// 设置为 None 表示不进行梯度路由。
    pub(crate) fn set_gradient_target(&self, target: Option<NodeId>) {
        *self.gradient_target.borrow_mut() = target;
    }

    /// 获取梯度路由目标
    pub(crate) fn gradient_target(&self) -> Option<NodeId> {
        *self.gradient_target.borrow()
    }
}

impl TraitNode for GradientRouter {
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
        true
    }

    fn calc_value_by_parents(
        &mut self,
        _parents: &[crate::nn::nodes::NodeHandle],
    ) -> Result<(), GraphError> {
        // GradientRouter 没有父节点，值通过 set_value 设置
        // 如果调用到这里，说明值还没设置
        if self.value.is_none() {
            return Err(GraphError::InvalidOperation(format!(
                "{} 是 GradientRouter 节点，其值应通过 set_value 设置",
                self.display_node()
            )));
        }
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn set_value(&mut self, value: Option<&Tensor>) -> Result<(), GraphError> {
        self.value = value.cloned();
        Ok(())
    }

    fn calc_grad_to_parent(
        &self,
        _target_parent: &crate::nn::nodes::NodeHandle,
        _upstream_grad: &Tensor,
        _assistant_parent: Option<&crate::nn::nodes::NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // GradientRouter 没有父节点，不需要计算对父节点的梯度
        // 梯度路由由 GraphInner 的 backward 逻辑处理
        Err(GraphError::InvalidOperation(
            "GradientRouter 没有父节点，不应计算父节点梯度".to_string(),
        ))
    }

    fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
    }

    fn set_grad(&mut self, grad: Option<&Tensor>) -> Result<(), GraphError> {
        self.grad = grad.cloned();
        Ok(())
    }

    fn clear_value(&mut self) -> Result<(), GraphError> {
        // GradientRouter 的值不应被清除（它是输入节点）
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
