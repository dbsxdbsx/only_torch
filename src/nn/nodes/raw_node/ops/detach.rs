/*
 * Detach 节点（梯度屏障）
 *
 * forward: y = x（直接传递父节点的值）
 * backward: 阻断梯度传播（不向上游传递梯度）
 *
 * # 与 Identity 的区别
 *
 * | 节点 | forward | backward | 用途 |
 * |------|---------|----------|------|
 * | Identity | y = x | 透传梯度 | pass-through / NEAT 占位 |
 * | Detach   | y = x | 阻断梯度 | 显式梯度截断边界 |
 *
 * # 用途
 *
 * Detach 节点通过 `Var::detach()` 创建，用于在计算图中建立**显式的梯度截断边界**。
 *
 * ## 典型使用场景
 *
 * 1. **GAN 训练中阻止梯度流向生成器**
 *    ```ignore
 *    let fake = generator.forward(&noise)?;
 *    let d_out = discriminator.forward(fake.detach())?;  // 梯度阻断
 *    ```
 *
 * 2. **调试/可视化时看到明确的 detach 边界**
 *    Detach 节点在 Graphviz 中显示为独立节点（椭圆形，虚线，浅紫色）。
 *
 * 3. **迁移学习/多任务学习**
 *    冻结部分网络时，在共享特征提取器后添加 detach，
 *    使不同任务头有独立的梯度流控制。
 *
 * # 可视化
 *
 * Detach 节点使用特殊样式：椭圆形、虚线边框、浅紫色背景。
 * 表明用户有意识创建的梯度截断边界。
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// Detach 节点（梯度屏障）
///
/// 前向传播：直接传递父节点的值（与 Identity 相同）。
/// 反向传播：阻断梯度传播，不向上游传递任何梯度。
///
/// 通过 `Var::detach()` 创建。
///
/// # 可视化
///
/// 椭圆形、虚线边框、浅紫色背景（`#E1BEE7`）
#[derive(Clone)]
pub(crate) struct Detach {
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

impl Detach {
    /// 从父节点形状信息创建 Detach 节点
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

impl TraitNode for Detach {
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

    fn dedup_fingerprint(&self) -> Option<u64> {
        None // Detach 是梯度控制节点，每次调用必须创建独立梯度边界
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        // 与 Identity 相同：直接复制父节点的值
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
        _upstream_grad: &Tensor,
    ) -> Result<Tensor, GraphError> {
        // Detach 的核心：不应该向上游传播梯度
        Err(GraphError::InvalidOperation(format!(
            "{}不应该向上游传播梯度（Detach 节点是梯度屏障）",
            self.display_node()
        )))
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
