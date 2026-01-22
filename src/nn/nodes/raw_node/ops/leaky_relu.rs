use crate::nn::shape::DynamicShape;
use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::tensor::Tensor;

/// Leaky `ReLU` 激活函数节点
///
/// forward: f(x) = x if x > 0, else `negative_slope` * x
/// backward: d(f)/dx = 1 if x > 0, else `negative_slope`
///
/// 当 `negative_slope` = 0 时，等价于标准 `ReLU`
/// `MatrixSlow` 使用 `negative_slope` = 0.1
#[derive(Clone)]
pub(crate) struct LeakyReLU {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 固定形状（用于 value_expected_shape）
    fixed_shape: Vec<usize>,
    /// 动态形状（支持动态 batch）
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    supports_dynamic: bool,
    /// 负半轴斜率，默认 0.0（标准 `ReLU`）
    negative_slope: f64,
}

impl LeakyReLU {
    /// 获取 `negative_slope（alpha）值`
    pub(crate) const fn alpha(&self) -> f64 {
        self.negative_slope
    }

    pub(crate) fn new(parents: &[&NodeHandle], negative_slope: f64) -> Result<Self, GraphError> {
        // 1. 必要的验证
        // 1.1 父节点数量验证
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "LeakyReLU节点只需要1个父节点".to_string(),
            ));
        }

        // 1.2 negative_slope 验证（通常应该是非负小数）
        if negative_slope < 0.0 {
            return Err(GraphError::InvalidOperation(format!(
                "LeakyReLU的negative_slope应为非负数，但得到: {negative_slope}"
            )));
        }

        // 2. 从父节点继承动态形状信息
        let parent = &parents[0];
        let fixed_shape = parent.value_expected_shape().to_vec();
        let dynamic_shape = parent.dynamic_expected_shape();
        let supports_dynamic = parent.supports_dynamic_batch();

        // 3. 返回
        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape,
            dynamic_shape,
            supports_dynamic,
            negative_slope,
        })
    }
}

impl TraitNode for LeakyReLU {
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
        // 1. 获取父节点的值
        let parent_value = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的父节点{}没有值。不该触及本错误，否则说明crate代码有问题",
                self.display_node(),
                parents[0]
            ))
        })?;

        // 2. 计算 LeakyReLU: f(x) = x if x > 0, else negative_slope * x
        // 注：不再缓存 parent_value，因为梯度计算已改为使用 value（输出）判断区域
        let slope = self.negative_slope as f32;
        let result = parent_value.where_with_f32(
            |x| x > 0.0,
            |x| x,         // x > 0 时保持原值
            |x| slope * x, // x <= 0 时乘以 slope
        );
        self.value = Some(result);

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
        // LeakyReLU 的梯度: upstream_grad * (1 if x > 0 else negative_slope)
        //
        // 重要：使用 value（输出）而非 parent_value（输入）判断区域
        // 这对 BPTT 很关键，因为 BPTT 只恢复 value，不恢复 parent_value
        // 数学上等价：output > 0 ⟺ input > 0（当 slope >= 0 时）
        let value = self.value().ok_or_else(|| {
            GraphError::ComputationError(format!("{}没有值，无法计算梯度", self.display_node()))
        })?;

        // 计算局部梯度（逐元素）
        let slope = self.negative_slope as f32;
        let local_grad = value.where_with_f32(
            |y| y > 0.0,
            |_| 1.0,   // y > 0 时导数为 1
            |_| slope, // y <= 0 时导数为 slope
        );

        // 逐元素乘以上游梯度
        Ok(upstream_grad * &local_grad)
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
