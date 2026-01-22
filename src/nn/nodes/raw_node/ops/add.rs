use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::shape::DynamicShape;
use crate::tensor::{Tensor, broadcast_shape};

#[derive(Clone)]
pub(crate) struct Add {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 固定形状（用于 `value_expected_shape`）
    fixed_shape: Vec<usize>,
    /// 动态形状（支持动态 batch）
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch（继承自父节点）
    supports_dynamic: bool,
}

impl Add {
    pub(crate) fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        // 1. 必要的验证
        // 1.1 父节点数量验证
        if parents.len() < 2 {
            return Err(GraphError::InvalidOperation(
                "Add节点至少需要2个父节点".to_string(),
            ));
        }

        // 1.2 计算广播后的固定形状
        let mut fixed_shape = parents[0].value_expected_shape().to_vec();

        for parent in parents.iter().skip(1) {
            let parent_shape = parent.value_expected_shape();

            // 使用 broadcast_shape 计算广播后的形状
            fixed_shape = broadcast_shape(&fixed_shape, parent_shape).ok_or_else(|| {
                GraphError::ShapeMismatch {
                    expected: fixed_shape.clone(),
                    got: parent_shape.to_vec(),
                    message: "Add节点的父节点形状无法广播".to_string(),
                }
            })?;
        }

        // 1.3 计算动态形状（使用父节点的动态形状）
        // 如果任一父节点支持动态 batch，输出也支持
        let supports_dynamic = parents.iter().any(|p| p.supports_dynamic_batch());

        // 合并所有父节点的动态形状
        let mut dynamic_shape = parents[0].dynamic_expected_shape();
        for parent in parents.iter().skip(1) {
            let parent_dyn = parent.dynamic_expected_shape();
            // 对于 Add，输出形状是广播后的形状
            // 简化处理：如果任一维度是动态的，输出该维度也是动态的
            dynamic_shape = dynamic_shape.broadcast_with(&parent_dyn);
        }

        // 2. 返回
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

impl TraitNode for Add {
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
        // 1. 从图中获取父节点的值
        let mut result = None;
        for parent in parents {
            let parent_value = parent.value().ok_or_else(|| {
                GraphError::ComputationError(format!(
                    "{}的父节点{}没有值。不该触及本错误，否则说明crate代码有问题",
                    self.display_node(),
                    parent
                ))
            })?;

            // 1.2 添加到结果中
            match &mut result {
                None => result = Some(parent_value.clone()),
                Some(sum) => {
                    *sum += parent_value;
                }
            }
        }

        // 2. 将结果赋值给当前节点
        self.value = result;

        // 3. 返回
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_grad_to_parent(
        &self,
        target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        // Add 节点的局部梯度是 identity，但需要处理广播
        // 如果父节点被广播过，需要将梯度沿广播维度求和
        //
        // 注意：使用实际值的形状（支持动态 batch），而不是 value_expected_shape
        let target_shape = target_parent
            .value()
            .ok_or_else(|| {
                GraphError::ComputationError(format!("Add 梯度计算时父节点 {target_parent} 没有值"))
            })?
            .shape();

        if upstream_grad.shape() == target_shape {
            // 形状匹配，直接传递
            Ok(upstream_grad.clone())
        } else {
            // 被广播过，需要对广播维度求和
            Ok(upstream_grad.sum_to_shape(target_shape))
        }
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
