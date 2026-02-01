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
    /// 从父节点形状信息创建 Add 节点（核心实现）
    ///
    /// # 参数
    /// - `parent_shapes`: 父节点的固定形状列表
    /// - `parent_dynamic_shapes`: 父节点的动态形状列表
    pub(in crate::nn) fn new_from_shapes(
        parent_shapes: &[&[usize]],
        parent_dynamic_shapes: &[DynamicShape],
    ) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parent_shapes.len() < 2 {
            return Err(GraphError::InvalidOperation(
                "Add节点至少需要2个父节点".to_string(),
            ));
        }
        if parent_shapes.len() != parent_dynamic_shapes.len() {
            return Err(GraphError::InvalidOperation(
                "父节点形状数量与动态形状数量不匹配".to_string(),
            ));
        }

        // 2. 计算广播后的固定形状
        let mut fixed_shape = parent_shapes[0].to_vec();
        for parent_shape in parent_shapes.iter().skip(1) {
            fixed_shape = broadcast_shape(&fixed_shape, parent_shape).ok_or_else(|| {
                GraphError::ShapeMismatch {
                    expected: fixed_shape.clone(),
                    got: parent_shape.to_vec(),
                    message: "Add节点的父节点形状无法广播".to_string(),
                }
            })?;
        }

        // 3. 计算动态形状
        let supports_dynamic = parent_dynamic_shapes
            .iter()
            .any(|ds| ds.has_dynamic_dims());
        let mut dynamic_shape = parent_dynamic_shapes[0].clone();
        for parent_dyn in parent_dynamic_shapes.iter().skip(1) {
            dynamic_shape = dynamic_shape.broadcast_with(parent_dyn);
        }

        // 4. 返回
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

    /// 从 NodeHandle 创建（过渡期 API，委托给 new_from_shapes）
    pub(crate) fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        // 提取形状信息
        let shapes: Vec<Vec<usize>> = parents
            .iter()
            .map(|p| p.value_expected_shape().to_vec())
            .collect();
        let shapes_ref: Vec<&[usize]> = shapes.iter().map(|s| s.as_slice()).collect();
        let dynamic_shapes: Vec<DynamicShape> =
            parents.iter().map(|p| p.dynamic_expected_shape()).collect();

        // 委托给核心实现
        Self::new_from_shapes(&shapes_ref, &dynamic_shapes)
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

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        // 逐元素相加（支持广播）
        let mut result = None;
        for &parent_value in parent_values {
            match &mut result {
                None => result = Some(parent_value.clone()),
                Some(sum) => {
                    *sum += parent_value;
                }
            }
        }
        self.value = result;
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
                GraphError::ComputationError(format!("Add 梯度计算时父{target_parent}没有值"))
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
