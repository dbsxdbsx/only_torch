use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// WhereCond 节点：条件选择（类似 `torch.where(condition, x, y)`）
///
/// condition 是构建时传入的布尔掩码张量（不参与梯度），
/// x（parent 0）和 y（parent 1）是参与计算图的 Var 父节点。
///
/// 前向：`output[i] = condition[i] ? x[i] : y[i]`
/// 反向：
/// - `grad_x = condition * upstream_grad`
/// - `grad_y = (1 - condition) * upstream_grad`
#[derive(Clone)]
pub(crate) struct WhereCond {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 归一化后的条件掩码（严格 0/1）
    condition: Tensor,
    /// 固定形状
    fixed_shape: Vec<usize>,
    /// 动态形状
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    #[allow(dead_code)]
    supports_dynamic: bool,
}

impl WhereCond {
    /// 从父节点形状和 condition 张量创建 WhereCond 节点
    ///
    /// # 参数
    /// - `x_shape`: x（parent 0）的形状
    /// - `x_dynamic_shape`: x 的动态形状
    /// - `condition`: 条件张量（非零视为 true），形状必须与 x 一致
    pub(in crate::nn) fn new(
        x_shape: &[usize],
        x_dynamic_shape: &DynamicShape,
        condition: Tensor,
    ) -> Result<Self, GraphError> {
        // 验证 condition 与 x 形状一致
        if condition.shape() != x_shape {
            return Err(GraphError::ShapeMismatch {
                expected: x_shape.to_vec(),
                got: condition.shape().to_vec(),
                message: "WhereCond: condition 形状必须与 x 一致".to_string(),
            });
        }

        // 归一化 condition 为严格 0/1
        let condition = condition.where_with_f32(|c| c != 0.0, |_| 1.0, |_| 0.0);

        let supports_dynamic = x_dynamic_shape.dims().first() == Some(&None);

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            condition,
            fixed_shape: x_shape.to_vec(),
            dynamic_shape: x_dynamic_shape.clone(),
            supports_dynamic,
        })
    }
}

impl TraitNode for WhereCond {
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
        None // condition 含运行时数据，不参与 CSE 去重
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        // parent_values[0] = x, parent_values[1] = y
        self.value = Some(Tensor::where_mask(
            &self.condition,
            parent_values[0],
            parent_values[1],
        ));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    fn calc_grad_to_parent(
        &self,
        target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        match target_parent_index {
            // grad_x = condition * upstream_grad
            0 => Ok(GradResult::Computed(&self.condition * upstream_grad)),
            // grad_y = (1 - condition) * upstream_grad
            1 => {
                let inv_cond = Tensor::ones(self.condition.shape()) - &self.condition;
                Ok(GradResult::Computed(&inv_cond * upstream_grad))
            }
            _ => Err(GraphError::InvalidOperation(format!(
                "WhereCond: parent_index {} 无效（应为 0 或 1）",
                target_parent_index
            ))),
        }
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
