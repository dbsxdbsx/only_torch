use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

use super::Reduction;

/// MAE（Mean Absolute Error，平均绝对误差）损失节点
///
/// 计算预测值和目标值之间的平均绝对误差损失。
/// 相比 MSE，MAE 对异常值更鲁棒，梯度恒定。
///
/// ## 公式
/// - Mean reduction: `MAE = mean(|input - target|) = sum(|input - target|) / N`
/// - Sum reduction: `MAE = sum(|input - target|)`
///
/// ## 梯度
/// - Mean: `∂L/∂input = sign(input - target) / N`
/// - Sum: `∂L/∂input = sign(input - target)`
///
/// ## 输入
/// - 父节点 0: input（预测值）
/// - 父节点 1: target（目标值）
///
/// ## 输出
/// - 标量损失值 [1, 1]
///
/// ## 参考
/// - `PyTorch`: `torch.nn.L1Loss`
#[derive(Clone)]
pub(crate) struct MAE {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 输出形状固定为 [1, 1]（标量损失）
    shape: Vec<usize>,
    /// Reduction 模式: "mean" 或 "sum"
    reduction: Reduction,
    /// 缓存 input - target，用于反向传播计算 sign
    diff_cache: Option<Tensor>,
    /// 缓存元素总数，用于 mean reduction
    numel_cache: usize,
    /// 父节点 ID，用于区分 input 和 target
    #[allow(dead_code)]
    parents_ids: Vec<NodeId>,
}

impl MAE {
    /// 获取 reduction 模式
    #[allow(dead_code)]
    pub(crate) fn reduction(&self) -> Reduction {
        self.reduction
    }

    /// 从父节点形状信息创建 MAE 节点（核心实现）
    pub(in crate::nn) fn new(
        input_shape: &[usize],
        target_shape: &[usize],
        input_dynamic_shape: &DynamicShape,
        target_dynamic_shape: &DynamicShape,
        parent_ids: Vec<NodeId>,
        reduction: Reduction,
    ) -> Result<Self, GraphError> {
        // 验证形状兼容性
        if !input_dynamic_shape.is_compatible(target_dynamic_shape) {
            return Err(GraphError::ShapeMismatch {
                expected: input_shape.to_vec(),
                got: target_shape.to_vec(),
                message: "input 和 target 动态形状必须兼容".to_string(),
            });
        }

        let numel: usize = input_shape.iter().product();

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            shape: vec![1, 1],
            reduction,
            diff_cache: None,
            numel_cache: numel,
            parents_ids: parent_ids,
        })
    }
}

impl TraitNode for MAE {
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
        &self.shape
    }

    fn calc_value_by_parents(&mut self, parent_values: &[&Tensor]) -> Result<(), GraphError> {
        let input = parent_values[0];
        let target = parent_values[1];
        // 计算 diff = input - target
        let diff = input - target;
        // 计算 abs_diff = |diff|
        let abs_diff = diff.abs();
        // 更新 numel（支持动态 batch size）
        self.numel_cache = input.size();
        // 缓存 diff 用于反向传播（计算 sign）
        self.diff_cache = Some(diff);
        // 根据 reduction 模式计算损失
        let loss_value = match self.reduction {
            Reduction::Mean => {
                let sum_tensor = abs_diff.sum();
                sum_tensor.get_data_number().unwrap() / (self.numel_cache as f32)
            }
            Reduction::Sum => abs_diff.sum().get_data_number().unwrap(),
        };
        self.value = Some(Tensor::new(&[loss_value], &[1, 1]));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// MAE 的 VJP 梯度计算
    ///
    /// 对于 input: [batch, features]，target: [batch, features]：
    /// - Mean: `dL/d_input` = sign(input - target) / N（N 是总元素数）
    /// - Sum: `dL/d_input` = sign(input - target)
    fn calc_grad_to_parent(
        &self,
        target_parent_index: usize,
        _parent_values: &[&Tensor],
        _upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        let diff = self.diff_cache.as_ref().ok_or_else(|| {
            GraphError::ComputationError("diff 缓存为空，需先执行前向传播".to_string())
        })?;

        if target_parent_index == 0 {
            // 对 input 的梯度: sign(diff) / N 或 sign(diff)
            let sign_diff = diff.sign();
            let grad = match self.reduction {
                Reduction::Mean => &sign_diff * (1.0 / self.numel_cache as f32),
                Reduction::Sum => sign_diff,
            };
            Ok(GradResult::Computed(grad))
        } else {
            // 对 target 的梯度（通常不需要）
            Err(GraphError::InvalidOperation(
                "不应该对 target 计算梯度".to_string(),
            ))
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
