use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::tensor::Tensor;

/// MSE（均方误差）损失节点
///
/// 计算预测值和目标值之间的均方误差损失。
///
/// ## 公式
/// - Mean reduction: `MSE = mean((input - target)^2) = sum((input - target)^2) / N`
/// - Sum reduction: `MSE = sum((input - target)^2)`
///
/// ## 梯度
/// - Mean: `∂L/∂input = 2 * (input - target) / N`
/// - Sum: `∂L/∂input = 2 * (input - target)`
///
/// ## 输入
/// - 父节点 0: input（预测值）
/// - 父节点 1: target（目标值）
///
/// ## 输出
/// - 标量损失值 [1, 1]
///
/// ## 参考
/// - `PyTorch`: `torch.nn.MSELoss`
#[derive(Clone)]
pub(crate) struct MSELoss {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 输出形状固定为 [1, 1]（标量损失）
    shape: Vec<usize>,
    /// Reduction 模式: "mean" 或 "sum"
    reduction: Reduction,
    /// 缓存 input - target，用于反向传播
    diff_cache: Option<Tensor>,
    /// 缓存元素总数，用于 mean reduction
    numel_cache: usize,
    /// 父节点 ID，用于区分 input 和 target
    parents_ids: Vec<NodeId>,
}

/// Reduction 模式
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reduction {
    /// 对所有元素求平均（默认）
    Mean,
    /// 对所有元素求和
    Sum,
}

impl MSELoss {
    pub(crate) fn new(parents: &[&NodeHandle], reduction: Reduction) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parents.len() != 2 {
            return Err(GraphError::InvalidOperation(
                "MSELoss 节点需要 2 个父节点（input 和 target）".to_string(),
            ));
        }

        // 2. 验证形状兼容性（使用动态形状比较，支持动态 batch）
        let input_dyn_shape = parents[0].dynamic_expected_shape();
        let target_dyn_shape = parents[1].dynamic_expected_shape();
        if !input_dyn_shape.is_compatible(&target_dyn_shape) {
            return Err(GraphError::ShapeMismatch {
                expected: parents[0].value_expected_shape().to_vec(),
                got: parents[1].value_expected_shape().to_vec(),
                message: "input 和 target 动态形状必须兼容".to_string(),
            });
        }

        // 计算元素总数（使用固定形状作为初始估计，运行时会更新）
        let input_shape = parents[0].value_expected_shape();
        let numel: usize = input_shape.iter().product();

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            shape: vec![1, 1], // 损失是标量
            reduction,
            diff_cache: None,
            numel_cache: numel,
            parents_ids: vec![parents[0].id(), parents[1].id()],
        })
    }

    /// 使用默认 Mean reduction 创建 `MSELoss`
    pub(crate) fn new_mean(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        Self::new(parents, Reduction::Mean)
    }
}

impl TraitNode for MSELoss {
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

    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
        // 获取 input 和 target
        let input = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的 input 父节点{}没有值",
                self.display_node(),
                parents[0]
            ))
        })?;
        let target = parents[1].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的 target 父节点{}没有值",
                self.display_node(),
                parents[1]
            ))
        })?;

        // 计算 diff = input - target
        let diff = input - target;

        // 计算 squared_diff = diff^2
        let squared_diff = &diff * &diff;

        // 更新 numel（支持动态 batch size）
        self.numel_cache = input.size();

        // 缓存 diff 用于反向传播
        self.diff_cache = Some(diff);

        // 根据 reduction 模式计算损失
        let loss_value = match self.reduction {
            Reduction::Mean => {
                let sum_tensor = squared_diff.sum();
                sum_tensor.get_data_number().unwrap() / (self.numel_cache as f32)
            }
            Reduction::Sum => squared_diff.sum().get_data_number().unwrap(),
        };

        self.value = Some(Tensor::new(&[loss_value], &[1, 1]));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// `MSELoss` 的 VJP 梯度计算
    ///
    /// 对于 input: [batch, features]，target: [batch, features]：
    /// - Mean: `dL/d_input` = 2 * (input - target) / N（N 是总元素数）
    /// - Sum: `dL/d_input` = 2 * (input - target)
    fn calc_grad_to_parent(
        &self,
        target_parent: &NodeHandle,
        _upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        let diff = self.diff_cache.as_ref().ok_or_else(|| {
            GraphError::ComputationError("diff 缓存为空，需先执行前向传播".to_string())
        })?;

        if target_parent.id() == self.parents_ids[0] {
            // 对 input 的梯度
            let grad = match self.reduction {
                Reduction::Mean => diff * (2.0 / self.numel_cache as f32),
                Reduction::Sum => diff * 2.0,
            };
            Ok(grad)
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
