use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

use super::Reduction;

/// Huber Loss（Smooth L1 Loss）损失节点
///
/// 结合 MSE（小误差）和 MAE（大误差）的优点，对异常值更鲁棒。
/// 当 |error| ≤ δ 时行为像 MSE，当 |error| > δ 时行为像 MAE。
///
/// ## 典型应用场景
/// - **强化学习**：DQN 等算法的 Q 值训练（δ=1.0 是标准配置）
/// - **带离群值的回归**：数据中存在异常值时
///
/// ## 公式
/// ```text
/// L(a) =
///   0.5 * a²              , if |a| ≤ δ
///   δ * |a| - 0.5 * δ²    , if |a| > δ
///
/// 其中 a = input - target
/// ```
///
/// ## 梯度
/// ```text
/// ∂L/∂a =
///   a              , if |a| ≤ δ    (线性，与 MSE 相同)
///   δ * sign(a)    , if |a| > δ    (常数 ±δ，被"裁剪")
/// ```
///
/// ## 输入
/// - 父节点 0: input（预测值）
/// - 父节点 1: target（目标值）
///
/// ## 输出
/// - 标量损失值 [1, 1]
///
/// ## 参考
/// - `PyTorch`: `torch.nn.HuberLoss`, `torch.nn.SmoothL1Loss`
/// - DQN 原论文的 "error clipping" 等价于 Huber Loss（δ=1）
#[derive(Clone)]
pub(crate) struct Huber {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 输出形状固定为 [1, 1]（标量损失）
    shape: Vec<usize>,
    /// Reduction 模式: Mean 或 Sum
    reduction: Reduction,
    /// δ 参数：小误差/大误差的分界阈值（默认 1.0）
    delta: f32,
    /// 缓存 input - target，用于反向传播
    diff_cache: Option<Tensor>,
    /// 缓存元素总数，用于 mean reduction
    numel_cache: usize,
    /// 父节点 ID，用于区分 input 和 target
    parents_ids: Vec<NodeId>,
}

/// Huber Loss 默认 δ 值（强化学习标准配置）
pub const DEFAULT_HUBER_DELTA: f32 = 1.0;

impl Huber {
    /// 从父节点形状信息创建 Huber 节点（核心实现）
    pub(in crate::nn) fn new(
        input_shape: &[usize],
        target_shape: &[usize],
        input_dynamic_shape: &DynamicShape,
        target_dynamic_shape: &DynamicShape,
        parent_ids: Vec<NodeId>,
        reduction: Reduction,
        delta: f32,
    ) -> Result<Self, GraphError> {
        // 验证 δ 参数
        if delta <= 0.0 {
            return Err(GraphError::InvalidOperation(format!(
                "Huber Loss 的 δ 参数必须为正数，当前值: {delta}"
            )));
        }

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
            delta,
            diff_cache: None,
            numel_cache: numel,
            parents_ids: parent_ids,
        })
    }


    /// 获取 δ 参数
    pub(crate) const fn delta(&self) -> f32 {
        self.delta
    }
}

impl TraitNode for Huber {
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
        // 更新 numel（支持动态 batch size）
        self.numel_cache = input.size();
        // 缓存 diff 用于反向传播
        self.diff_cache = Some(diff.clone());
        // 计算 Huber Loss
        let delta = self.delta;
        let half_delta_sq = 0.5 * delta * delta;
        let diff_view = diff.flatten_view();
        let loss_sum: f32 = diff_view
            .iter()
            .map(|&a| {
                let abs_a = a.abs();
                if abs_a <= delta {
                    0.5 * a * a // MSE 分支
                } else {
                    delta.mul_add(abs_a, -half_delta_sq) // MAE 分支
                }
            })
            .sum();
        // 根据 reduction 模式计算损失
        let loss_value = match self.reduction {
            Reduction::Mean => loss_sum / (self.numel_cache as f32),
            Reduction::Sum => loss_sum,
        };
        self.value = Some(Tensor::new(&[loss_value], &[1, 1]));
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// Huber Loss 的 VJP 梯度计算
    ///
    /// 对于 input: [batch, features]，target: [batch, features]：
    /// - |a| ≤ δ: `dL/d_input` = a / N（Mean）或 a（Sum）
    /// - |a| > δ: `dL/d_input` = δ * sign(a) / N（Mean）或 δ * sign(a)（Sum）
    fn calc_grad_to_parent(
        &self,
        target_parent_index: usize,
        _parent_values: &[&Tensor],
        _upstream_grad: &Tensor,
    ) -> Result<Tensor, GraphError> {
        let diff = self.diff_cache.as_ref().ok_or_else(|| {
            GraphError::ComputationError("diff 缓存为空，需先执行前向传播".to_string())
        })?;

        if target_parent_index == 0 {
            // 对 input 的梯度
            let delta = self.delta;
            let diff_view = diff.flatten_view();

            // 计算每个元素的梯度
            let grad_data: Vec<f32> = diff_view
                .iter()
                .map(|&a| {
                    let abs_a = a.abs();
                    if abs_a <= delta {
                        a // MSE 梯度: a
                    } else {
                        delta * a.signum() // MAE 梯度: δ * sign(a)
                    }
                })
                .collect();

            let grad = Tensor::new(&grad_data, diff.shape());

            // 应用 reduction
            let grad = match self.reduction {
                Reduction::Mean => &grad * (1.0 / self.numel_cache as f32),
                Reduction::Sum => grad,
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
