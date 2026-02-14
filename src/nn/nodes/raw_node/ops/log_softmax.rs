/*
 * @Author       : 老董
 * @Date         : 2026-01-31
 * @Description  : LogSoftmax 节点
 *                 实现数值稳定的 log(softmax(x))
 *
 * 数学公式：
 * log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
 *
 * 比直接计算 softmax().ln() 更数值稳定。
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;
use rayon::prelude::*;

/// LogSoftmax 节点
///
/// 对输入张量沿最后一维计算 log_softmax，输出与输入形状相同。
/// 使用数值稳定的 log-sum-exp 技巧避免溢出和下溢。
///
/// ## 输入
/// - 父节点: [batch, `num_classes`] 或 [1, `num_classes`]
///
/// ## 输出
/// - 与输入形状相同，每个元素是对应位置的 log 概率
///
/// ## 梯度计算
/// 对于 y = log_softmax(x)，有：
/// ∂y_i/∂x_j = δ_ij - softmax(x)_j
///
/// VJP: grad_to_parent = upstream_grad - sum(upstream_grad) * softmax(x)
#[derive(Clone)]
pub(crate) struct LogSoftmax {
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
    /// 缓存 softmax 输出，用于反向传播
    softmax_cache: Option<Tensor>,
}

impl LogSoftmax {
    /// 从父节点形状信息创建 LogSoftmax 节点（核心实现）
    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
    ) -> Result<Self, GraphError> {
        // 验证形状：LogSoftmax 需要 2D 输入
        if parent_shape.len() != 2 {
            return Err(GraphError::InvalidOperation(format!(
                "LogSoftmax 节点需要 2D 输入 [batch, num_classes]，但得到 {parent_shape:?}"
            )));
        }

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape: parent_shape.to_vec(),
            dynamic_shape: parent_dynamic_shape.clone(),
            supports_dynamic: parent_dynamic_shape.has_dynamic_dims(),
            softmax_cache: None,
        })
    }
}

impl TraitNode for LogSoftmax {
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
        // 计算 log_softmax（沿最后一维）
        let output = parent_values[0].log_softmax_last_dim();
        // 缓存 softmax 输出用于反向传播
        self.softmax_cache = Some(output.exp());
        self.value = Some(output);
        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// LogSoftmax 反向传播的 VJP 计算
    ///
    /// 对于 y = log_softmax(x)，Jacobian 矩阵为：
    /// ∂y_i/∂x_j = δ_ij - softmax(x)_j
    ///
    /// 所以 VJP 为：
    /// `dL/dx_i` = `Σ_j` (`dL/dy_j` * `∂y_j/∂x_i`)
    ///         = `dL/dy_i` - `softmax(x)_i` * `Σ_j` `dL/dy_j`
    ///         = `upstream_grad_i` - `softmax_i` * `sum(upstream_grad)`
    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<Tensor, GraphError> {
        let softmax_output = self.softmax_cache.as_ref().ok_or_else(|| {
            GraphError::ComputationError("LogSoftmax 缓存为空，需先执行前向传播".to_string())
        })?;

        let shape = softmax_output.shape();
        let batch_size = shape[0];
        let num_classes = shape[1];

        // 并行计算每个样本的梯度
        let batch_grads: Vec<Vec<f32>> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                // 计算 sum(upstream_grad) 对当前行
                let mut sum_upstream = 0.0f32;
                for c in 0..num_classes {
                    sum_upstream += upstream_grad[[b, c]];
                }

                // dL/dx_i = upstream_grad_i - softmax_i * sum(upstream_grad)
                let mut sample_grad = vec![0.0f32; num_classes];
                for c in 0..num_classes {
                    sample_grad[c] = upstream_grad[[b, c]] - softmax_output[[b, c]] * sum_upstream;
                }

                sample_grad
            })
            .collect();

        // 合并结果
        let all_grads: Vec<f32> = batch_grads.into_iter().flatten().collect();
        Ok(Tensor::new(&all_grads, shape))
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
        self.softmax_cache = None;
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
