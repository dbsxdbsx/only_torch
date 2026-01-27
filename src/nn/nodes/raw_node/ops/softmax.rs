/*
 * @Author       : 老董
 * @Date         : 2026-01-09
 * @Description  : Softmax 激活节点
 *                 实现沿最后一维的 softmax: softmax(x)_i = exp(x_i) / Σ exp(x_j)
 */

use crate::nn::GraphError;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::nodes::{NodeHandle, NodeId};
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;
use rayon::prelude::*;

/// Softmax 激活节点
///
/// 对输入张量沿最后一维计算 softmax，输出与输入形状相同。
/// 使用数值稳定的 log-sum-exp 技巧避免溢出。
///
/// ## 输入
/// - 父节点: [batch, `num_classes`] 或 [1, `num_classes`]
///
/// ## 输出
/// - 与输入形状相同，每行归一化为概率分布
#[derive(Clone)]
pub(crate) struct Softmax {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 固定形状（用于 `value_expected_shape`）
    fixed_shape: Vec<usize>,
    /// 动态形状（支持动态 batch）
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    supports_dynamic: bool,
    /// 缓存输出结果，用于反向传播
    output_cache: Option<Tensor>,
}

impl Softmax {
    pub(crate) fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        // 1. 验证父节点数量
        if parents.len() != 1 {
            return Err(GraphError::InvalidOperation(
                "Softmax 节点需要正好 1 个父节点".to_string(),
            ));
        }

        // 2. 获取输入形状
        let parent = &parents[0];
        let fixed_shape = parent.value_expected_shape().to_vec();
        if fixed_shape.len() != 2 {
            return Err(GraphError::InvalidOperation(format!(
                "Softmax 节点需要 2D 输入 [batch, num_classes]，但得到 {fixed_shape:?}"
            )));
        }

        // 3. 从父节点继承动态形状信息
        let dynamic_shape = parent.dynamic_expected_shape();
        let supports_dynamic = parent.supports_dynamic_batch();

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape,
            dynamic_shape,
            supports_dynamic,
            output_cache: None,
        })
    }

    /// 计算数值稳定的 softmax（支持 batch，Rayon 并行）
    /// 输入: [batch, `num_classes`]
    /// 输出: [batch, `num_classes`]
    fn stable_softmax_batch(logits: &Tensor) -> Tensor {
        let shape = logits.shape();
        let batch_size = shape[0];
        let num_classes = shape[1];

        // Rayon 并行处理每个 batch 样本
        let batch_results: Vec<Vec<f32>> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let mut sample_result = vec![0.0f32; num_classes];

                // 找到该样本的最大值
                let mut max_val = logits[[b, 0]];
                for c in 1..num_classes {
                    if logits[[b, c]] > max_val {
                        max_val = logits[[b, c]];
                    }
                }

                // 计算 exp(x - max) 和 sum
                let mut sum_exp = 0.0f32;
                for c in 0..num_classes {
                    let exp_val = (logits[[b, c]] - max_val).exp();
                    sample_result[c] = exp_val;
                    sum_exp += exp_val;
                }

                // 归一化
                for c in 0..num_classes {
                    sample_result[c] /= sum_exp;
                }

                sample_result
            })
            .collect();

        // 合并结果
        let all_data: Vec<f32> = batch_results.into_iter().flatten().collect();
        Tensor::new(&all_data, shape)
    }
}

impl TraitNode for Softmax {
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
        let input = parents[0].value().ok_or_else(|| {
            GraphError::ComputationError(format!(
                "{}的父{}没有值",
                self.display_node(),
                parents[0]
            ))
        })?;

        let output = Self::stable_softmax_batch(input);
        self.output_cache = Some(output.clone());
        self.value = Some(output);

        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// Softmax 反向传播的 VJP 计算
    ///
    /// 对于 y = softmax(x)，Jacobian 矩阵为：
    /// `dL/dx_i` = `Σ_j` (`dL/dy_j` * `dy_j/dx_i`)
    ///         = `Σ_j` (`dL/dy_j` * (`y_i` * (`δ_ij` - `y_j`)))
    ///         = `y_i` * (`dL/dy_i` - `Σ_j` (`dL/dy_j` * `y_j`))
    ///         = `y_i` * (`dL/dy_i` - <dL/dy, y>)
    ///
    /// 其中 <dL/dy, y> 是上游梯度与 softmax 输出的内积。
    fn calc_grad_to_parent(
        &self,
        _target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        _assistant_parent: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError> {
        let softmax_output = self.output_cache.as_ref().ok_or_else(|| {
            GraphError::ComputationError("Softmax 缓存为空，需先执行前向传播".to_string())
        })?;

        let shape = softmax_output.shape();
        let batch_size = shape[0];
        let num_classes = shape[1];

        // 并行计算每个样本的梯度
        let batch_grads: Vec<Vec<f32>> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                // 计算 <dL/dy, y> = Σ_j (dL/dy_j * y_j)
                let mut dot_product = 0.0f32;
                for c in 0..num_classes {
                    dot_product += upstream_grad[[b, c]] * softmax_output[[b, c]];
                }

                // dL/dx_i = y_i * (dL/dy_i - dot_product)
                let mut sample_grad = vec![0.0f32; num_classes];
                for c in 0..num_classes {
                    sample_grad[c] = softmax_output[[b, c]] * (upstream_grad[[b, c]] - dot_product);
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

    fn set_grad(&mut self, grad: Option<&Tensor>) -> Result<(), GraphError> {
        self.grad = grad.cloned();
        Ok(())
    }

    fn clear_value(&mut self) -> Result<(), GraphError> {
        self.value = None;
        self.output_cache = None;
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
