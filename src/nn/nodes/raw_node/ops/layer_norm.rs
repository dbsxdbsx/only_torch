/*
 * @Author       : 老董
 * @Date         : 2026-02-15
 * @Description  : LayerNormOp（层归一化运算）节点
 *                 实现层归一化的核心计算（不含 gamma/beta 可学习参数）
 *
 * forward:
 *   对输入的最后 `normalized_dims` 个维度进行归一化：
 *   x_hat = (x - mean) / sqrt(var + eps)
 *   无 train/eval 模式区别。
 *
 * backward:
 *   dx = (1/d) * (1/std) * (d * upstream - sum(upstream) - x_hat * sum(upstream * x_hat))
 *   其中 d = 归一化维度元素总数，sum 沿归一化维度归约
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// 层归一化运算节点（不含 gamma/beta）
///
/// 输入形状任意，对最后 `normalized_dims` 个维度归一化。
/// 常见用法：输入 [N, T, D]，normalized_dims=1 → 沿 D 归一化。
///
/// gamma/beta 由外层 Layer 通过 Mul/Add 节点处理。
#[derive(Clone)]
pub(crate) struct LayerNormOp {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 固定形状
    fixed_shape: Vec<usize>,
    /// 动态形状
    dynamic_shape: DynamicShape,
    /// 是否支持动态 batch
    #[allow(dead_code)]
    supports_dynamic: bool,
    /// 归一化的维度数（从最后一维开始）
    normalized_dims: usize,
    /// 数值稳定性常数
    eps: f32,
    // ---- 缓存（反向传播用）----
    /// 归一化后的值 x_hat
    x_hat_cache: Option<Tensor>,
    /// 标准差（可广播形状）
    std_cache: Option<Tensor>,
    /// 归一化维度的元素总数 d
    d: usize,
}

impl LayerNormOp {
    pub(crate) const fn normalized_dims(&self) -> usize {
        self.normalized_dims
    }
    pub(crate) const fn eps(&self) -> f32 {
        self.eps
    }

    /// 创建 LayerNormOp 节点
    ///
    /// # 参数
    /// - `parent_shape`: 输入形状
    /// - `parent_dynamic_shape`: 动态形状
    /// - `normalized_dims`: 归一化维度数（从最后一维开始）
    /// - `eps`: 数值稳定性常数（默认 1e-5）
    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
        normalized_dims: usize,
        eps: f32,
    ) -> Result<Self, GraphError> {
        let ndim = parent_shape.len();
        if normalized_dims == 0 || normalized_dims > ndim {
            return Err(GraphError::InvalidOperation(format!(
                "LayerNormOp: normalized_dims={normalized_dims} 必须在 [1, {ndim}] 范围内"
            )));
        }

        let d: usize = parent_shape[ndim - normalized_dims..].iter().product();
        let supports_dynamic = parent_dynamic_shape.dims().first() == Some(&None);

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape: parent_shape.to_vec(),
            dynamic_shape: parent_dynamic_shape.clone(),
            supports_dynamic,
            normalized_dims,
            eps,
            x_hat_cache: None,
            std_cache: None,
            d,
        })
    }
}

impl TraitNode for LayerNormOp {
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
        let x = parent_values[0];
        let shape = x.shape();
        let ndim = shape.len();
        let norm_start = ndim - self.normalized_dims;

        // d = 归一化维度的元素总数
        let d: usize = shape[norm_start..].iter().product();
        self.d = d;

        // batch_size = 前面维度的元素总数
        let batch_size: usize = shape[..norm_start].iter().product();

        let flat = x.flatten_view();
        let mut x_hat_data = vec![0.0f32; x.size()];
        let mut std_data = vec![0.0f32; batch_size];

        for b in 0..batch_size {
            let offset = b * d;

            // 计算均值
            let mut mean = 0.0f32;
            for i in 0..d {
                mean += flat[offset + i];
            }
            mean /= d as f32;

            // 计算方差
            let mut var = 0.0f32;
            for i in 0..d {
                let diff = flat[offset + i] - mean;
                var += diff * diff;
            }
            var /= d as f32;

            // std = sqrt(var + eps)
            let std = (var + self.eps).sqrt();
            std_data[b] = std;

            // x_hat = (x - mean) / std
            let inv_std = 1.0 / std;
            for i in 0..d {
                x_hat_data[offset + i] = (flat[offset + i] - mean) * inv_std;
            }
        }

        let x_hat = Tensor::new(&x_hat_data, shape);

        // 构建 std 的可广播形状
        let mut std_shape: Vec<usize> = shape[..norm_start].to_vec();
        for _ in 0..self.normalized_dims {
            std_shape.push(1);
        }
        let std_tensor = Tensor::new(&std_data, &std_shape);

        self.x_hat_cache = Some(x_hat.clone());
        self.std_cache = Some(std_tensor);
        self.value = Some(x_hat);

        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// LayerNormOp 反向传播
    ///
    /// dx = (1/d) * (1/std) * (d * upstream - sum(upstream) - x_hat * sum(upstream * x_hat))
    /// 其中 sum 沿归一化维度归约
    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        let std_t = self.std_cache.as_ref().ok_or_else(|| {
            GraphError::ComputationError("LayerNormOp std 缓存为空，需先执行前向传播".to_string())
        })?;
        let x_hat = self.x_hat_cache.as_ref().ok_or_else(|| {
            GraphError::ComputationError("LayerNormOp x_hat 缓存为空".to_string())
        })?;

        let shape = upstream_grad.shape();
        let ndim = shape.len();
        let norm_start = ndim - self.normalized_dims;
        let d = self.d;
        let batch_size: usize = shape[..norm_start].iter().product();

        let up_flat = upstream_grad.flatten_view();
        let xh_flat = x_hat.flatten_view();
        let std_flat = std_t.flatten_view();

        let mut dx_data = vec![0.0f32; upstream_grad.size()];

        for b in 0..batch_size {
            let offset = b * d;

            // sum(upstream) 和 sum(upstream * x_hat)
            let mut sum_up = 0.0f32;
            let mut sum_up_xh = 0.0f32;
            for i in 0..d {
                let idx = offset + i;
                sum_up += up_flat[idx];
                sum_up_xh += up_flat[idx] * xh_flat[idx];
            }

            let inv_std = 1.0 / std_flat[b];
            let inv_d = 1.0 / d as f32;

            // dx = (1/d) * (1/std) * (d * upstream - sum_up - x_hat * sum_up_xh)
            for i in 0..d {
                let idx = offset + i;
                dx_data[idx] =
                    inv_d * inv_std * (d as f32 * up_flat[idx] - sum_up - xh_flat[idx] * sum_up_xh);
            }
        }

        Ok(GradResult::Computed(Tensor::new(&dx_data, shape)))
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
        self.x_hat_cache = None;
        self.std_cache = None;
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
