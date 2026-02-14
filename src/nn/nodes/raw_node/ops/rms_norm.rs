/*
 * @Author       : 老董
 * @Date         : 2026-02-15
 * @Description  : RMSNormOp（RMS 归一化运算）节点
 *                 LayerNorm 的简化版，去掉 mean centering。
 *
 * forward:
 *   rms = sqrt(mean(x^2) + eps)
 *   x_hat = x / rms
 *
 * backward:
 *   dx = (1/d) * (1/rms) * (d * upstream - x_hat * sum(upstream * x_hat))
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;

/// RMS 归一化运算节点（不含 gamma）
///
/// 输入形状任意，对最后 `normalized_dims` 个维度归一化。
/// 与 LayerNorm 的区别：不减均值，只除以 RMS。
#[derive(Clone)]
pub(crate) struct RMSNormOp {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    fixed_shape: Vec<usize>,
    dynamic_shape: DynamicShape,
    #[allow(dead_code)]
    supports_dynamic: bool,
    normalized_dims: usize,
    eps: f32,
    // 缓存
    x_hat_cache: Option<Tensor>,
    rms_cache: Option<Tensor>,
    d: usize,
}

impl RMSNormOp {
    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
        normalized_dims: usize,
        eps: f32,
    ) -> Result<Self, GraphError> {
        let ndim = parent_shape.len();
        if normalized_dims == 0 || normalized_dims > ndim {
            return Err(GraphError::InvalidOperation(format!(
                "RMSNormOp: normalized_dims={normalized_dims} 必须在 [1, {ndim}] 范围内"
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
            rms_cache: None,
            d,
        })
    }
}

impl TraitNode for RMSNormOp {
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
        let d: usize = shape[norm_start..].iter().product();
        self.d = d;

        let batch_size: usize = shape[..norm_start].iter().product();
        let flat = x.flatten_view();
        let mut x_hat_data = vec![0.0f32; x.size()];
        let mut rms_data = vec![0.0f32; batch_size];

        for b in 0..batch_size {
            let offset = b * d;

            // rms = sqrt(mean(x^2) + eps)
            let mut mean_sq = 0.0f32;
            for i in 0..d {
                let v = flat[offset + i];
                mean_sq += v * v;
            }
            mean_sq /= d as f32;
            let rms = (mean_sq + self.eps).sqrt();
            rms_data[b] = rms;

            // x_hat = x / rms
            let inv_rms = 1.0 / rms;
            for i in 0..d {
                x_hat_data[offset + i] = flat[offset + i] * inv_rms;
            }
        }

        let x_hat = Tensor::new(&x_hat_data, shape);
        let mut rms_shape: Vec<usize> = shape[..norm_start].to_vec();
        for _ in 0..self.normalized_dims {
            rms_shape.push(1);
        }
        let rms_tensor = Tensor::new(&rms_data, &rms_shape);

        self.x_hat_cache = Some(x_hat.clone());
        self.rms_cache = Some(rms_tensor);
        self.value = Some(x_hat);

        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// RMSNormOp 反向传播
    ///
    /// dx = (1/d) * (1/rms) * (d * upstream - x_hat * sum(upstream * x_hat))
    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        let rms_t = self.rms_cache.as_ref().ok_or_else(|| {
            GraphError::ComputationError("RMSNormOp rms 缓存为空".to_string())
        })?;
        let x_hat = self.x_hat_cache.as_ref().ok_or_else(|| {
            GraphError::ComputationError("RMSNormOp x_hat 缓存为空".to_string())
        })?;

        let shape = upstream_grad.shape();
        let ndim = shape.len();
        let norm_start = ndim - self.normalized_dims;
        let d = self.d;
        let batch_size: usize = shape[..norm_start].iter().product();

        let up_flat = upstream_grad.flatten_view();
        let xh_flat = x_hat.flatten_view();
        let rms_flat = rms_t.flatten_view();

        let mut dx_data = vec![0.0f32; upstream_grad.size()];

        for b in 0..batch_size {
            let offset = b * d;

            // sum(upstream * x_hat)
            let mut sum_up_xh = 0.0f32;
            for i in 0..d {
                let idx = offset + i;
                sum_up_xh += up_flat[idx] * xh_flat[idx];
            }

            let inv_rms = 1.0 / rms_flat[b];
            let inv_d = 1.0 / d as f32;

            // dx = (1/d) * (1/rms) * (d * upstream - x_hat * sum_up_xh)
            for i in 0..d {
                let idx = offset + i;
                dx_data[idx] = inv_d
                    * inv_rms
                    * (d as f32 * up_flat[idx] - xh_flat[idx] * sum_up_xh);
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
        self.rms_cache = None;
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }
}
