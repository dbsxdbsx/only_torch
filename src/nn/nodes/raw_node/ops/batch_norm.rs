/*
 * @Author       : 老董
 * @Date         : 2026-02-15
 * @Description  : BatchNormOp（批归一化运算）节点
 *                 实现批归一化的核心计算（不含 gamma/beta 可学习参数）
 *
 * forward (训练模式):
 *   1. 计算 batch 均值/方差（沿 channel 以外的所有维度）
 *   2. 归一化：x_hat = (x - mean) / sqrt(var + eps)
 *   3. 更新 running_mean/running_var (EMA)
 *
 * forward (评估模式):
 *   x_hat = (x - running_mean) / sqrt(running_var + eps)
 *
 * backward (训练模式):
 *   dx = (1/N) * (1/std) * (N * upstream - sum(upstream) - x_hat * sum(upstream * x_hat))
 *   其中 sum 沿 channel 以外的所有维度归约
 */

use crate::nn::GraphError;
use crate::nn::nodes::NodeId;
use crate::nn::nodes::raw_node::GradResult;
use crate::nn::nodes::raw_node::TraitNode;
use crate::nn::shape::DynamicShape;
use crate::tensor::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

/// 批归一化运算节点（不含 gamma/beta）
///
/// 输入：[N, C, ...] 其中 ... 可为空（1D）或 H,W（2D）
/// 输出：与输入形状相同
///
/// 沿 channel 以外的所有维度（N 和空间维度）进行归一化。
/// gamma/beta 由外层 Layer 通过 Mul/Add 节点处理。
#[derive(Clone)]
pub(crate) struct BatchNormOp {
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
    /// 通道数（dim 1 的大小）
    num_features: usize,
    /// 数值稳定性常数
    eps: f32,
    /// EMA 动量（用于更新 running stats）
    momentum: f32,
    /// 训练/评估模式
    is_training: bool,
    /// 运行均值 [C]（共享引用，跨 forward 调用持久化）
    running_mean: Rc<RefCell<Tensor>>,
    /// 运行方差 [C]（共享引用，跨 forward 调用持久化）
    running_var: Rc<RefCell<Tensor>>,
    // ---- 缓存（反向传播用）----
    /// 归一化后的值 x_hat
    x_hat_cache: Option<Tensor>,
    /// 标准差 sqrt(var + eps)，形状为 broadcastable
    std_cache: Option<Tensor>,
    /// 每个 channel 的归约元素数 N
    n_reduce: usize,
}

impl BatchNormOp {
    pub(crate) const fn eps(&self) -> f32 { self.eps }
    pub(crate) const fn momentum(&self) -> f32 { self.momentum }
    pub(crate) const fn num_features(&self) -> usize { self.num_features }
    pub(crate) fn running_mean(&self) -> &std::rc::Rc<std::cell::RefCell<crate::tensor::Tensor>> { &self.running_mean }
    pub(crate) fn running_var(&self) -> &std::rc::Rc<std::cell::RefCell<crate::tensor::Tensor>> { &self.running_var }

    /// 创建 BatchNormOp 节点
    ///
    /// # 参数
    /// - `parent_shape`: 输入形状 [N, C, ...]
    /// - `parent_dynamic_shape`: 动态形状
    /// - `eps`: 数值稳定性常数（默认 1e-5）
    /// - `momentum`: EMA 动量（默认 0.1）
    pub(in crate::nn) fn new(
        parent_shape: &[usize],
        parent_dynamic_shape: &DynamicShape,
        eps: f32,
        momentum: f32,
        running_mean: Rc<RefCell<Tensor>>,
        running_var: Rc<RefCell<Tensor>>,
    ) -> Result<Self, GraphError> {
        if parent_shape.len() < 2 {
            return Err(GraphError::InvalidOperation(
                "BatchNormOp: 输入至少需要 2 维 [N, C, ...]".to_string(),
            ));
        }

        let num_features = parent_shape[1];
        let supports_dynamic = parent_dynamic_shape.dims().first() == Some(&None);

        Ok(Self {
            id: None,
            name: None,
            value: None,
            grad: None,
            fixed_shape: parent_shape.to_vec(),
            dynamic_shape: parent_dynamic_shape.clone(),
            supports_dynamic,
            num_features,
            eps,
            momentum,
            is_training: true,
            running_mean,
            running_var,
            x_hat_cache: None,
            std_cache: None,
            n_reduce: 0,
        })
    }

    /// 计算 per-channel 均值
    /// 输入 [N, C, ...] → 输出形状可广播回输入（如 [1, C, 1, 1]）
    fn channel_mean(x: &Tensor) -> Tensor {
        let shape = x.shape();
        let ndim = shape.len();
        let c = shape[1];

        // 将 [N, C, ...] 转为 [N, C, -1]，沿 dim 0 和 dim 2 求均值
        let spatial_size: usize = shape[2..].iter().product();
        let n = shape[0];
        let total = n * spatial_size;

        let flat = x.flatten_view();
        let mut means = vec![0.0f32; c];
        for sample in 0..n {
            for ch in 0..c {
                for s in 0..spatial_size {
                    let idx = sample * c * spatial_size + ch * spatial_size + s;
                    means[ch] += flat[idx];
                }
            }
        }
        for m in &mut means {
            *m /= total as f32;
        }

        // 返回可广播的形状
        let mut out_shape = vec![1usize; ndim];
        out_shape[1] = c;
        Tensor::new(&means, &out_shape)
    }

    /// 计算 per-channel 方差（总体方差，ddof=0）
    fn channel_var(x: &Tensor, mean: &Tensor) -> Tensor {
        let shape = x.shape();
        let ndim = shape.len();
        let c = shape[1];
        let spatial_size: usize = shape[2..].iter().product();
        let n = shape[0];
        let total = n * spatial_size;

        let flat = x.flatten_view();
        let mean_flat = mean.flatten_view();
        let mut vars = vec![0.0f32; c];
        for sample in 0..n {
            for ch in 0..c {
                let m = mean_flat[ch];
                for s in 0..spatial_size {
                    let idx = sample * c * spatial_size + ch * spatial_size + s;
                    let diff = flat[idx] - m;
                    vars[ch] += diff * diff;
                }
            }
        }
        for v in &mut vars {
            *v /= total as f32;
        }

        let mut out_shape = vec![1usize; ndim];
        out_shape[1] = c;
        Tensor::new(&vars, &out_shape)
    }
}

impl TraitNode for BatchNormOp {
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
        let spatial_size: usize = shape[2..].iter().product();
        self.n_reduce = shape[0] * spatial_size.max(1);

        if self.is_training {
            // 训练模式：用 batch 统计量
            let mean = Self::channel_mean(x);
            let var = Self::channel_var(x, &mean);

            // std = sqrt(var + eps)
            let std = (&var + self.eps).powf(0.5);

            // x_hat = (x - mean) / std
            let x_hat = &(x - &mean) / &std;

            // 更新 running stats (EMA)
            // running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            let mean_1d = Tensor::new(mean.data_as_slice(), &[self.num_features]);
            let var_1d = Tensor::new(var.data_as_slice(), &[self.num_features]);

            // 用无偏方差更新 running_var（Bessel 修正：N/(N-1)）
            let n = self.n_reduce as f32;
            let unbiased_var = &var_1d * (n / (n - 1.0));

            {
                let mut rm = self.running_mean.borrow_mut();
                *rm = &(&*rm * (1.0 - self.momentum)) + &(&mean_1d * self.momentum);
            }
            {
                let mut rv = self.running_var.borrow_mut();
                *rv = &(&*rv * (1.0 - self.momentum)) + &(&unbiased_var * self.momentum);
            }

            self.x_hat_cache = Some(x_hat.clone());
            self.std_cache = Some(std);
            self.value = Some(x_hat);
        } else {
            // 评估模式：用 running stats
            let ndim = shape.len();
            let mut bcast_shape = vec![1usize; ndim];
            bcast_shape[1] = self.num_features;

            let mean = self.running_mean.borrow().reshape(&bcast_shape);
            let var = self.running_var.borrow().reshape(&bcast_shape);
            let std = (&var + self.eps).powf(0.5);

            let x_hat = &(x - &mean) / &std;
            self.std_cache = Some(std);
            self.x_hat_cache = None; // 评估模式不需要缓存 x_hat（不做复杂 backward）
            self.value = Some(x_hat);
        }

        Ok(())
    }

    fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    /// BatchNormOp 反向传播
    ///
    /// 训练模式:
    ///   dx = (1/N) * (1/std) * (N * upstream - sum(upstream) - x_hat * sum(upstream * x_hat))
    ///   其中 sum 沿 channel 以外的所有维度归约
    ///
    /// 评估模式:
    ///   dx = upstream / std
    fn calc_grad_to_parent(
        &self,
        _target_parent_index: usize,
        _parent_values: &[&Tensor],
        upstream_grad: &Tensor,
    ) -> Result<GradResult, GraphError> {
        let std = self.std_cache.as_ref().ok_or_else(|| {
            GraphError::ComputationError(
                "BatchNormOp std 缓存为空，需先执行前向传播".to_string(),
            )
        })?;

        if self.is_training {
            let x_hat = self.x_hat_cache.as_ref().ok_or_else(|| {
                GraphError::ComputationError(
                    "BatchNormOp x_hat 缓存为空".to_string(),
                )
            })?;

            let n = self.n_reduce as f32;
            let shape = upstream_grad.shape();
            let ndim = shape.len();
            let c = self.num_features;
            let spatial_size: usize = shape[2..].iter().product();
            let batch_size = shape[0];

            // 计算 per-channel sum(upstream) 和 sum(upstream * x_hat)
            let up_flat = upstream_grad.flatten_view();
            let xh_flat = x_hat.flatten_view();

            let mut sum_up = vec![0.0f32; c];
            let mut sum_up_xh = vec![0.0f32; c];

            for sample in 0..batch_size {
                for ch in 0..c {
                    for s in 0..spatial_size.max(1) {
                        let idx = sample * c * spatial_size.max(1) + ch * spatial_size.max(1) + s;
                        sum_up[ch] += up_flat[idx];
                        sum_up_xh[ch] += up_flat[idx] * xh_flat[idx];
                    }
                }
            }

            // 构建可广播的 sum 张量
            let mut bcast_shape = vec![1usize; ndim];
            bcast_shape[1] = c;
            let sum_up_t = Tensor::new(&sum_up, &bcast_shape);
            let sum_up_xh_t = Tensor::new(&sum_up_xh, &bcast_shape);

            // dx = (1/N) * (1/std) * (N * upstream - sum_up - x_hat * sum_up_xh)
            let dx = &(&(upstream_grad * n) - &sum_up_t - &(x_hat * &sum_up_xh_t))
                / &(std * n);

            Ok(GradResult::Computed(dx))
        } else {
            // 评估模式：简单除以 std
            Ok(GradResult::Computed(upstream_grad / std))
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
        self.x_hat_cache = None;
        self.std_cache = None;
        Ok(())
    }

    fn set_value_unchecked(&mut self, value: Option<&Tensor>) {
        self.value = value.cloned();
    }

    fn set_training_mode(&mut self, is_training: bool) {
        self.is_training = is_training;
    }
}
