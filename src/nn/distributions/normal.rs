/*
 * @Author       : 老董
 * @Date         : 2026-02-12
 * @LastEditTime : 2026-02-13
 * @Description  : 正态分布（Normal Distribution）
 *
 * 支持重参数化采样（reparameterization trick）和计算图内梯度传播，
 * 适用于 SAC-Continuous 等需要连续动作采样的强化学习算法。
 *
 * # 设计原则
 * - **需要梯度 → Var，不需要梯度 → Tensor**
 * - 构造时缓存 `log_std = ln(σ)`，entropy() 和 log_prob() 共享该节点
 *
 * # 公式
 * - rsample: sample = μ + σ * ε, ε ~ N(0,1)
 * - log_prob: log p(x|μ,σ) = -0.5 * ((x-μ)/σ)² - ln(σ) - 0.5 * ln(2π)
 * - entropy: H = 0.5 + 0.5 * ln(2π) + ln(σ)
 *
 * # 参考
 * - PyTorch: `torch.distributions.Normal`
 * - Haarnoja et al. 2018（SAC 论文）
 */

use crate::nn::Var;
use crate::nn::VarActivationOps;
use crate::tensor::Tensor;

/// 正态分布
///
/// 构造时预计算 `log_std = ln(σ)`，entropy() 和 log_prob() 共享该节点。
///
/// # 构造
/// ```ignore
/// let mean = actor_mean.forward(&obs)?;
/// let std = actor_log_std.forward(&obs)?.exp();
/// let dist = Normal::new(mean, std);
/// ```
///
/// # 方法
/// - [`rsample()`](Normal::rsample) — 重参数化采样（Var，梯度可传播）
/// - [`log_prob(value)`](Normal::log_prob) — 对数概率密度（Var，逐元素）
/// - [`entropy()`](Normal::entropy) — 分布熵（Var，逐元素）
#[derive(Clone)]
pub struct Normal {
    mean: Var,
    std: Var,
    /// ln(σ) — 构造时缓存，entropy() 和 log_prob() 共享
    log_std: Var,
}

impl Normal {
    /// 创建正态分布
    ///
    /// 构造时立即创建 `ln(σ)` 节点并缓存。
    ///
    /// # 参数
    /// - `mean` — 均值（来自网络输出的 Var）
    /// - `std` — 标准差（来自网络输出的 Var，必须 > 0）
    pub fn new(mean: Var, std: Var) -> Self {
        let log_std = std.ln();
        Self {
            mean,
            std,
            log_std,
        }
    }

    /// 均值（Var 级）
    pub fn mean(&self) -> Var {
        self.mean.clone()
    }

    /// 标准差（Var 级）
    pub fn std(&self) -> Var {
        self.std.clone()
    }

    /// ln(σ)（Var 级，缓存的）
    pub fn log_std(&self) -> Var {
        self.log_std.clone()
    }

    /// 重参数化采样：sample = μ + σ * ε, ε ~ N(0,1)
    ///
    /// 梯度可通过 mean（∂/∂μ = 1）和 std（∂/∂σ = ε）传播。
    /// 这是连续策略梯度方法（如 SAC）的核心采样方式。
    ///
    /// # 返回
    /// Var，在计算图中，可反向传播
    pub fn rsample(&self) -> Var {
        let shape = self.mean.value_expected_shape();
        let eps = Tensor::normal(0.0, 1.0, &shape);
        &self.mean + &self.std * eps
    }

    /// 对数概率密度（逐元素，Var 级，可反向传播）
    ///
    /// log p(x|μ,σ) = -0.5 * ((x-μ)/σ)² - ln(σ) - 0.5 * ln(2π)
    ///
    /// 使用缓存的 log_std，不会创建冗余的 ln(σ) 节点。
    ///
    /// # 参数
    /// - `value` — 要计算概率的值（形状应与 mean/std 兼容）
    pub fn log_prob(&self, value: &Var) -> Var {
        let diff = value - &self.mean;
        let normalized = &diff / &self.std;
        let squared = &normalized * &normalized;

        // -0.5 * ((x - μ) / σ)²
        let neg_half = Tensor::new(&[-0.5], &[1, 1]);
        let term1 = &squared * neg_half;

        // -ln(σ)（复用缓存的 log_std）
        let term2 = -&self.log_std;

        // -0.5 * ln(2π)
        let const_val = -0.5 * (2.0_f32 * std::f32::consts::PI).ln();
        let term3 = Tensor::new(&[const_val], &[1, 1]);

        &term1 + &term2 + term3
    }

    /// 分布熵（逐元素，Var 级，可反向传播）
    ///
    /// H(X) = 0.5 + 0.5 * ln(2π) + ln(σ)
    ///
    /// 使用缓存的 log_std，不会创建冗余的 ln(σ) 节点。
    pub fn entropy(&self) -> Var {
        let const_val = 0.5 + 0.5 * (2.0_f32 * std::f32::consts::PI).ln();
        let const_tensor = Tensor::new(&[const_val], &[1, 1]);
        &self.log_std + const_tensor
    }
}
