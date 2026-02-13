/*
 * @Author       : 老董
 * @Date         : 2026-02-12
 * @LastEditTime : 2026-02-13
 * @Description  : TanhNormal 分布（Squashed Gaussian）
 *
 * SAC-Continuous 的标准策略分布：先从正态分布采样 u，再用 tanh 压缩到 [-1, 1]。
 * log_prob 需要 Jacobian 修正以补偿 tanh 变换引入的概率密度变化。
 *
 * # 公式
 * - rsample: u ~ N(μ, σ), a = tanh(u)
 * - log_prob: log π(a|s) = log N(u|μ,σ) - log(1 - tanh²(u) + ε)
 *
 * # 参考
 * - Haarnoja et al. 2018 Appendix C（Enforcing Action Bounds）
 * - PyTorch: `TransformedDistribution(Normal, [TanhTransform()])`
 */

use super::Normal;
use crate::nn::Var;
use crate::nn::VarActivationOps;
use crate::tensor::Tensor;

/// TanhNormal 分布（Squashed Gaussian）
///
/// 将正态分布的采样通过 tanh 压缩到 [-1, 1] 范围，
/// 是 SAC-Continuous 算法的标准策略分布。
///
/// # 构造
/// ```ignore
/// let dist = TanhNormal::new(mean_var, std_var);
/// ```
///
/// # 典型 SAC 使用模式
/// ```ignore
/// // 方式一：分步调用
/// let (action, raw_action) = dist.rsample();
/// let log_prob = dist.log_prob(&raw_action);
///
/// // 方式二：一步完成（推荐）
/// let (action, log_prob) = dist.rsample_and_log_prob();
/// ```
#[derive(Clone)]
pub struct TanhNormal {
    base_dist: Normal,
}

/// TanhNormal 的 epsilon 值（数值稳定常量）
const TANH_NORMAL_EPS: f32 = 1e-6;

impl TanhNormal {
    /// 创建 TanhNormal 分布
    ///
    /// # 参数
    /// - `mean` — 正态分布均值（来自网络输出的 Var）
    /// - `std` — 正态分布标准差（来自网络输出的 Var，必须 > 0）
    pub fn new(mean: Var, std: Var) -> Self {
        Self {
            base_dist: Normal::new(mean, std),
        }
    }

    /// 均值（Var 级，来自内部 Normal 分布）
    pub fn mean(&self) -> Var {
        self.base_dist.mean()
    }

    /// 标准差（Var 级，来自内部 Normal 分布）
    pub fn std(&self) -> Var {
        self.base_dist.std()
    }

    /// 重参数化采样 + tanh 压缩
    ///
    /// 返回 `(squashed_action, raw_action)`：
    /// - `squashed_action` = tanh(u)，范围 [-1, 1]，用于环境交互
    /// - `raw_action` = u ~ N(μ, σ)，用于计算 log_prob
    ///
    /// # 注意
    /// 必须保留 `raw_action` 用于 `log_prob()` 计算。
    pub fn rsample(&self) -> (Var, Var) {
        let raw = self.base_dist.rsample();
        let squashed = raw.tanh();
        (squashed, raw)
    }

    /// 对数概率密度（带 tanh Jacobian 修正，逐元素，Var 级）
    ///
    /// log π(a|s) = log N(u|μ,σ) - log(1 - tanh²(u) + ε)
    ///
    /// # 参数
    /// - `raw_action` — tanh 前的原始采样值 u（来自 `rsample()` 的第二个返回值）
    pub fn log_prob(&self, raw_action: &Var) -> Var {
        let base_lp = self.base_dist.log_prob(raw_action);

        // Jacobian 修正：-log(1 - tanh²(u) + ε)
        let tanh_u = raw_action.tanh();
        let tanh_sq = &tanh_u * &tanh_u;
        let one = Tensor::new(&[1.0], &[1, 1]);
        let eps = Tensor::new(&[TANH_NORMAL_EPS], &[1, 1]);
        let inner = one - &tanh_sq + eps;
        let correction = inner.ln();

        &base_lp - &correction
    }

    /// 采样 + 计算 log_prob（一步完成，推荐用于 SAC）
    ///
    /// 返回 `(squashed_action, log_prob_per_element)`。
    ///
    /// # 典型用法
    /// ```ignore
    /// let (action, lp) = dist.rsample_and_log_prob();
    /// let log_prob = lp.sum_axis(1);  // [batch, 1]
    /// let actor_loss = (&log_prob * alpha - &q_value).mean();
    /// ```
    pub fn rsample_and_log_prob(&self) -> (Var, Var) {
        let (squashed, raw) = self.rsample();
        let log_prob = self.log_prob(&raw);
        (squashed, log_prob)
    }
}
