//! PUCT (Predictor Upper Confidence bound for Trees) 策略实现

use rand::RngCore;

use super::min_max::MinMaxStats;
use super::traits::SearchPolicy;
use super::types::{ChildStat, MctsConfig};

/// PUCT 搜索策略
///
/// 实现 AlphaZero 风格的 UCB 选择 + Dirichlet 噪声 + 温度采样。
#[derive(Debug, Clone, Default)]
pub struct PuctPolicy;

impl PuctPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl SearchPolicy for PuctPolicy {
    /// 向根子节点注入 Dirichlet 噪声以鼓励探索
    ///
    /// 使用 Gamma(alpha, 1.0) 独立采样后归一化来合成 Dirichlet，
    /// 不引入 rand_distr 依赖。
    ///
    /// Sampled MuZero 在展开前已对 β/π 加噪，此处跳过（见 [`MctsConfig::sampled_k`]）。
    fn prepare_root(&self, children: &mut [ChildStat], cfg: &MctsConfig, rng: &mut dyn RngCore) {
        if cfg.sampled_k.is_some() {
            return;
        }
        if children.is_empty() {
            return;
        }
        let mut priors: Vec<f32> = children.iter().map(|c| c.prior).collect();
        mix_dirichlet_prior(&mut priors, cfg, rng);
        for (child, p) in children.iter_mut().zip(priors) {
            child.prior = p;
        }
    }

    /// PUCT 公式选择子节点（父节点视角）
    ///
    /// `Q(a) = r(a) + γ · perspective · V(child(a))`
    /// - perspective = +1（同一玩家）或 -1（对手，negamax）
    fn select_child(
        &self,
        parent_visit: u32,
        parent_to_play: u8,
        children: &[ChildStat],
        stats: &MinMaxStats,
        cfg: &MctsConfig,
    ) -> usize {
        let pb_c =
            ((1.0 + parent_visit as f32 + cfg.pb_c_base) / cfg.pb_c_base).ln() + cfg.pb_c_init;
        let sqrt_parent = (parent_visit as f32).sqrt();

        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (i, child) in children.iter().enumerate() {
            let v = if child.visit_count == 0 {
                0.0
            } else {
                child.value_sum / child.visit_count as f32
            };
            // 视角翻转：子节点 value_sum 是子方视角，父方选择需翻转
            let perspective = if child.to_play == parent_to_play {
                1.0
            } else {
                -1.0
            };
            let q = child.reward + child.discount * perspective * v;
            let normalized_q = stats.normalize(q);
            let exploration = pb_c * child.prior * sqrt_parent / (1.0 + child.visit_count as f32);
            let score = normalized_q + exploration;

            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        best_idx
    }

    /// 按温度随机采样 visit count 推荐动作
    ///
    /// 训练时 temperature=1.0 → 按 visit count 比例采样（保证探索多样性）
    /// 评测时 temperature→0 → 贪心选最大 visit count
    fn recommend(&self, children: &[ChildStat], cfg: &MctsConfig, rng: &mut dyn RngCore) -> usize {
        if children.is_empty() {
            return 0;
        }

        // 温度极低 → 贪心
        if cfg.temperature < 1e-6 {
            return children
                .iter()
                .enumerate()
                .max_by_key(|(_, c)| c.visit_count)
                .map(|(i, _)| i)
                .unwrap_or(0);
        }

        // 全 0 visit → uniform fallback
        let total_visits: u32 = children.iter().map(|c| c.visit_count).sum();
        if total_visits == 0 {
            return (rng.next_u32() as usize) % children.len();
        }

        // log-space 防溢出：未访问动作 weight=0
        let inv_temp = 1.0 / cfg.temperature;
        let log_counts: Vec<f32> = children
            .iter()
            .map(|c| {
                if c.visit_count > 0 {
                    (c.visit_count as f32).ln()
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        let max_log = log_counts.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = log_counts
            .iter()
            .map(|&lc| {
                if lc == f32::NEG_INFINITY {
                    0.0
                } else {
                    ((lc - max_log) * inv_temp).exp()
                }
            })
            .collect();
        let sum: f32 = weights.iter().sum();
        if sum <= 0.0 {
            return 0;
        }

        let threshold = (rng.next_u32() as f32 / u32::MAX as f32) * sum;
        let mut cumulative = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            cumulative += w;
            if cumulative >= threshold {
                return i;
            }
        }
        weights.len() - 1
    }

    /// 将 visit count 按温度归一化为概率分布（学习目标）
    ///
    /// `π(a) ∝ N(a)^(1/τ)`；τ→0 时退化为 one-hot，τ→∞ 时趋近均匀。
    fn make_targets(&self, children: &[ChildStat], cfg: &MctsConfig) -> Vec<f32> {
        let n = children.len();
        if n == 0 {
            return Vec::new();
        }
        let total_visits: u32 = children.iter().map(|c| c.visit_count).sum();
        if total_visits == 0 {
            return vec![1.0 / n as f32; n];
        }
        // 温度极低 → one-hot
        if cfg.temperature < 1e-6 {
            let max_visits = children.iter().map(|c| c.visit_count).max().unwrap_or(0);
            let mut targets = vec![0.0; n];
            for (i, c) in children.iter().enumerate() {
                if c.visit_count == max_visits {
                    targets[i] = 1.0;
                }
            }
            let s: f32 = targets.iter().sum();
            if s > 0.0 {
                for t in &mut targets {
                    *t /= s;
                }
            }
            return targets;
        }
        let inv_temp = 1.0 / cfg.temperature;
        let log_counts: Vec<f32> = children
            .iter()
            .map(|c| {
                if c.visit_count > 0 {
                    (c.visit_count as f32).ln()
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        let max_log = log_counts.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = log_counts
            .iter()
            .map(|&lc| {
                if lc == f32::NEG_INFINITY {
                    0.0
                } else {
                    ((lc - max_log) * inv_temp).exp()
                }
            })
            .collect();
        let sum: f32 = weights.iter().sum();
        weights.iter().map(|&w| w / sum).collect()
    }
}

/// 将 Dirichlet 噪声混入 prior 向量（根探索；Sampled MuZero 根展开前亦调用）。
pub(crate) fn mix_dirichlet_prior(prior: &mut [f32], cfg: &MctsConfig, rng: &mut dyn RngCore) {
    if prior.is_empty() {
        return;
    }
    let noise = sample_dirichlet(cfg.root_dirichlet_alpha, prior.len(), rng);
    let frac = cfg.root_exploration_fraction;
    for (p, &n) in prior.iter_mut().zip(noise.iter()) {
        *p = *p * (1.0 - frac) + n * frac;
    }
}

/// 用 Gamma(alpha, 1.0) 合成 Dirichlet 分布采样
///
/// 基于 Marsaglia & Tsang 方法（alpha >= 1）和 alpha < 1 的变换。
fn sample_dirichlet(alpha: f32, n: usize, rng: &mut dyn RngCore) -> Vec<f32> {
    let mut samples: Vec<f32> = (0..n).map(|_| sample_gamma(alpha, rng)).collect();
    let sum: f32 = samples.iter().sum();
    if sum > 0.0 {
        for s in &mut samples {
            *s /= sum;
        }
    } else {
        // fallback: 均匀分布
        let uniform = 1.0 / n as f32;
        samples.fill(uniform);
    }
    samples
}

/// Gamma(alpha, 1.0) 采样 —— Marsaglia & Tsang (2000)
///
/// 对 alpha < 1，使用变换：Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
fn sample_gamma(alpha: f32, rng: &mut dyn RngCore) -> f32 {
    if alpha < 1.0 {
        // Gamma(alpha, 1) = Gamma(alpha+1, 1) * U^(1/alpha)
        let g = sample_gamma_ge1(alpha + 1.0, rng);
        let u = rand_uniform(rng);
        g * u.powf(1.0 / alpha)
    } else {
        sample_gamma_ge1(alpha, rng)
    }
}

/// Marsaglia & Tsang 方法（要求 alpha >= 1）
fn sample_gamma_ge1(alpha: f32, rng: &mut dyn RngCore) -> f32 {
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let x = rand_normal(rng);
        let v_base = 1.0 + c * x;
        if v_base <= 0.0 {
            continue;
        }
        let v = v_base * v_base * v_base;
        let u = rand_uniform(rng);
        let x2 = x * x;

        // 快速接受
        if u < 1.0 - 0.0331 * x2 * x2 {
            return d * v;
        }
        // 慢速接受
        if u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// [0, 1) 均匀分布
fn rand_uniform(rng: &mut dyn RngCore) -> f32 {
    let bits = rng.next_u32();
    (bits >> 8) as f32 / (1u32 << 24) as f32
}

/// 标准正态分布（Box-Muller）
fn rand_normal(rng: &mut dyn RngCore) -> f32 {
    loop {
        let u1 = rand_uniform(rng);
        let u2 = rand_uniform(rng);
        if u1 > 0.0 {
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            return r * theta.cos();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::mcts::types::ActionPayload;

    #[test]
    fn test_dirichlet_sums_to_one() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(42);
        let d = sample_dirichlet(0.3, 5, &mut rng);
        assert_eq!(d.len(), 5);
        let sum: f32 = d.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Dirichlet 样本之和应为 1，实际: {sum}"
        );
        for &x in &d {
            assert!(x >= 0.0, "Dirichlet 分量不应为负");
        }
    }

    #[test]
    fn test_puct_select_prefers_high_prior_unvisited() {
        let policy = PuctPolicy::new();
        let cfg = MctsConfig::default();
        let children = vec![
            ChildStat {
                action: ActionPayload::Discrete(0),
                visit_count: 10,
                value_sum: 5.0,
                prior: 0.1,
                reward: 0.0,
                to_play: 0,
                discount: 1.0,
            },
            ChildStat {
                action: ActionPayload::Discrete(1),
                visit_count: 0,
                value_sum: 0.0,
                prior: 0.9,
                reward: 0.0,
                to_play: 0,
                discount: 1.0,
            },
        ];
        let idx = policy.select_child(10, 0, &children, &MinMaxStats::new(), &cfg);
        assert_eq!(idx, 1, "高 prior 未访问节点应被优先选择");
    }

    #[test]
    fn test_make_targets_normalization() {
        let policy = PuctPolicy::new();
        let cfg = MctsConfig::default();
        let children = vec![
            ChildStat {
                action: ActionPayload::Discrete(0),
                visit_count: 3,
                value_sum: 1.0,
                prior: 0.5,
                reward: 0.0,
                to_play: 0,
                discount: 1.0,
            },
            ChildStat {
                action: ActionPayload::Discrete(1),
                visit_count: 7,
                value_sum: 2.0,
                prior: 0.5,
                reward: 0.0,
                to_play: 0,
                discount: 1.0,
            },
        ];
        let targets = policy.make_targets(&children, &cfg);
        assert_eq!(targets.len(), 2);
        assert!((targets[0] - 0.3).abs() < 1e-5);
        assert!((targets[1] - 0.7).abs() < 1e-5);
    }
}
