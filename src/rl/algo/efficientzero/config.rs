//! EfficientZero V2 配置：**组合而非扁平**（reviewer P1）。
//!
//! 把 `num_simulations` / `gumbel_sims` / `eval_sims` 等易互相覆盖的魔法常量按职责分组，
//! 避免一个扁平大结构里语义打架。各子配置独立 `Default`，组合成 [`EfficientZeroConfig`]。
//!
//! `Default` 给的是 **CartPole-v1 级（CPU only）** 的 EZ 友好起点；换环境（Pendulum / 五子棋 /
//! Atari）时覆盖相应子配置字段，而不是在训练循环里写死。

use crate::rl::algo::muzero::MuZeroConfig;
use crate::rl::mcts::MctsConfig;

/// Gumbel 搜索配置（Phase 2a 起用；离散 CartPole 走 PUCT 时忽略）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GumbelConfig {
    /// 根候选采样数 m（Gumbel-Top-k 的 k）。连续/大动作空间从策略采样 m 个候选。
    pub num_sampled_actions: usize,
    /// sequential halving 的总模拟预算（通常对齐 `MctsConfig::num_simulations`）。
    pub num_simulations: u32,
    /// Gumbel σ 变换的 c_visit 参数（论文默认 50）。
    pub c_visit: f32,
    /// Gumbel σ 变换的 c_scale 参数（论文默认 1.0）。
    pub c_scale: f32,
}

impl Default for GumbelConfig {
    fn default() -> Self {
        Self {
            num_sampled_actions: 16,
            num_simulations: 50,
            c_visit: 50.0,
            c_scale: 1.0,
        }
    }
}

/// Reanalyze 配置（用最新/目标网络重跑 MCTS 刷新旧轨迹目标）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReanalyzeConfig {
    /// 每次训练采样的整局中，以此概率 reanalyze ∈ [0,1]。
    ///
    /// EZ 推荐开启（batch-time）以提样本效率；但 CPU only 下每个被重算位置 = 一整棵 MCTS，
    /// 故 `Default` 保守置 `0.0`，由 Phase 1 EZ 示例按算力调高（与 `MuZeroConfig` 一致的取舍）。
    pub fraction: f32,
    /// reanalyze 用 target net（`true`）还是 online net（`false`）。EZ 增强用 target net。
    pub use_target_net: bool,
}

impl Default for ReanalyzeConfig {
    fn default() -> Self {
        Self {
            fraction: 0.0,
            use_target_net: true,
        }
    }
}

/// Target network 配置（EZ 稳定性增强；base MuZero 不需要）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetConfig {
    /// 是否启用 target net（Phase 1 +target 消融开关）。
    pub enabled: bool,
    /// EMA 软更新系数 τ（`sync_interval == 0` 时生效）。
    pub tau: f32,
    /// hard update 间隔（步）：`> 0` 走 hard copy；`== 0` 走 EMA（用 `tau`）。
    pub sync_interval: u32,
}

impl Default for TargetConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tau: 0.01,
            sync_interval: 0,
        }
    }
}

/// EZ 各损失项系数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EzLossConfig {
    /// value loss 系数。
    pub value_coef: f32,
    /// reward / value-prefix loss 系数。
    pub reward_coef: f32,
    /// policy loss 系数。
    pub policy_coef: f32,
    /// 自监督 consistency loss 系数（EZ 论文 ~2.0）。
    pub consistency_coef: f32,
}

impl Default for EzLossConfig {
    fn default() -> Self {
        Self {
            value_coef: 0.25,
            reward_coef: 1.0,
            policy_coef: 1.0,
            consistency_coef: 2.0,
        }
    }
}

/// EfficientZero V2 总配置（组合 base + search + gumbel + reanalyze + target + loss）。
///
/// 复用 `MuZeroConfig`（训练/搜索通用超参）与 `MctsConfig`（PUCT 搜索参数），EZ 专属增量
/// 各自独立子配置，互不覆盖。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EfficientZeroConfig {
    /// 训练/搜索通用超参（γ、k_unroll、td_steps、lr、buffer 等）。
    pub base: MuZeroConfig,
    /// PUCT 搜索参数（离散 CartPole / 五子棋用）。
    pub search: MctsConfig,
    /// Gumbel 搜索参数（连续 / 混合 / 大动作空间用）。
    pub gumbel: GumbelConfig,
    /// Reanalyze 参数。
    pub reanalyze: ReanalyzeConfig,
    /// Target network 参数。
    pub target: TargetConfig,
    /// 损失项系数。
    pub loss: EzLossConfig,
}

impl Default for EfficientZeroConfig {
    /// CartPole-v1 级 EZ 友好默认（CPU only）。换环境时覆盖相应子配置。
    fn default() -> Self {
        Self {
            base: MuZeroConfig::default(),
            search: MctsConfig::default(),
            gumbel: GumbelConfig::default(),
            reanalyze: ReanalyzeConfig::default(),
            target: TargetConfig::default(),
            loss: EzLossConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_composes_subconfigs() {
        let cfg = EfficientZeroConfig::default();
        // 复用 MuZero/MCTS 默认
        assert_eq!(cfg.base, MuZeroConfig::default());
        assert_eq!(
            cfg.search.num_simulations,
            MctsConfig::default().num_simulations
        );
        // EZ 增量默认：reanalyze 保守关闭、target 关闭（消融时开）
        assert_eq!(
            cfg.reanalyze.fraction, 0.0,
            "Default reanalyze 关闭（CPU 友好）"
        );
        assert!(
            cfg.reanalyze.use_target_net,
            "EZ reanalyze 默认用 target net"
        );
        assert!(!cfg.target.enabled, "target net 默认关闭，消融时开");
        assert!((cfg.loss.consistency_coef - 2.0).abs() < 1e-6);
    }

    #[test]
    fn per_env_override_is_localized() {
        // 换 Pendulum：只调 Gumbel 子配置，不影响 base/search
        let cfg = EfficientZeroConfig {
            gumbel: GumbelConfig {
                num_sampled_actions: 8,
                ..GumbelConfig::default()
            },
            ..EfficientZeroConfig::default()
        };
        assert_eq!(cfg.gumbel.num_sampled_actions, 8);
        assert_eq!(
            cfg.base,
            MuZeroConfig::default(),
            "base 不受 gumbel 覆盖影响"
        );
    }
}
