//! MyZero 统一配置
//!
//! 组合 MuZero 基础超参 + MCTS 搜索参数 + 消融特征开关，
//! 一个 config 适配所有环境（通过覆盖字段调参，无需切换算法）。

use super::feature::FeatureSet;
use crate::rl::algo::muzero::MuZeroConfig;
use crate::rl::mcts::MctsConfig;

/// MyZero 统一配置
///
/// `Default` 给出 CartPole-v1 级 CPU 友好起点（全关 = S0 base）。
/// 换环境时覆盖 `search` / `features` 字段。
#[derive(Debug, Clone, PartialEq)]
pub struct MyZeroConfig {
    /// 训练超参（gamma / k_unroll / td_steps / lr / buffer 等）
    pub base: MuZeroConfig,
    /// MCTS 搜索参数（num_simulations / PUCT 系数 / 温度等）
    pub search: MctsConfig,
    /// 消融特征开关
    pub features: FeatureSet,
}

impl Default for MyZeroConfig {
    fn default() -> Self {
        Self {
            base: MuZeroConfig::default(),
            search: MctsConfig::default(),
            features: FeatureSet::base(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_base_muzero() {
        let cfg = MyZeroConfig::default();
        assert_eq!(cfg.features, FeatureSet::base(), "默认全关 = S0 base");
        assert_eq!(cfg.base, MuZeroConfig::default());
    }

    #[test]
    fn single_feature_override() {
        let cfg = MyZeroConfig {
            features: FeatureSet {
                consistency: true,
                ..FeatureSet::default()
            },
            ..MyZeroConfig::default()
        };
        assert!(cfg.features.consistency);
        assert!(!cfg.features.value_prefix, "只开 S1，其余不变");
    }
}
