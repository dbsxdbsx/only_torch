//! MuZero 训练 + 搜索超参数容器（库级，跨 `*Zero` 家族复用）
//!
//! 把示例里分散的魔法常量收敛为一个可配置结构，供 MuZero / EfficientZero / 未来变体共用。
//!
//! # 按环境配置（重要）
//! `num_simulations` 等搜索/训练预算**应按环境调整**，库不钦定单一值：
//! - 完全信息棋类（AlphaZero/MuZero 自对弈）：每步 ~800 次模拟；
//! - Atari / 低维向量控制（如 CartPole）：每步 ~50 次模拟即可。
//!
//! 调用方按环境构造 [`MuZeroConfig`]（`Default` 给的是 CartPole 级 CPU 友好默认），
//! 而不是在训练循环里写死常量。

/// MuZero 系列训练 + 搜索超参数
///
/// 仅承载**算法超参**；网络拓扑（latent/obs/action 维度、隐藏层）属环境相关结构，留在示例。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MuZeroConfig {
    /// 折扣因子 γ
    pub gamma: f32,
    /// K 步 unroll 展开步数
    pub k_unroll: usize,
    /// n-step bootstrap 的步数（value target）
    pub td_steps: usize,
    /// 每步 MCTS 模拟数（**按环境调**：棋类 ~800、Atari/向量 ~50）
    pub num_simulations: u32,
    /// 学习率
    pub lr: f32,
    /// 每次训练采样的整局数（batch）
    pub batch_games: usize,
    /// 每个 episode 结束后的训练次数
    pub trains_per_episode: usize,
    /// replay buffer 容量（按整局计）
    pub buffer_capacity: usize,
    /// 开始训练前需累积的局数
    pub start_training_after: usize,
    /// reanalyze 比例 ∈ [0,1]：每次训练采样的整局中，以此概率用最新网络重跑 MCTS
    /// 刷新 policy/value 目标。
    ///
    /// `0.0` = 关闭（base MuZero 即可达标，CPU 友好）。reanalyze 是「算力换样本效率」：
    /// 每个被重算的位置 = 一整棵 MCTS，CPU only 下开销显著，故默认关闭、按需调高。
    pub reanalyze_fraction: f32,
}

impl Default for MuZeroConfig {
    /// CartPole 级别（低维向量 obs / CPU only）的合理默认。
    ///
    /// 换环境时按需覆盖字段，尤其 [`num_simulations`](Self::num_simulations)
    /// 与 [`td_steps`](Self::td_steps)。
    fn default() -> Self {
        Self {
            gamma: 0.997,
            k_unroll: 5,
            td_steps: 50,
            num_simulations: 50,
            lr: 0.02,
            batch_games: 8,
            trains_per_episode: 8,
            buffer_capacity: 1000,
            start_training_after: 10,
            reanalyze_fraction: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_cartpole_friendly() {
        let cfg = MuZeroConfig::default();
        assert_eq!(cfg.num_simulations, 50);
        assert_eq!(cfg.k_unroll, 5);
        assert!((cfg.gamma - 0.997).abs() < 1e-6);
        assert_eq!(
            cfg.reanalyze_fraction, 0.0,
            "reanalyze 默认关闭（CPU 友好）"
        );
    }

    #[test]
    fn per_env_override() {
        // 棋类自对弈：模拟数显著更高
        let board = MuZeroConfig {
            num_simulations: 800,
            ..MuZeroConfig::default()
        };
        assert_eq!(board.num_simulations, 800);
        // 其余字段保持默认
        assert_eq!(board.k_unroll, MuZeroConfig::default().k_unroll);
    }
}
