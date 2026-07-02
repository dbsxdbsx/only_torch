//! Sampled MuZero 的 B / N / K 解析（团队公式见 `.issue/items/my_zero_action_space_sampled_policy.md` §2.1）。
//!
//! ```text
//! K_cfg = min( max(5, N / 2),  floor(sims × 2 / 3) )
//! K_eff = min(K_cfg, N)   // 实现侧 `sampled.rs` 亦会 clamp
//! ```

use super::config::ActionPlan;

/// Sampled MuZero 连续维默认档数 B（Hubert et al. 2021 · Appendix A）。
pub const DEFAULT_CONTINUOUS_BUCKETS: usize = 7;

/// 设计期 joint 空间与运行时 Sampled K（日志 / 测试用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampledParams {
    /// 连续每维档数 B；纯离散为 `None`。
    pub b: Option<usize>,
    /// joint 候选总数 N（= MCTS 候选动作数）。
    pub n: usize,
    /// 公式算出的 K（配置层）。
    pub k_cfg: usize,
    /// 送入 MCTS 的有效 K（`min(k_cfg, n)`）。
    pub k_effective: usize,
}

/// `K_cfg = min(max(5, N/2), floor(sims×2/3))`（N=0 时返回 1）。
pub fn compute_sampled_k_cfg(n: usize, num_simulations: u32) -> usize {
    if n == 0 {
        return 1;
    }
    let from_n = (n / 2).max(5);
    let sim_cap = (num_simulations as usize * 2 / 3).max(1);
    from_n.min(sim_cap)
}

/// `K_eff = min(K_cfg, N)`。
pub fn sampled_k_effective(k_cfg: usize, n: usize) -> usize {
    if n == 0 { 1 } else { k_cfg.max(1).min(n) }
}

/// 从 env 动作方案 + 已解析的 joint 候选数构造 B/N/K。
pub fn resolve_sampled_params(
    action: ActionPlan,
    joint_n: usize,
    num_simulations: u32,
) -> SampledParams {
    let b = match action {
        ActionPlan::Discretize { buckets } => Some(buckets),
        ActionPlan::Auto => None,
    };
    let k_cfg = compute_sampled_k_cfg(joint_n, num_simulations);
    let k_effective = sampled_k_effective(k_cfg, joint_n);
    SampledParams {
        b,
        n: joint_n,
        k_cfg,
        k_effective,
    }
}

/// 启动日志一行：`[Sampled] N=2 B=— K_cfg=5 K_eff=2 (sims=20)`。
pub fn format_sampled_log(p: &SampledParams, num_simulations: u32) -> String {
    let b =
        p.b.map(|v| v.to_string())
            .unwrap_or_else(|| "—".to_string());
    format!(
        "[Sampled] N={} B={} K_cfg={} K_eff={} (sims={num_simulations})",
        p.n, b, p.k_cfg, p.k_effective
    )
}
