//! MyZero 策略目标构造
//!
//! **completedQ 改进策略**（Gumbel MuZero，Danihelka et al. 2022, Eq.10-12）：
//! 用「Q 值算出的改进策略 π'」替代 visit-count 作为策略网络训练目标。
//! 动机：visit-count 在**少模拟**时分辨率低、噪声大（两整数之比），是低模拟不稳的根因；
//! π' 直接由 Q 值构造，少模拟下仍是一次有保证的策略提升（与 Grill 2020 的正则化策略优化同源，
//! 但闭式、无需二分搜索 α）。

use crate::rl::mcts::ChildStat;

/// completedQ 改进策略目标（闭式，返回与 `children` 平行的概率向量）。
///
/// `π'(a) ∝ prior(a) · exp(σ(completedQ(a)))`，其中：
/// - `completedQ(a) = Q(a)`（已访问）或 `vπ`（未访问 → 零优势，Eq.10；`vπ` = value network）
/// - `Q(a) = reward(a) + discount(a) · value_sum(a)/visit_count(a)`
/// - completedQ 经 min-max 归一化到 `[0,1]`（参考点含 `vπ`）
/// - `σ(q) = (c_visit + max_b N(b)) · c_scale · q`（Eq.8）
///
/// 由于 `softmax(logits + σ)` 中常数偏移相消，等价实现为 `prior·exp(σ·norm_q)` 归一化
/// （`logits = ln prior`），无需策略 logits 原值。
///
/// 论文默认 `c_visit=50`；`c_scale=1.0`（棋类）/ `0.1`（图像/向量，Q 噪声较大）。
pub fn completed_q_policy_target(
    children: &[ChildStat],
    v_pi: f32,
    c_visit: f32,
    c_scale: f32,
) -> Vec<f32> {
    let n = children.len();
    if n == 0 {
        return Vec::new();
    }

    // 每动作 completedQ：已访问用搜索 Q，未访问补 vπ（零优势，Eq.10）
    let completed: Vec<f32> = children
        .iter()
        .map(|c| {
            if c.visit_count > 0 {
                let child_v = c.value_sum / c.visit_count as f32;
                c.reward + c.discount * child_v
            } else {
                v_pi
            }
        })
        .collect();

    // min-max 归一化到 [0,1]（参考点含 vπ）
    let mut lo = v_pi;
    let mut hi = v_pi;
    for &q in &completed {
        lo = lo.min(q);
        hi = hi.max(q);
    }
    let range = (hi - lo).max(1e-8);

    let max_n = children.iter().map(|c| c.visit_count).max().unwrap_or(0) as f32;
    let sigma_scale = (c_visit + max_n) * c_scale;

    // logits = ln(prior) + σ·norm_q；数值稳定 softmax（减最大值）
    let logits: Vec<f32> = (0..n)
        .map(|i| {
            let norm_q = (completed[i] - lo) / range;
            children[i].prior.max(1e-12).ln() + sigma_scale * norm_q
        })
        .collect();
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![1.0 / n as f32; n];
    }
    exps.iter().map(|&e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::mcts::ActionPayload;

    fn child(prior: f32, visit: u32, value_sum: f32, reward: f32) -> ChildStat {
        ChildStat {
            action: ActionPayload::Discrete(0),
            visit_count: visit,
            value_sum,
            prior,
            reward,
            to_play: 0,
            discount: 1.0,
        }
    }

    #[test]
    fn empty_children_returns_empty() {
        assert!(completed_q_policy_target(&[], 0.0, 50.0, 0.1).is_empty());
    }

    #[test]
    fn higher_q_gets_higher_prob() {
        // 两动作同 prior；动作1 的 Q 更高 → π' 应偏向动作1
        let children = vec![
            child(0.5, 5, 0.0, 0.0), // Q=0
            child(0.5, 5, 5.0, 0.0), // Q=1.0
        ];
        let t = completed_q_policy_target(&children, 0.5, 50.0, 1.0);
        assert!((t.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(t[1] > t[0], "高 Q 动作应获更高概率：{t:?}");
    }

    #[test]
    fn equal_q_equal_prior_is_uniform() {
        let children = vec![child(0.5, 5, 2.5, 0.0), child(0.5, 5, 2.5, 0.0)];
        let t = completed_q_policy_target(&children, 0.5, 50.0, 1.0);
        assert!((t[0] - t[1]).abs() < 1e-5, "同 Q 同 prior 应均匀：{t:?}");
    }

    #[test]
    fn all_unvisited_falls_back_uniform() {
        let children = vec![child(0.5, 0, 0.0, 0.0), child(0.5, 0, 0.0, 0.0)];
        let t = completed_q_policy_target(&children, 0.0, 50.0, 0.1);
        assert!((t[0] - 0.5).abs() < 1e-5 && (t[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn unvisited_baseline_is_v_pi_not_visit_weighted_q() {
        // 动作 0 已访问 Q≈0.8；动作 1 未访问。
        // 论文 Eq.10：未访问应补 vπ=0，而非 visit 加权 root Q≈0.8。
        let children = vec![
            child(0.5, 10, 8.0, 0.0),
            child(0.5, 0, 0.0, 0.0),
        ];
        let t_correct = completed_q_policy_target(&children, 0.0, 50.0, 1.0);
        let t_wrong = completed_q_policy_target(&children, 0.8, 50.0, 1.0);
        assert!(
            t_correct[0] > t_correct[1],
            "vπ=0 时应偏向已访问高 Q：{t_correct:?}"
        );
        assert!(
            (t_wrong[0] - t_wrong[1]).abs() < 1e-5,
            "错把未访问填成 visit 加权 Q 时两动作 completedQ 相同：{t_wrong:?}"
        );
    }
}
