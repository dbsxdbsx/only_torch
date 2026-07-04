//! MyZero 策略目标构造
//!
//! **completedQ 改进策略**（Gumbel MuZero，Danihelka et al. 2022, Eq.10-12）：
//! 用「Q 值算出的改进策略 π'」替代 visit-count 作为策略网络训练目标。
//! 动机：visit-count 在**少模拟**时分辨率低、噪声大（两整数之比），是低模拟不稳的根因；
//! π' 直接由 Q 值构造，少模拟下仍是一次有保证的策略提升（与 Grill 2020 的正则化策略优化同源，
//! 但闭式、无需二分搜索 α）。

use crate::rl::mcts::{ChildStat, SearchResult};

/// 从 MCTS 搜索结果构造策略训练目标（visit-count 或 completedQ，与 self-play / reanalyze 共用）。
///
/// `action_dim` 为完整 joint 动作数；Sampled MuZero 搜索子集长度为 K 时，会投射回全长向量。
/// `root_to_play`：根节点执子方（单智能体恒 0）——双人零和时子节点 value 是子方视角，
/// completedQ 的 Q 需按 negamax 翻转回根方视角（与 PUCT select / backup 同口径）。
pub fn mcts_policy_target(
    result: &SearchResult,
    cq: Option<(f32, f32)>,
    action_dim: usize,
    root_to_play: u8,
) -> Vec<f32> {
    let partial = match cq {
        Some((c_visit, c_scale)) => completed_q_policy_target(
            &result.children,
            result.network_value,
            result.q_range,
            c_visit,
            c_scale,
            root_to_play,
        ),
        None => result.learn_policy.clone(),
    };
    scatter_policy_target(&result.children, &partial, action_dim)
}

/// 把搜索子集（K 路）上的策略目标投射回完整动作空间 `[0, action_dim)`。
///
/// 未出现在 `children` 中的动作概率为 0，再对全长 renormalize（Sampled MuZero 训练蒸馏用）。
pub fn scatter_policy_target(
    children: &[ChildStat],
    partial: &[f32],
    action_dim: usize,
) -> Vec<f32> {
    assert!(action_dim > 0, "action_dim 必须 > 0");
    if partial.len() == action_dim {
        return partial.to_vec();
    }
    assert_eq!(
        children.len(),
        partial.len(),
        "Sampled policy target 要求 children 与 partial 一一对应"
    );
    let mut full = vec![0.0; action_dim];
    for (child, &p) in children.iter().zip(partial.iter()) {
        let idx = child.action_id.index();
        assert!(
            idx < action_dim,
            "Sampled policy target 动作 id {idx} 超出 action_dim={action_dim}"
        );
        full[idx] = p;
    }
    let sum: f32 = full.iter().sum();
    if sum > 1e-8 {
        for x in &mut full {
            *x /= sum;
        }
    } else {
        full.fill(1.0 / action_dim as f32);
    }
    full
}

/// completedQ 改进策略目标（闭式，返回与 `children` 平行的概率向量）。
///
/// `π'(a) ∝ prior(a) · exp(σ(completedQ(a)))`，其中：
/// - `completedQ(a) = Q(a)`（已访问）或 `vmix`（未访问；Appendix D / mctx completed-by-mix-value）
/// - `Q(a) = reward(a) + discount(a) · value_sum(a)/visit_count(a)`
/// - completedQ 用 `q_range`（tree-level 全局 Q 范围）归一化到 `[0,1]`；`None` 时 fallback 到
///   局部 over-children min-max。用全局范围是为修复 `|A|=2` 时局部 min-max 把两动作恒拉成
///   `{0,1}`、σ 退化为符号开关的问题（见 [`crate::rl::mcts::MinMaxStats::range`]）。
/// - `σ(q) = (c_visit + max_b N(b)) · c_scale · q`（Eq.8）
///
/// 由于 `softmax(logits + σ)` 中常数偏移相消，等价实现为 `prior·exp(σ·norm_q)` 归一化
/// （`logits = ln prior`），无需策略 logits 原值。
///
/// 论文默认 `c_visit=50`、`c_scale=1.0`；tree-level 归一化下向量环境同样可用 1.0（旧局部 min-max 才需调小）。
pub fn completed_q_policy_target(
    children: &[ChildStat],
    v_hat_pi: f32,
    q_range: Option<(f32, f32)>,
    c_visit: f32,
    c_scale: f32,
    root_to_play: u8,
) -> Vec<f32> {
    let n = children.len();
    if n == 0 {
        return Vec::new();
    }

    // 每动作 completedQ：已访问用搜索 Q，未访问补 vmix。
    // 这里对齐 mctx qtransform_completed_by_mix_value：
    // qvalues -> complete_by_mix_value -> rescale -> visit_scale * value_scale。
    let mix_value = v_mix(children, v_hat_pi, root_to_play);
    let completed: Vec<f32> = children
        .iter()
        .map(|c| child_q(c, root_to_play).unwrap_or(mix_value))
        .collect();

    // σ 归一化的 Q 范围：优先用 tree-level 全局范围（search 维护的 MinMaxStats）。
    // |A|=2 时局部 over-children min-max 恒把两动作拉成 {0,1}，σ 退化为与 Q 差无关的符号开关；
    // 改用整棵搜索树的 Q 范围后，根动作的 norm_q 才反映其在全局分布里的真实相对位置。
    // 无有效全局范围（空搜索 / 测试直接构造 ChildStat）时 fallback 到 completed qvalues 局部 min-max。
    let (lo, hi) = match q_range {
        Some((lo, hi)) if hi > lo => (lo, hi),
        _ => {
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for &q in &completed {
                lo = lo.min(q);
                hi = hi.max(q);
            }
            (lo, hi)
        }
    };
    let range = (hi - lo).max(1e-8);

    let max_n = children.iter().map(|c| c.visit_count).max().unwrap_or(0) as f32;
    let sigma_scale = (c_visit + max_n) * c_scale;

    // logits = ln(prior) + σ·norm_q；数值稳定 softmax（减最大值）
    let logits: Vec<f32> = (0..n)
        .map(|i| {
            // tree-level range 下根动作 Q 必在范围内；vmix 填充值可能略超界，clamp 保险。
            let norm_q = ((completed[i] - lo) / range).clamp(0.0, 1.0);
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

/// Gumbel MuZero Appendix D / mctx 的 mixed value。
///
/// `v_hat_pi` 是当前状态 value network 的原始估计；已访问动作的 Q 按 prior 加权，
/// 再按总访问数与 `v_hat_pi` 做一致性混合。未访问动作在 completedQ 中用该值填充。
pub(super) fn v_mix(children: &[ChildStat], v_hat_pi: f32, root_to_play: u8) -> f32 {
    let total_visits: u32 = children.iter().map(|c| c.visit_count).sum();
    if total_visits == 0 {
        return v_hat_pi;
    }

    let prior_sum: f32 = children
        .iter()
        .filter(|c| c.visit_count > 0)
        .map(|c| safe_prior(c.prior))
        .sum();
    if prior_sum <= 0.0 || !prior_sum.is_finite() {
        return v_hat_pi;
    }

    let weighted_q: f32 = children
        .iter()
        .filter(|c| c.visit_count > 0)
        .filter_map(|c| child_q(c, root_to_play).map(|q| safe_prior(c.prior) * q / prior_sum))
        .sum();
    (v_hat_pi + total_visits as f32 * weighted_q) / (total_visits as f32 + 1.0)
}

/// 已访问子节点的 Q（根方视角）：`Q(a) = r(a) + discount·perspective·V(child)`。
///
/// `perspective`：子节点 value_sum 是**子方视角**累计（backup 契约），双人零和时
/// 子方 ≠ 根方须取负（negamax）；单智能体 to_play 恒同 → 恒 +1，行为不变。
fn child_q(c: &ChildStat, root_to_play: u8) -> Option<f32> {
    (c.visit_count > 0).then(|| {
        let child_v = c.value_sum / c.visit_count as f32;
        let perspective = if c.to_play == root_to_play { 1.0 } else { -1.0 };
        c.reward + c.discount * perspective * child_v
    })
}

fn safe_prior(prior: f32) -> f32 {
    if prior.is_finite() {
        prior.max(1e-12)
    } else {
        1e-12
    }
}
