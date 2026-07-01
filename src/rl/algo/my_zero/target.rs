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
pub fn mcts_policy_target(
    result: &SearchResult,
    cq: Option<(f32, f32)>,
    action_dim: usize,
) -> Vec<f32> {
    let partial = match cq {
        Some((c_visit, c_scale)) => completed_q_policy_target(
            &result.children,
            result.network_value,
            result.q_range,
            c_visit,
            c_scale,
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
) -> Vec<f32> {
    let n = children.len();
    if n == 0 {
        return Vec::new();
    }

    // 每动作 completedQ：已访问用搜索 Q，未访问补 vmix。
    // 这里对齐 mctx qtransform_completed_by_mix_value：
    // qvalues -> complete_by_mix_value -> rescale -> visit_scale * value_scale。
    let mix_value = v_mix(children, v_hat_pi);
    let completed: Vec<f32> = children
        .iter()
        .map(|c| child_q(c).unwrap_or(mix_value))
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
fn v_mix(children: &[ChildStat], v_hat_pi: f32) -> f32 {
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
        .filter_map(|c| child_q(c).map(|q| safe_prior(c.prior) * q / prior_sum))
        .sum();
    (v_hat_pi + total_visits as f32 * weighted_q) / (total_visits as f32 + 1.0)
}

fn child_q(c: &ChildStat) -> Option<f32> {
    (c.visit_count > 0).then(|| {
        let child_v = c.value_sum / c.visit_count as f32;
        c.reward + c.discount * child_v
    })
}

fn safe_prior(prior: f32) -> f32 {
    if prior.is_finite() {
        prior.max(1e-12)
    } else {
        1e-12
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::mcts::ActionPayload;

    fn child(prior: f32, visit: u32, value_sum: f32, reward: f32) -> ChildStat {
        ChildStat {
            action_id: 0.into(),
            action: ActionPayload::Discrete(0),
            visit_count: visit,
            value_sum,
            prior,
            reward,
            to_play: 0,
            discount: 1.0,
        }
    }

    fn child_with_discount(
        prior: f32,
        visit: u32,
        value_sum: f32,
        reward: f32,
        discount: f32,
    ) -> ChildStat {
        ChildStat {
            discount,
            ..child(prior, visit, value_sum, reward)
        }
    }

    #[test]
    fn scatter_sampled_subset_to_full_action_dim() {
        let children = vec![
            ChildStat {
                action_id: 1.into(),
                action: ActionPayload::Discrete(1),
                visit_count: 3,
                value_sum: 0.0,
                prior: 0.2,
                reward: 0.0,
                to_play: 0,
                discount: 1.0,
            },
            ChildStat {
                action_id: 4.into(),
                action: ActionPayload::Discrete(4),
                visit_count: 7,
                value_sum: 0.0,
                prior: 0.3,
                reward: 0.0,
                to_play: 0,
                discount: 1.0,
            },
        ];
        let partial = vec![0.3, 0.7];
        let full = scatter_policy_target(&children, &partial, 7);
        assert_eq!(full.len(), 7);
        assert!((full[1] - 0.3).abs() < 1e-5);
        assert!((full[4] - 0.7).abs() < 1e-5);
        assert!((full.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!((full[0] + full[2] + full[3] + full[5] + full[6]).abs() < 1e-5);
    }

    #[test]
    fn scatter_uses_action_id_not_payload_shape() {
        let children = vec![ChildStat {
            action_id: 1.into(),
            action: ActionPayload::Continuous(vec![0.0]),
            visit_count: 1,
            value_sum: 0.0,
            prior: 1.0,
            reward: 0.0,
            to_play: 0,
            discount: 1.0,
        }];
        let full = scatter_policy_target(&children, &[1.0], 3);
        assert_eq!(full, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "超出 action_dim")]
    fn scatter_rejects_out_of_range_action_id() {
        let children = vec![ChildStat {
            action_id: 3.into(),
            action: ActionPayload::Discrete(0),
            visit_count: 1,
            value_sum: 0.0,
            prior: 1.0,
            reward: 0.0,
            to_play: 0,
            discount: 1.0,
        }];
        let _ = scatter_policy_target(&children, &[1.0], 2);
    }

    #[test]
    #[should_panic(expected = "一一对应")]
    fn scatter_rejects_children_partial_len_mismatch() {
        let children = vec![ChildStat {
            action_id: 0.into(),
            action: ActionPayload::Discrete(0),
            visit_count: 1,
            value_sum: 0.0,
            prior: 1.0,
            reward: 0.0,
            to_play: 0,
            discount: 1.0,
        }];
        let _ = scatter_policy_target(&children, &[0.5, 0.5], 3);
    }

    #[test]
    fn empty_children_returns_empty() {
        assert!(completed_q_policy_target(&[], 0.0, None, 50.0, 0.1).is_empty());
    }

    #[test]
    fn higher_q_gets_higher_prob() {
        // 两动作同 prior；动作1 的 Q 更高 → π' 应偏向动作1（fallback 局部 min-max 路径）
        let children = vec![
            child(0.5, 5, 0.0, 0.0), // Q=0
            child(0.5, 5, 5.0, 0.0), // Q=1.0
        ];
        let t = completed_q_policy_target(&children, 0.5, None, 50.0, 1.0);
        assert!((t.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(t[1] > t[0], "高 Q 动作应获更高概率：{t:?}");
    }

    #[test]
    fn completed_q_includes_reward_and_discount() {
        // 两动作 child value 相同，但动作 1 即时 reward 更高、discount 更低后总 Q 仍更高。
        let children = vec![
            child_with_discount(0.5, 5, 10.0, 0.0, 1.0), // Q = 2.0
            child_with_discount(0.5, 5, 10.0, 2.0, 0.5), // Q = 3.0
        ];
        let t = completed_q_policy_target(&children, 0.0, None, 50.0, 1.0);
        assert!(
            t[1] > t[0],
            "completedQ 应按 reward + discount * value 排序：{t:?}"
        );
    }

    #[test]
    fn equal_q_equal_prior_is_uniform() {
        let children = vec![child(0.5, 5, 2.5, 0.0), child(0.5, 5, 2.5, 0.0)];
        let t = completed_q_policy_target(&children, 0.5, None, 50.0, 1.0);
        assert!((t[0] - t[1]).abs() < 1e-5, "同 Q 同 prior 应均匀：{t:?}");
    }

    #[test]
    fn all_unvisited_falls_back_uniform() {
        let children = vec![child(0.5, 0, 0.0, 0.0), child(0.5, 0, 0.0, 0.0)];
        let t = completed_q_policy_target(&children, 0.0, None, 50.0, 0.1);
        assert!((t[0] - 0.5).abs() < 1e-5 && (t[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn v_mix_all_unvisited_returns_network_value() {
        let children = vec![child(0.5, 0, 0.0, 0.0), child(0.5, 0, 0.0, 0.0)];
        assert!((v_mix(&children, 0.3) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn v_mix_all_visited_matches_prior_weighted_q() {
        let children = vec![
            child(0.25, 2, 2.0, 0.0), // Q=1.0
            child(0.75, 3, 9.0, 0.0), // Q=3.0
        ];
        let expected_weighted_q = 0.25 * 1.0 + 0.75 * 3.0;
        let expected = (0.5 + 5.0 * expected_weighted_q) / 6.0;
        assert!((v_mix(&children, 0.5) - expected).abs() < 1e-6);
    }

    #[test]
    fn v_mix_mixed_visit_numeric_match() {
        let children = vec![
            child(0.6, 10, 8.0, 0.0), // Q=0.8
            child(0.4, 0, 0.0, 0.0),
        ];
        let expected = (0.0 + 10.0 * 0.8) / 11.0;
        assert!((v_mix(&children, 0.0) - expected).abs() < 1e-6);
    }

    #[test]
    fn unvisited_baseline_uses_vmix_not_raw_network_value() {
        // 动作 0 已访问 Q≈0.8；动作 1 未访问。
        // Appendix D：未访问应补 vmix，而非裸 vπ；vmix 会向已访问 Q 靠拢但保留一份 value 先验。
        let children = vec![child(0.5, 10, 8.0, 0.0), child(0.5, 0, 0.0, 0.0)];
        let mix = v_mix(&children, 0.0);
        assert!((mix - (8.0 / 11.0)).abs() < 1e-6);
        let t_correct = completed_q_policy_target(&children, 0.0, None, 50.0, 1.0);
        let t_wrong = completed_q_policy_target(&children, 0.8, None, 50.0, 1.0);
        assert!(
            t_correct[0] > t_correct[1],
            "vmix 低于已访问 Q 时应偏向已访问高 Q：{t_correct:?}"
        );
        assert!(
            (t_wrong[0] - t_wrong[1]).abs() < 1e-5,
            "v̂π 已等于已访问 Q 时 vmix=Q，两动作 completedQ 相同：{t_wrong:?}"
        );
    }

    #[test]
    fn fallback_local_minmax_invariant_to_positive_value_scale() {
        // fallback（q_range=None）路径：局部 min-max 对 value 整体缩放不变。
        let children = vec![
            child_with_discount(0.5, 5, 0.0, 0.0, 1.0), // Q=0
            child_with_discount(0.5, 5, 5.0, 0.0, 1.0), // Q=1
        ];
        let scaled = vec![
            child_with_discount(0.5, 5, 0.0, 0.0, 1.0),  // Q=0
            child_with_discount(0.5, 5, 50.0, 0.0, 1.0), // Q=10
        ];
        let a = completed_q_policy_target(&children, 0.5, None, 50.0, 0.1);
        let b = completed_q_policy_target(&scaled, 5.0, None, 50.0, 0.1);
        assert!((a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6);
    }

    #[test]
    fn fallback_local_minmax_two_action_is_sharp() {
        // fallback 路径下 |A|=2 局部 min-max 必把两动作拉成 {0,1}：这正是 tree-level range 要修的退化。
        let children = vec![
            child(0.5, 12, 0.0, 0.0),  // Q=0
            child(0.5, 12, 12.0, 0.0), // Q=1
        ];
        let vector_scale = completed_q_policy_target(&children, 0.5, None, 50.0, 0.1);
        let board_scale = completed_q_policy_target(&children, 0.5, None, 50.0, 1.0);
        assert!(
            vector_scale[1] < board_scale[1],
            "value_scale=0.1 应软于 1.0：0.1={vector_scale:?}, 1.0={board_scale:?}"
        );
    }

    // ---- tree-level range（生产路径）：σ 归一化用全局 Q 范围，修复 |A|=2 局部 min-max 退化 ----

    #[test]
    fn tree_range_small_q_gap_is_not_one_hot() {
        // |A|=2，两动作 Q 仅差 0.4，但全局树 Q 范围宽 [0,200]（CartPole 量级）。
        // 局部 min-max 会把 0.4 差拉成满幅 {0,1} → near one-hot；tree-level range 下应仍接近 uniform。
        let children = vec![
            child(0.5, 10, 1796.0, 0.0), // Q=179.6
            child(0.5, 10, 1800.0, 0.0), // Q=180.0
        ];
        let tree = completed_q_policy_target(&children, 180.0, Some((0.0, 200.0)), 50.0, 1.0);
        let local = completed_q_policy_target(&children, 180.0, None, 50.0, 1.0);
        assert!(
            tree[1] > tree[0] && tree[1] < 0.65,
            "tree-range 下小 Q 差不应 one-hot：{tree:?}"
        );
        assert!(
            local[1] > 0.99,
            "对照：局部 min-max 把同样小 Q 差拉成 near one-hot：{local:?}"
        );
    }

    #[test]
    fn tree_range_target_monotonic_in_q_gap() {
        // 固定全局范围，Q 差增大 → 高 Q 动作概率单调增。
        let range = Some((0.0, 200.0));
        let gap_small = vec![child(0.5, 10, 1000.0, 0.0), child(0.5, 10, 1040.0, 0.0)]; // Q 100 vs 104
        let gap_large = vec![child(0.5, 10, 1000.0, 0.0), child(0.5, 10, 1400.0, 0.0)]; // Q 100 vs 140
        let t_small = completed_q_policy_target(&gap_small, 100.0, range, 50.0, 1.0);
        let t_large = completed_q_policy_target(&gap_large, 100.0, range, 50.0, 1.0);
        assert!(
            t_large[1] > t_small[1],
            "Q 差越大目标越偏向高 Q 动作：small={t_small:?}, large={t_large:?}"
        );
    }

    #[test]
    fn tree_range_invariant_to_global_scale() {
        // Q 与全局范围同比例缩放（reward_scale 效应）→ 目标形状不变。
        let base = vec![child(0.5, 10, 800.0, 0.0), child(0.5, 10, 1200.0, 0.0)]; // Q 80 vs 120
        let scaled = vec![child(0.5, 10, 80.0, 0.0), child(0.5, 10, 120.0, 0.0)]; // Q 8 vs 12（×0.1）
        let a = completed_q_policy_target(&base, 100.0, Some((0.0, 200.0)), 50.0, 1.0);
        let b = completed_q_policy_target(&scaled, 10.0, Some((0.0, 20.0)), 50.0, 1.0);
        assert!(
            (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5,
            "同比例缩放目标应不变：a={a:?}, b={b:?}"
        );
    }

    #[test]
    fn tree_range_degenerate_falls_back_without_panic() {
        // hi==lo 的退化范围应被忽略，fallback 到局部 min-max，不 panic、不 NaN。
        let children = vec![child(0.5, 5, 0.0, 0.0), child(0.5, 5, 5.0, 0.0)];
        let t = completed_q_policy_target(&children, 0.5, Some((3.0, 3.0)), 50.0, 1.0);
        assert!(
            (t.iter().sum::<f32>() - 1.0).abs() < 1e-5,
            "退化 range 应仍输出合法分布：{t:?}"
        );
        assert!(t.iter().all(|p| p.is_finite()));
    }
}
