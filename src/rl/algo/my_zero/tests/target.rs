//! `target.rs` 策略目标构造测试：scatter 投射 / completedQ / vmix / tree-level σ 归一化

use super::super::target::{completed_q_policy_target, scatter_policy_target, v_mix};
use crate::rl::mcts::{ActionPayload, ChildStat};

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
    assert!(completed_q_policy_target(&[], 0.0, None, 50.0, 0.1, 0).is_empty());
}

#[test]
fn higher_q_gets_higher_prob() {
    // 两动作同 prior；动作1 的 Q 更高 → π' 应偏向动作1（fallback 局部 min-max 路径）
    let children = vec![
        child(0.5, 5, 0.0, 0.0), // Q=0
        child(0.5, 5, 5.0, 0.0), // Q=1.0
    ];
    let t = completed_q_policy_target(&children, 0.5, None, 50.0, 1.0, 0);
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
    let t = completed_q_policy_target(&children, 0.0, None, 50.0, 1.0, 0);
    assert!(
        t[1] > t[0],
        "completedQ 应按 reward + discount * value 排序：{t:?}"
    );
}

/// 双人零和（negamax）：子节点 value 是子方（对手）视角，completedQ 须翻转回根方——
/// 对手视角 value 高 = 对根方坏棋，π' 应偏向对手 value 低的动作。
#[test]
fn completed_q_flips_perspective_for_opponent_children() {
    let opp_child = |value_sum: f32| ChildStat {
        to_play: 1, // 子节点执子方 = 对手
        ..child(0.5, 5, value_sum, 0.0)
    };
    // 动作0：对手视角 V=1.0（好棋给对手）；动作1：对手视角 V=0.0
    let children = vec![opp_child(5.0), opp_child(0.0)];
    let t = completed_q_policy_target(&children, 0.0, None, 50.0, 1.0, 0);
    assert!(
        t[1] > t[0],
        "根方（player 0）应偏向让对手 value 更低的动作：{t:?}"
    );
    // 同一组数据按单智能体口径（to_play 同根）则应反向——防止翻转恒开
    let same_children = vec![child(0.5, 5, 5.0, 0.0), child(0.5, 5, 0.0, 0.0)];
    let t_same = completed_q_policy_target(&same_children, 0.0, None, 50.0, 1.0, 0);
    assert!(t_same[0] > t_same[1], "单智能体路径不应翻转：{t_same:?}");
}

#[test]
fn equal_q_equal_prior_is_uniform() {
    let children = vec![child(0.5, 5, 2.5, 0.0), child(0.5, 5, 2.5, 0.0)];
    let t = completed_q_policy_target(&children, 0.5, None, 50.0, 1.0, 0);
    assert!((t[0] - t[1]).abs() < 1e-5, "同 Q 同 prior 应均匀：{t:?}");
}

#[test]
fn all_unvisited_falls_back_uniform() {
    let children = vec![child(0.5, 0, 0.0, 0.0), child(0.5, 0, 0.0, 0.0)];
    let t = completed_q_policy_target(&children, 0.0, None, 50.0, 0.1, 0);
    assert!((t[0] - 0.5).abs() < 1e-5 && (t[1] - 0.5).abs() < 1e-5);
}

#[test]
fn v_mix_all_unvisited_returns_network_value() {
    let children = vec![child(0.5, 0, 0.0, 0.0), child(0.5, 0, 0.0, 0.0)];
    assert!((v_mix(&children, 0.3, 0) - 0.3).abs() < 1e-6);
}

#[test]
fn v_mix_all_visited_matches_prior_weighted_q() {
    let children = vec![
        child(0.25, 2, 2.0, 0.0), // Q=1.0
        child(0.75, 3, 9.0, 0.0), // Q=3.0
    ];
    let expected_weighted_q = 0.25 * 1.0 + 0.75 * 3.0;
    let expected = (0.5 + 5.0 * expected_weighted_q) / 6.0;
    assert!((v_mix(&children, 0.5, 0) - expected).abs() < 1e-6);
}

#[test]
fn v_mix_mixed_visit_numeric_match() {
    let children = vec![
        child(0.6, 10, 8.0, 0.0), // Q=0.8
        child(0.4, 0, 0.0, 0.0),
    ];
    let expected = (0.0 + 10.0 * 0.8) / 11.0;
    assert!((v_mix(&children, 0.0, 0) - expected).abs() < 1e-6);
}

#[test]
fn unvisited_baseline_uses_vmix_not_raw_network_value() {
    // 动作 0 已访问 Q≈0.8；动作 1 未访问。
    // Appendix D：未访问应补 vmix，而非裸 vπ；vmix 会向已访问 Q 靠拢但保留一份 value 先验。
    let children = vec![child(0.5, 10, 8.0, 0.0), child(0.5, 0, 0.0, 0.0)];
    let mix = v_mix(&children, 0.0, 0);
    assert!((mix - (8.0 / 11.0)).abs() < 1e-6);
    let t_correct = completed_q_policy_target(&children, 0.0, None, 50.0, 1.0, 0);
    let t_wrong = completed_q_policy_target(&children, 0.8, None, 50.0, 1.0, 0);
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
    let a = completed_q_policy_target(&children, 0.5, None, 50.0, 0.1, 0);
    let b = completed_q_policy_target(&scaled, 5.0, None, 50.0, 0.1, 0);
    assert!((a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6);
}

#[test]
fn fallback_local_minmax_two_action_is_sharp() {
    // fallback 路径下 |A|=2 局部 min-max 必把两动作拉成 {0,1}：这正是 tree-level range 要修的退化。
    let children = vec![
        child(0.5, 12, 0.0, 0.0),  // Q=0
        child(0.5, 12, 12.0, 0.0), // Q=1
    ];
    let vector_scale = completed_q_policy_target(&children, 0.5, None, 50.0, 0.1, 0);
    let board_scale = completed_q_policy_target(&children, 0.5, None, 50.0, 1.0, 0);
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
    let tree = completed_q_policy_target(&children, 180.0, Some((0.0, 200.0)), 50.0, 1.0, 0);
    let local = completed_q_policy_target(&children, 180.0, None, 50.0, 1.0, 0);
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
    let t_small = completed_q_policy_target(&gap_small, 100.0, range, 50.0, 1.0, 0);
    let t_large = completed_q_policy_target(&gap_large, 100.0, range, 50.0, 1.0, 0);
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
    let a = completed_q_policy_target(&base, 100.0, Some((0.0, 200.0)), 50.0, 1.0, 0);
    let b = completed_q_policy_target(&scaled, 10.0, Some((0.0, 20.0)), 50.0, 1.0, 0);
    assert!(
        (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5,
        "同比例缩放目标应不变：a={a:?}, b={b:?}"
    );
}

#[test]
fn tree_range_degenerate_falls_back_without_panic() {
    // hi==lo 的退化范围应被忽略，fallback 到局部 min-max，不 panic、不 NaN。
    let children = vec![child(0.5, 5, 0.0, 0.0), child(0.5, 5, 5.0, 0.0)];
    let t = completed_q_policy_target(&children, 0.5, Some((3.0, 3.0)), 50.0, 1.0, 0);
    assert!(
        (t.iter().sum::<f32>() - 1.0).abs() < 1e-5,
        "退化 range 应仍输出合法分布：{t:?}"
    );
    assert!(t.iter().all(|p| p.is_finite()));
}
