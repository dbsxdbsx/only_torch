//! Gumbel 根调度器测试：Top-m 截断、Sequential Halving 预算分配、seed 确定性

use crate::rl::mcts::gumbel::GumbelRootScheduler;
use crate::rl::mcts::traits::RootScheduler;
use crate::rl::mcts::types::ActionPayload;
use crate::rl::mcts::{ChildStat, MctsConfig};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn child(prior: f32, visit: u32, value_sum: f32) -> ChildStat {
    ChildStat {
        action_id: 0.into(),
        action: ActionPayload::Discrete(0),
        visit_count: visit,
        value_sum,
        prior,
        reward: 0.0,
        to_play: 0,
        discount: 1.0,
    }
}

#[test]
fn gumbel_top_k_respects_max_m() {
    let mut rng = StdRng::seed_from_u64(1);
    let mut sched = GumbelRootScheduler::new(2, 50.0, 1.0);
    let children: Vec<ChildStat> = (0..5)
        .map(|_| child(0.2, 0, 0.0))
        .enumerate()
        .map(|(i, mut c)| {
            c.action_id = i.into();
            c.action = ActionPayload::Discrete(i);
            c
        })
        .collect();
    sched.init(
        &children,
        0.0,
        0,
        &MctsConfig {
            num_simulations: 20,
            ..MctsConfig::default()
        },
        &mut rng,
    );
    assert_eq!(sched.active.len(), 2);
}

#[test]
fn sequential_halving_allocates_all_sims_to_active_set() {
    let mut rng = StdRng::seed_from_u64(42);
    let children = vec![child(0.5, 0, 0.0), child(0.5, 0, 0.0)];
    let cfg = MctsConfig {
        num_simulations: 8,
        ..MctsConfig::default()
    };
    let mut sched = GumbelRootScheduler::new(16, 50.0, 1.0);
    sched.init(&children, 0.0, 0, &cfg, &mut rng);
    let mut counts = [0usize; 2];
    for sim in 0..cfg.num_simulations as usize {
        if let Some(i) = sched.next_root_child(&children, sim, &cfg, None) {
            counts[i] += 1;
        }
    }
    assert_eq!(counts[0] + counts[1], cfg.num_simulations as usize);
    assert!(counts[0] > 0 && counts[1] > 0);
}

#[test]
fn final_recommendation_is_deterministic_with_seed() {
    let children = vec![child(0.3, 4, 0.0), child(0.7, 4, 4.0)];
    let cfg = MctsConfig {
        num_simulations: 10,
        ..MctsConfig::default()
    };
    let mut a = GumbelRootScheduler::new(16, 50.0, 1.0);
    let mut b = GumbelRootScheduler::new(16, 50.0, 1.0);
    let mut ra = StdRng::seed_from_u64(7);
    let mut rb = StdRng::seed_from_u64(7);
    a.init(&children, 0.0, 0, &cfg, &mut ra);
    b.init(&children, 0.0, 0, &cfg, &mut rb);
    for sim in 0..cfg.num_simulations as usize {
        let _ = a.next_root_child(&children, sim, &cfg, None);
        let _ = b.next_root_child(&children, sim, &cfg, None);
    }
    assert_eq!(
        a.final_recommendation(&children, None),
        b.final_recommendation(&children, None)
    );
}

/// greedy（temperature=0）推荐必须跨 rng 确定，且选中 Q 最优动作——
/// 回归负结果 issue §7.1：旧实现 eval 路径仍注入 Gumbel 噪声，`|A|=2` 时
/// 噪声 std（≈1.28）不小于 σ 信号差，greedy eval 近半随机。
#[test]
fn greedy_final_recommendation_is_noise_free() {
    // 动作 1 的 Q 明显更优（value_sum/visit = 1.0 vs 0.1），prior 反而偏向动作 0
    let children = vec![child(0.7, 4, 0.4), child(0.3, 4, 4.0)];
    let cfg = MctsConfig {
        num_simulations: 10,
        temperature: 0.0,
        ..MctsConfig::default()
    };
    for seed in 0..20u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut sched = GumbelRootScheduler::new(16, 50.0, 1.0);
        sched.init(&children, 0.5, 0, &cfg, &mut rng);
        assert_eq!(
            sched.final_recommendation(&children, None),
            Some(1),
            "greedy 推荐应确定选 Q 最优动作（seed={seed} 不应影响结果）"
        );
    }
}

/// self-play（temperature>0）路径保留 Gumbel 噪声：不同 seed 应能产生不同推荐
/// （防止去噪修复误伤探索路径）。
#[test]
fn selfplay_recommendation_keeps_gumbel_noise() {
    // 两动作 Q/prior 完全相同 → 打分差全部来自 Gumbel 噪声
    let children = vec![child(0.5, 4, 2.0), child(0.5, 4, 2.0)];
    let cfg = MctsConfig {
        num_simulations: 10,
        temperature: 1.0,
        ..MctsConfig::default()
    };
    let mut seen = std::collections::HashSet::new();
    for seed in 0..30u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut sched = GumbelRootScheduler::new(16, 50.0, 1.0);
        sched.init(&children, 0.5, 0, &cfg, &mut rng);
        seen.insert(sched.final_recommendation(&children, None));
    }
    assert!(
        seen.len() > 1,
        "self-play 路径应保留 Gumbel 噪声多样性，got {seen:?}"
    );
}

/// σ 归一化应优先用 tree-level q_range：范围远宽于局部 Q 差时，σ 被压平、
/// 推荐由 `g + ln π` 主导（回归 §7.2 局部 min-max 在 |A|=2 恒拉成 {0,1} 的退化）。
#[test]
fn halving_score_uses_tree_level_q_range() {
    // 两动作 Q 差微小（2.0 vs 2.01），prior 强烈偏向动作 0
    let children = vec![child(0.9, 4, 8.0), child(0.1, 4, 8.04)];
    let cfg = MctsConfig {
        num_simulations: 10,
        temperature: 0.0,
        ..MctsConfig::default()
    };
    let mut rng = StdRng::seed_from_u64(3);

    // 局部归一化（None）：微小 Q 差被拉成 {0,1}，σ=(50+4)·1.0 压倒 ln π → 误选动作 1
    let mut local = GumbelRootScheduler::new(16, 50.0, 1.0);
    local.init(&children, 2.0, 0, &cfg, &mut rng);
    assert_eq!(local.final_recommendation(&children, None), Some(1));

    // tree-level 宽范围（整棵树 Q ∈ [0,10]）：norm_q 差仅 0.001，σ 差 ≈ 0.054，
    // ln(0.9/0.1) ≈ 2.2 主导 → 正确选 prior/Q 综合更优的动作 0
    let mut tree = GumbelRootScheduler::new(16, 50.0, 1.0);
    let mut rng2 = StdRng::seed_from_u64(3);
    tree.init(&children, 2.0, 0, &cfg, &mut rng2);
    assert_eq!(
        tree.final_recommendation(&children, Some((0.0, 10.0))),
        Some(0),
        "tree-level 归一化下微小 Q 差不应压倒 prior"
    );
}
