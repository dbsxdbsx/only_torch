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
    sched.init(&children, 0.0, &cfg, &mut rng);
    let mut counts = [0usize; 2];
    for sim in 0..cfg.num_simulations as usize {
        if let Some(i) = sched.next_root_child(&children, sim, &cfg) {
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
    a.init(&children, 0.0, &cfg, &mut ra);
    b.init(&children, 0.0, &cfg, &mut rb);
    for sim in 0..cfg.num_simulations as usize {
        let _ = a.next_root_child(&children, sim, &cfg);
        let _ = b.next_root_child(&children, sim, &cfg);
    }
    assert_eq!(
        a.final_recommendation(&children),
        b.final_recommendation(&children)
    );
}
