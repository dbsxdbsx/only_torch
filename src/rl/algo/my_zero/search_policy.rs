//! MyZero 搜索策略选择：PUCT 基线 vs Gumbel MuZero 标准根搜索。

use super::component::Components;
use crate::rl::mcts::{ChildStat, MctsConfig};
use crate::rl::mcts::{GumbelPolicy, PuctPolicy, RootScheduler, SearchPolicy};
use rand::RngCore;

/// 按组件开关构造搜索策略（Gumbel 开启时用 [`GumbelPolicy`]，否则 [`PuctPolicy`]）。
pub(crate) enum MyZeroSearchPolicy {
    Puct(PuctPolicy),
    Gumbel(GumbelPolicy),
}

impl MyZeroSearchPolicy {
    pub(crate) fn from_components(c: &Components) -> Self {
        if c.gumbel {
            Self::Gumbel(GumbelPolicy::new().with_halving_scale(c.cq_c_visit, c.cq_c_scale))
        } else {
            Self::Puct(PuctPolicy::new())
        }
    }
}

impl SearchPolicy for MyZeroSearchPolicy {
    fn prepare_root(&self, children: &mut [ChildStat], cfg: &MctsConfig, rng: &mut dyn RngCore) {
        match self {
            Self::Puct(p) => p.prepare_root(children, cfg, rng),
            Self::Gumbel(p) => p.prepare_root(children, cfg, rng),
        }
    }

    fn select_child(
        &self,
        parent_visit: u32,
        parent_to_play: u8,
        children: &[ChildStat],
        stats: &crate::rl::mcts::MinMaxStats,
        cfg: &MctsConfig,
    ) -> usize {
        match self {
            Self::Puct(p) => p.select_child(parent_visit, parent_to_play, children, stats, cfg),
            Self::Gumbel(p) => p.select_child(parent_visit, parent_to_play, children, stats, cfg),
        }
    }

    fn recommend(&self, children: &[ChildStat], cfg: &MctsConfig, rng: &mut dyn RngCore) -> usize {
        match self {
            Self::Puct(p) => p.recommend(children, cfg, rng),
            Self::Gumbel(p) => p.recommend(children, cfg, rng),
        }
    }

    fn make_targets(&self, children: &[ChildStat], cfg: &MctsConfig) -> Vec<f32> {
        match self {
            Self::Puct(p) => p.make_targets(children, cfg),
            Self::Gumbel(p) => p.make_targets(children, cfg),
        }
    }

    fn make_root_scheduler(
        &self,
        num_root_children: usize,
        cfg: &MctsConfig,
    ) -> Box<dyn RootScheduler> {
        match self {
            Self::Puct(p) => p.make_root_scheduler(num_root_children, cfg),
            Self::Gumbel(p) => p.make_root_scheduler(num_root_children, cfg),
        }
    }
}
