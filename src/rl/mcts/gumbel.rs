//! Gumbel MuZero 根搜索（Danihelka et al. 2022 · 标准版，非 Full）。
//!
//! - 根：Gumbel-Top-k + Sequential Halving（Algorithm 2）
//! - 非根：仍 PUCT（[`PuctPolicy`]）
//! - 根出动作：`final_recommendation` 幸存者，非 visit 采样
//! - 训练 target：由上层 `mcts_policy_target` 接 completedQ

use rand::RngCore;

use super::min_max::MinMaxStats;
use super::puct::PuctPolicy;
use super::traits::{RootScheduler, RootStrategy, SelectionRule, TargetRule};
use super::types::{ChildStat, MctsConfig};

/// Gumbel MuZero 搜索策略（标准版：仅改根，非根 PUCT）。
#[derive(Debug, Clone)]
pub struct GumbelPolicy {
    puct: PuctPolicy,
    /// Gumbel-Top-k 上限 `m`（论文默认 16）
    pub max_considered_actions: usize,
    /// Sequential Halving 打分 `σ` 的 `c_visit`（论文默认 50）
    pub c_visit: f32,
    /// Sequential Halving 打分 `σ` 的 `c_scale`
    pub c_scale: f32,
}

impl Default for GumbelPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl GumbelPolicy {
    pub fn new() -> Self {
        Self {
            puct: PuctPolicy::new(),
            max_considered_actions: 16,
            c_visit: 50.0,
            c_scale: 0.1,
        }
    }

    pub fn with_halving_scale(mut self, c_visit: f32, c_scale: f32) -> Self {
        self.c_visit = c_visit;
        self.c_scale = c_scale;
        self
    }
}

impl RootStrategy for GumbelPolicy {
    /// Gumbel 根不用 Dirichlet；探索由 Gumbel 噪声 + Sequential Halving 承担。
    fn prepare_root(&self, _children: &mut [ChildStat], _cfg: &MctsConfig, _rng: &mut dyn RngCore) {
    }

    fn make_root_scheduler(
        &self,
        _num_root_children: usize,
        _cfg: &MctsConfig,
    ) -> Box<dyn RootScheduler> {
        Box::new(GumbelRootScheduler::new(
            self.max_considered_actions,
            self.c_visit,
            self.c_scale,
        ))
    }
}

impl SelectionRule for GumbelPolicy {
    fn select_child(
        &self,
        parent_visit: u32,
        parent_to_play: u8,
        children: &[ChildStat],
        stats: &MinMaxStats,
        cfg: &MctsConfig,
    ) -> usize {
        self.puct
            .select_child(parent_visit, parent_to_play, children, stats, cfg)
    }
}

impl TargetRule for GumbelPolicy {
    fn recommend(&self, children: &[ChildStat], cfg: &MctsConfig, rng: &mut dyn RngCore) -> usize {
        self.puct.recommend(children, cfg, rng)
    }

    fn make_targets(&self, children: &[ChildStat], cfg: &MctsConfig) -> Vec<f32> {
        self.puct.make_targets(children, cfg)
    }
}

/// Sequential Halving + Gumbel 根调度器（有状态）。
struct GumbelRootScheduler {
    max_considered: usize,
    c_visit: f32,
    c_scale: f32,
    gumbel: Vec<f32>,
    active: Vec<usize>,
    v_pi: f32,
    num_phases: usize,
    phase: usize,
    phase_budget: usize,
    sims_in_phase: usize,
    round_robin: usize,
    total_sims: u32,
}

impl GumbelRootScheduler {
    fn new(max_considered: usize, c_visit: f32, c_scale: f32) -> Self {
        Self {
            max_considered,
            c_visit,
            c_scale,
            gumbel: Vec::new(),
            active: Vec::new(),
            v_pi: 0.0,
            num_phases: 0,
            phase: 0,
            phase_budget: 0,
            sims_in_phase: 0,
            round_robin: 0,
            total_sims: 0,
        }
    }

    fn init(
        &mut self,
        root_children: &[ChildStat],
        network_value: f32,
        cfg: &MctsConfig,
        rng: &mut dyn RngCore,
    ) {
        let k = root_children.len();
        if k == 0 {
            return;
        }
        self.v_pi = network_value;
        self.total_sims = cfg.num_simulations.max(1);
        self.gumbel = (0..k).map(|_| sample_gumbel(rng)).collect();

        let m = k
            .min(self.max_considered)
            .min(self.total_sims as usize)
            .max(1);
        let mut ranked: Vec<(f32, usize)> = (0..k)
            .map(|i| {
                let logit = root_children[i].prior.max(1e-12).ln();
                (self.gumbel[i] + logit, i)
            })
            .collect();
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        self.active = ranked.into_iter().take(m).map(|(_, i)| i).collect();

        self.num_phases = halving_phases(self.active.len());
        self.phase = 0;
        self.phase_budget = phase_budget(self.total_sims, self.num_phases);
        self.sims_in_phase = 0;
        self.round_robin = 0;
    }

    fn end_phase(&mut self, root_children: &[ChildStat]) {
        if self.active.len() <= 1 {
            return;
        }
        let mut scored: Vec<(f32, usize)> = self
            .active
            .iter()
            .map(|&i| {
                (
                    gumbel_halving_score(
                        root_children,
                        i,
                        &self.gumbel,
                        self.v_pi,
                        self.c_visit,
                        self.c_scale,
                    ),
                    i,
                )
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let keep = (self.active.len() / 2).max(1);
        self.active = scored.into_iter().take(keep).map(|(_, i)| i).collect();
        self.phase += 1;
        self.sims_in_phase = 0;
        self.round_robin = 0;
        let remaining_phases = self.num_phases.saturating_sub(self.phase).max(1);
        let sims_done = self.phase.saturating_mul(self.phase_budget);
        let sims_left = self
            .total_sims
            .saturating_sub(sims_done.min(self.total_sims as usize) as u32)
            as usize;
        self.phase_budget = (sims_left / remaining_phases).max(1);
    }
}

impl RootScheduler for GumbelRootScheduler {
    fn is_active(&self) -> bool {
        true
    }

    fn on_search_start(
        &mut self,
        root_children: &[ChildStat],
        network_value: f32,
        cfg: &MctsConfig,
        rng: &mut dyn RngCore,
    ) {
        self.init(root_children, network_value, cfg, rng);
    }

    fn next_root_child(
        &mut self,
        root_children: &[ChildStat],
        sim_idx: usize,
        cfg: &MctsConfig,
    ) -> Option<usize> {
        if root_children.is_empty() || self.active.is_empty() {
            return None;
        }
        if self.sims_in_phase >= self.phase_budget
            && self.active.len() > 1
            && self.phase + 1 < self.num_phases
        {
            self.end_phase(root_children);
        }
        let idx = self.active[self.round_robin % self.active.len()];
        self.round_robin += 1;
        self.sims_in_phase += 1;
        let _ = (sim_idx, cfg);
        Some(idx)
    }

    fn final_recommendation(&self, root_children: &[ChildStat]) -> Option<usize> {
        if self.active.is_empty() || root_children.is_empty() {
            return None;
        }
        self.active
            .iter()
            .max_by(|&&a, &&b| {
                let sa = gumbel_halving_score(
                    root_children,
                    a,
                    &self.gumbel,
                    self.v_pi,
                    self.c_visit,
                    self.c_scale,
                );
                let sb = gumbel_halving_score(
                    root_children,
                    b,
                    &self.gumbel,
                    self.v_pi,
                    self.c_visit,
                    self.c_scale,
                );
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    }
}

/// Gumbel(0) 采样：`g = -ln(-ln U)`，`U ~ Uniform(0,1)`。
fn sample_gumbel(rng: &mut dyn RngCore) -> f32 {
    loop {
        let u = rand_uniform(rng);
        if u > 0.0 && u < 1.0 {
            return -(-u.ln()).ln();
        }
    }
}

fn rand_uniform(rng: &mut dyn RngCore) -> f32 {
    (rng.next_u32() as f32 + 1.0) / (u32::MAX as f32 + 1.0)
}

fn halving_phases(m: usize) -> usize {
    if m <= 1 {
        1
    } else {
        (m as f32).log2().ceil() as usize
    }
}

fn phase_budget(total_sims: u32, num_phases: usize) -> usize {
    (total_sims as usize / num_phases.max(1)).max(1)
}

/// 根 Sequential Halving 打分：`g(a) + ln π(a) + σ(q̂(a))`（论文 Algorithm 2）。
fn gumbel_halving_score(
    children: &[ChildStat],
    idx: usize,
    gumbel: &[f32],
    v_pi: f32,
    c_visit: f32,
    c_scale: f32,
) -> f32 {
    let c = &children[idx];
    let logit = c.prior.max(1e-12).ln();
    let q_hat = if c.visit_count > 0 {
        let child_v = c.value_sum / c.visit_count as f32;
        c.reward + c.discount * child_v
    } else {
        v_pi
    };
    let (lo, hi) = q_range(children, v_pi);
    let range = (hi - lo).max(1e-8);
    let norm_q = (q_hat - lo) / range;
    let max_n = children.iter().map(|x| x.visit_count).max().unwrap_or(0) as f32;
    let sigma = (c_visit + max_n) * c_scale * norm_q;
    gumbel[idx] + logit + sigma
}

fn q_range(children: &[ChildStat], v_pi: f32) -> (f32, f32) {
    let mut lo = v_pi;
    let mut hi = v_pi;
    for c in children {
        let q = if c.visit_count > 0 {
            let child_v = c.value_sum / c.visit_count as f32;
            c.reward + c.discount * child_v
        } else {
            v_pi
        };
        lo = lo.min(q);
        hi = hi.max(q);
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::mcts::types::ActionPayload;
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
}
