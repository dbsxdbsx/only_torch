//! MyZero Reanalyze：用最新网络对旧轨迹重跑 MCTS，刷新 policy/value 目标。
//!
//! MuZero / EfficientZero 语义：**按训练 position**（unroll 窗口内各步）重搜，而非整局每步。
//!
//! - **只刷 policy / value**：`reward` / `terminated` 是环境事实，不变。
//! - **policy target** 与 self-play 共用 [`mcts_policy_target`](super::target::mcts_policy_target)（visit 或 completedQ）。
//! - **训练路径**（`Components.reanalyze = true`）：`sample_indexed` clone → unroll 窗口 reanalyze
//!   → `train_batch` → [`writeback_reanalyzed_samples`](super::runner::writeback_reanalyzed_samples)。
//! - **CartPole**：recipe 暂不 promote（实测学习失效，见 `.issue/items/my_zero_reanalyze_cartpole_regression.md`）。

use rand::RngCore;

use super::target::mcts_policy_target;
use crate::rl::SelfPlayGame;
use crate::rl::SelfPlayStep;
use crate::rl::mcts::{MctsConfig, MctsModel, SearchPolicy, mcts_search};

/// 用当前模型对单步 **原地**刷新 `policy_target` 与 `root_value`。
pub fn reanalyze_step<M, P>(
    model: &M,
    policy: &P,
    step: &mut SelfPlayStep,
    cfg: &MctsConfig,
    cq: Option<(f32, f32)>,
    action_dim: usize,
    rng: &mut dyn RngCore,
) where
    M: MctsModel,
    P: SearchPolicy,
{
    // 仅向量 obs 路径（图像模式的 reanalyze 在 runner 入口拒绝，Phase 3 接回时需堆叠组装）
    let result = mcts_search(model, policy, step.obs.as_f32(), cfg, rng);
    if result.children.is_empty() {
        return;
    }
    step.policy_target = mcts_policy_target(&result, cq, action_dim);
    step.root_value = Some(result.root_value());
}

/// 刷新 unroll 窗口 `[start, start + unroll_k]` 内各步标签（position 级 reanalyze）。
pub fn reanalyze_unroll_window<M, P>(
    model: &M,
    policy: &P,
    steps: &mut [SelfPlayStep],
    start: usize,
    unroll_k: usize,
    cfg: &MctsConfig,
    cq: Option<(f32, f32)>,
    action_dim: usize,
    rng: &mut dyn RngCore,
) where
    M: MctsModel,
    P: SearchPolicy,
{
    if steps.is_empty() || start >= steps.len() {
        return;
    }
    let end = (start + unroll_k).min(steps.len() - 1);
    for step in &mut steps[start..=end] {
        reanalyze_step(model, policy, step, cfg, cq, action_dim, rng);
    }
}

/// 用当前模型对一局 self-play **整局**刷新（测试 / 调试；训练路径用 [`reanalyze_unroll_window`]）。
pub fn reanalyze_game<M, P>(
    model: &M,
    policy: &P,
    game: &mut SelfPlayGame,
    cfg: &MctsConfig,
    cq: Option<(f32, f32)>,
    action_dim: usize,
    rng: &mut dyn RngCore,
) where
    M: MctsModel,
    P: SearchPolicy,
{
    let len = game.steps.len();
    if len == 0 {
        return;
    }
    reanalyze_unroll_window(
        model,
        policy,
        &mut game.steps,
        0,
        len - 1,
        cfg,
        cq,
        action_dim,
        rng,
    );
}
