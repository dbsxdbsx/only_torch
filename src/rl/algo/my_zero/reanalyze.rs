//! MyZero Reanalyze：用最新网络对旧轨迹重跑 MCTS，刷新 policy/value 目标。
//!
//! MuZero / EfficientZero 语义：**按训练 position**（unroll 窗口内各步）重搜，而非整局每步。
//!
//! - **只刷 policy / value**：`reward` / `terminated` 是环境事实，不变。
//! - **训练路径**（`Components.reanalyze = true`）：`sample_indexed` clone → unroll 窗口 reanalyze
//!   → `train_batch` → [`writeback_reanalyzed_samples`](super::runner::writeback_reanalyzed_samples)。
//! - **CartPole**：recipe 暂不 promote（实测学习失效，见 `.issue/items/my_zero_reanalyze_cartpole_regression.md`）。

use rand::RngCore;

use crate::rl::SelfPlayGame;
use crate::rl::SelfPlayStep;
use crate::rl::mcts::{MctsConfig, MctsModel, SearchPolicy, mcts_search};

/// 用当前模型对单步 **原地**刷新 `policy_target` 与 `root_value`。
pub fn reanalyze_step<M, P>(
    model: &M,
    policy: &P,
    step: &mut SelfPlayStep,
    cfg: &MctsConfig,
    rng: &mut dyn RngCore,
) where
    M: MctsModel,
    P: SearchPolicy,
{
    let result = mcts_search(model, policy, &step.obs, cfg, rng);
    if result.children.is_empty() {
        return;
    }
    let root_value = result.root_value();
    step.policy_target = result.learn_policy;
    step.root_value = Some(root_value);
}

/// 刷新 unroll 窗口 `[start, start + unroll_k]` 内各步标签（position 级 reanalyze）。
pub fn reanalyze_unroll_window<M, P>(
    model: &M,
    policy: &P,
    steps: &mut [SelfPlayStep],
    start: usize,
    unroll_k: usize,
    cfg: &MctsConfig,
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
        reanalyze_step(model, policy, step, cfg, rng);
    }
}

/// 用当前模型对一局 self-play **整局**刷新（测试 / 调试；训练路径用 [`reanalyze_unroll_window`]）。
pub fn reanalyze_game<M, P>(
    model: &M,
    policy: &P,
    game: &mut SelfPlayGame,
    cfg: &MctsConfig,
    rng: &mut dyn RngCore,
) where
    M: MctsModel,
    P: SearchPolicy,
{
    let len = game.steps.len();
    if len == 0 {
        return;
    }
    reanalyze_unroll_window(model, policy, &mut game.steps, 0, len - 1, cfg, rng);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::mcts::{ActionPayload, PuctPolicy, RecurrentOut, RootOut};
    use crate::rl::{GameOutcome, SelfPlayStep};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// 极简确定性 MctsModel：2 个离散动作、固定 prior/value、不终止。
    struct MockModel;

    impl MctsModel for MockModel {
        type State = Vec<f32>;
        fn root(&self, obs: &[f32]) -> RootOut<Self::State> {
            RootOut {
                state: obs.to_vec(),
                prior: vec![0.5, 0.5],
                value: 0.0,
                candidate_actions: vec![ActionPayload::Discrete(0), ActionPayload::Discrete(1)],
                to_play: 0,
            }
        }
        fn recurrent(
            &self,
            state: &Self::State,
            _action: &ActionPayload,
        ) -> RecurrentOut<Self::State> {
            RecurrentOut {
                state: state.clone(),
                reward: 1.0,
                value: 0.5,
                prior: vec![0.5, 0.5],
                candidate_actions: vec![ActionPayload::Discrete(0), ActionPayload::Discrete(1)],
                terminal: false,
                to_play: 0,
                discount: 0.99,
            }
        }
    }

    fn make_step(obs: Vec<f32>) -> SelfPlayStep {
        SelfPlayStep {
            obs,
            action: vec![0.0],
            policy_target: vec![1.0, 0.0],
            player: 0,
            reward: 1.0,
            root_value: Some(99.0),
            terminated: false,
            extras: Default::default(),
        }
    }

    #[test]
    fn reanalyze_refreshes_targets_preserves_facts() {
        let model = MockModel;
        let policy = PuctPolicy::new();
        let cfg = MctsConfig {
            num_simulations: 16,
            root_exploration_fraction: 0.0,
            ..MctsConfig::default()
        };

        let mut game = SelfPlayGame {
            steps: vec![make_step(vec![0.0; 4]), make_step(vec![1.0; 4])],
            outcome: GameOutcome::InProgress,
        };

        let mut rng = StdRng::seed_from_u64(7);
        reanalyze_game(&model, &policy, &mut game, &cfg, &mut rng);

        for step in &game.steps {
            assert_eq!(step.policy_target.len(), 2);
            let sum: f32 = step.policy_target.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "policy_target 应为概率分布, sum={sum}"
            );
            let rv = step.root_value.expect("root_value 应被刷新");
            assert!(rv < 50.0, "root_value 应被重算, got {rv}");
            assert_eq!(step.reward, 1.0);
            assert!(!step.terminated);
        }
    }

    #[test]
    fn unroll_window_only_touches_requested_range() {
        let model = MockModel;
        let policy = PuctPolicy::new();
        let cfg = MctsConfig {
            num_simulations: 8,
            root_exploration_fraction: 0.0,
            ..MctsConfig::default()
        };
        let mut steps = vec![
            make_step(vec![0.0; 4]),
            make_step(vec![1.0; 4]),
            make_step(vec![2.0; 4]),
        ];
        steps[0].root_value = Some(99.0);
        steps[2].root_value = Some(99.0);

        let mut rng = StdRng::seed_from_u64(1);
        reanalyze_unroll_window(&model, &policy, &mut steps, 1, 0, &cfg, &mut rng);

        assert!(
            steps[0].root_value == Some(99.0),
            "窗口外 step 0 不应被刷新"
        );
        assert!(steps[1].root_value.expect("step 1") < 50.0);
        assert!(
            steps[2].root_value == Some(99.0),
            "窗口外 step 2 不应被刷新"
        );
    }
}
