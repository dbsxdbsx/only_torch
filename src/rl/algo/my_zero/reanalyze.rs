//! MyZero Reanalyze：用最新网络对旧轨迹重跑 MCTS，刷新 policy/value 目标。
//!
//! revisit 旧的 self-play 时间步，用**最新**模型参数重新执行 MCTS，得到更优质的 policy 与
//! value 目标——旧数据被反复榨取出新鲜监督信号（样本效率的核心机制之一）。
//!
//! - **只刷 policy / value**：`reward` / `terminated` 是环境事实，不变。
//! - **算力换样本效率**：每个被 reanalyze 的位置 = 一整棵 MCTS；由调用方控制比例 / 频率。

use rand::RngCore;

use crate::rl::SelfPlayGame;
use crate::rl::mcts::{MctsConfig, MctsModel, SearchPolicy, mcts_search};

/// 用当前模型对一局 self-play **原地**刷新每步的 `policy_target` 与 `root_value`。
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
    for step in game.steps.iter_mut() {
        let result = mcts_search(model, policy, &step.obs, cfg, rng);
        if result.children.is_empty() {
            continue; // 终局 / 空候选：保留原 target
        }
        let root_value = result.root_value();
        step.policy_target = result.learn_policy;
        step.root_value = Some(root_value);
    }
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
}
