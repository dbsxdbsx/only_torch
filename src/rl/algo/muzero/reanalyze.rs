//! MuZero Reanalyze：用最新网络对旧轨迹重跑 MCTS，刷新 policy/value 目标。
//!
//! 对齐 canonical MuZero（Schrittwieser et al., 2020）的 Reanalyze：revisit 旧的 self-play
//! 时间步，用**最新**模型参数重新执行 MCTS，得到更优质的 policy 与 value 目标。这是
//! MuZero 样本效率的核心机制——旧数据被反复榨取出新鲜监督信号。
//!
//! # 设计要点
//! - **用最新网络**（非冻结 target network）：忠于 MuZero Reanalyze 原文。独立 target
//!   network 是 DQN / EfficientZero 的稳定性增强，不属 base MuZero，留 EZ 阶段。
//! - **只刷 policy / value**：`reward` / `terminated` 是环境事实，不随网络变，保持不动。
//! - **算力换样本效率**：每个被 reanalyze 的位置 = 一整棵 MCTS；CPU only 下开销显著，
//!   故由调用方控制 reanalyze 的比例 / 频率（见示例训练循环），库不强制全量。

use crate::rl::SelfPlayGame;
use crate::rl::mcts::{MctsConfig, MctsModel, SearchPolicy, mcts_search};

/// 用当前模型对一局 self-play **原地**刷新每步的 `policy_target` 与 `root_value`。
///
/// - `reward` / `terminated` 不变（环境事实）。
/// - 终局 / 空候选位置（搜索无子节点）保留原 target，避免写入退化值。
/// - `model` 应携带最新参数（如 `DynamicsModel::new(latest_model, ...)`）；`cfg.discount`
///   与 self-play 保持一致，使 `root_value` 口径统一（见 [`SearchResult::root_value`]）。
///
/// [`SearchResult::root_value`]: crate::rl::mcts::SearchResult::root_value
pub fn reanalyze_game<M, P>(model: &M, policy: &P, game: &mut SelfPlayGame, cfg: &MctsConfig)
where
    M: MctsModel,
    P: SearchPolicy,
{
    for step in game.steps.iter_mut() {
        let result = mcts_search(model, policy, &step.obs, cfg);
        if result.children.is_empty() {
            continue; // 终局 / 空候选：保留原 target
        }
        // 先取 root_value（借用），再 move learn_policy
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
            policy_target: vec![1.0, 0.0], // 故意失衡，reanalyze 后应被刷新成搜索分布
            player: 0,
            reward: 1.0,
            root_value: Some(99.0), // 故意离谱占位，reanalyze 后应被重算
            terminated: false,
        }
    }

    /// reanalyze 应刷新 policy/value 目标，但保持 reward/terminated（环境事实）不变。
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

        reanalyze_game(&model, &policy, &mut game, &cfg);

        for step in &game.steps {
            // policy_target 刷新为合法概率分布（长度 2、和 ≈ 1）
            assert_eq!(step.policy_target.len(), 2);
            let sum: f32 = step.policy_target.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "policy_target 应为概率分布, sum={sum}"
            );
            // root_value 被重算（不再是 99.0 占位；mock 下 Q≈1+0.99·0.5 远小于 50）
            let rv = step.root_value.expect("root_value 应被刷新");
            assert!(rv < 50.0, "root_value 应被重算, got {rv}");
            // 环境事实保持不变
            assert_eq!(step.reward, 1.0);
            assert!(!step.terminated);
        }
    }
}
