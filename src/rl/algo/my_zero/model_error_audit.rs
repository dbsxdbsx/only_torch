//! Gomoku Phase 3A0：误差 proxy 的任务相关性审计。
//!
//! 审计 episode 始终从真实环境 `reset()` 开始，行为由 learned Task MCTS 决定；
//! Reference Transition Model 只生成诊断标签，不生成动作、训练 target 或 replay。

use super::board::BoardMctsModel;
use super::component::Components;
use super::gomoku::{RulesBoard, TrueRulesBoardModel};
use super::model_error::{
    ModelErrorComponents, jensen_shannon_divergence, spearman_rank_correlation,
    top_fraction_event_lift,
};
use super::network::MyZeroModel;
use super::search_policy::MyZeroSearchPolicy;
use crate::rl::mcts::{ActionId, ChildStat, MctsConfig, SearchResult, mcts_search};
use crate::rl::{GameOutcome, GymEnv, SelfPlayGame, SelfPlayStep};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// 由冻结 behavior model 从真实 reset 采得、可被后续 model revision 重复评分的数据块。
#[derive(Debug, Clone)]
pub(crate) struct BoardAuditTransition {
    pub obs: Vec<f32>,
    pub action: ActionId,
    pub reward: f32,
    pub continuation: f32,
    pub next_obs: Option<Vec<f32>>,
    pub next_legal_mask: Option<Vec<bool>>,
    pub reference_policy_jsd: f32,
    pub tactical_position: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct BoardAuditBlock {
    pub transitions: Vec<BoardAuditTransition>,
    pub games: Vec<SelfPlayGame>,
}

/// 单条真实 transition 的 proxy 与 reference-only 任务标签。
#[derive(Debug, Clone)]
pub(crate) struct BoardProxyAuditRecord {
    pub error: ModelErrorComponents,
    /// learned-tree 与 true-rules-tree 的根策略 JSD（同一 `h/f`，只替换树内 transition）。
    pub reference_policy_jsd: f32,
    pub tactical_position: bool,
}

/// 一个误差分量对任务诊断标签的关联摘要。
#[derive(Debug, Clone)]
pub(crate) struct ProxyComponentAudit {
    pub name: &'static str,
    pub count: usize,
    pub mean: f32,
    pub spearman_to_reference_policy: Option<f32>,
    pub tactical_position_top_decile_lift: Option<f32>,
}

/// 冻结 model revision 上若干未来真实 episode 的审计结果。
#[derive(Debug, Clone)]
pub(crate) struct BoardProxyAuditReport {
    pub records: Vec<BoardProxyAuditRecord>,
    pub components: Vec<ProxyComponentAudit>,
    pub tactical_positions: usize,
}

impl BoardProxyAuditReport {
    fn from_records(records: Vec<BoardProxyAuditRecord>) -> Self {
        let tactical_positions = records
            .iter()
            .filter(|record| record.tactical_position)
            .count();
        let components = [
            (
                "reward_kl",
                records
                    .iter()
                    .map(|record| Some(record.error.reward_kl))
                    .collect::<Vec<Option<f32>>>(),
            ),
            (
                "continuation_brier",
                records
                    .iter()
                    .map(|record| Some(record.error.continuation_brier))
                    .collect::<Vec<Option<f32>>>(),
            ),
            (
                "policy_jsd",
                records
                    .iter()
                    .map(|record| record.error.policy_jsd)
                    .collect::<Vec<Option<f32>>>(),
            ),
            (
                "value_abs_diff",
                records
                    .iter()
                    .map(|record| record.error.value_abs_diff)
                    .collect::<Vec<Option<f32>>>(),
            ),
        ]
        .into_iter()
        .map(|(name, scores)| summarize_component(name, &scores, &records))
        .collect();

        Self {
            records,
            components,
            tactical_positions,
        }
    }
}

/// 在冻结模型上运行未来真实 episode；函数内部不训练、不写 replay。
pub(crate) fn audit_gomoku_proxy(
    env: &GymEnv,
    model: &MyZeroModel,
    components: &Components,
    num_simulations: u32,
    n_episodes: usize,
    seed: u64,
) -> BoardProxyAuditReport {
    let block =
        collect_gomoku_audit_block(env, model, components, num_simulations, n_episodes, seed);
    score_gomoku_audit_block(model, &block)
}

/// 用冻结 behavior model 采集真实 transition 与 reference-only 标签，不计算 proxy。
pub(crate) fn collect_gomoku_audit_block(
    env: &GymEnv,
    behavior_model: &MyZeroModel,
    components: &Components,
    num_simulations: u32,
    n_episodes: usize,
    seed: u64,
) -> Vec<BoardAuditTransition> {
    collect_gomoku_audit_data(
        env,
        behavior_model,
        components,
        num_simulations,
        n_episodes,
        seed,
    )
    .transitions
}

/// 同时保留原 MuZero 训练所需的完整真实 game；仅供 3A0 reducibility 干预。
pub(crate) fn collect_gomoku_audit_data(
    env: &GymEnv,
    behavior_model: &MyZeroModel,
    components: &Components,
    num_simulations: u32,
    n_episodes: usize,
    seed: u64,
) -> BoardAuditBlock {
    let action_dim = behavior_model.action_dim;
    let side = (action_dim as f32).sqrt() as usize;
    assert_eq!(side * side, action_dim, "Gomoku 动作数应为平方数");
    let behavior_cfg = MctsConfig {
        num_simulations,
        temperature: 1.0,
        discount: 1.0,
        root_dirichlet_alpha: 0.15,
        root_exploration_fraction: MctsConfig::default().root_exploration_fraction,
        ..MctsConfig::default()
    };
    let diagnostic_cfg = MctsConfig {
        num_simulations,
        temperature: 0.0,
        discount: 1.0,
        root_dirichlet_alpha: 0.15,
        root_exploration_fraction: 0.0,
        ..MctsConfig::default()
    };
    let policy = MyZeroSearchPolicy::from_components(components);
    let mut transitions = Vec::new();
    let mut games = Vec::with_capacity(n_episodes);

    for episode in 0..n_episodes {
        let reset_seed = seed.wrapping_add(episode as u64);
        env.reset(Some(reset_seed));
        let mut position = 0_u64;
        let mut steps = Vec::new();
        loop {
            let obs = env.board_observation_flat();
            let player = env.current_player();
            let legal = env.legal_mask();
            let learned_model =
                BoardMctsModel::new(behavior_model, legal.clone(), player, action_dim);
            let reference_model = TrueRulesBoardModel::new(behavior_model, &obs, player, side);
            let search_seed = reset_seed.wrapping_mul(1_000_003).wrapping_add(position);
            let mut behavior_rng = StdRng::seed_from_u64(search_seed ^ 0xBEA0_1001);
            let mut learned_rng = StdRng::seed_from_u64(search_seed ^ 0x1EA2_3EED);
            let mut reference_rng = StdRng::seed_from_u64(search_seed ^ 0x7EFA_11CE);
            let behavior = mcts_search(
                &learned_model,
                &policy,
                &obs,
                &behavior_cfg,
                &mut behavior_rng,
            );
            let learned = mcts_search(
                &learned_model,
                &policy,
                &obs,
                &diagnostic_cfg,
                &mut learned_rng,
            );
            let reference = mcts_search(
                &reference_model,
                &policy,
                &obs,
                &diagnostic_cfg,
                &mut reference_rng,
            );
            let action = behavior.recommended_id.index();
            assert!(legal[action], "Task MCTS 实际动作必须合法");

            let learned_policy = visit_policy(&learned, action_dim);
            let reference_policy = visit_policy(&reference, action_dim);
            let reference_policy_jsd =
                jensen_shannon_divergence(&learned_policy, &reference_policy, Some(&legal))
                    .unwrap_or(0.0);
            let behavior_policy = visit_policy(&behavior, action_dim);
            let root_value = negamax_root_value(&behavior.children, player);

            let mut rules = RulesBoard::from_obs(&obs, player, side);
            let current_has_win = rules.has_win_in_1(player);
            let opponent_has_win = rules.has_win_in_1(1 - player);
            let (rules_reward, rules_terminal) = rules.step(action);

            let (reward, terminal) = env.board_step(action);
            debug_assert_eq!(
                (reward, terminal),
                (rules_reward, rules_terminal),
                "reference-only RulesBoard 必须与真实 env step 一致"
            );
            let (next_obs, next_legal) = if terminal {
                (None, None)
            } else {
                (Some(env.board_observation_flat()), Some(env.legal_mask()))
            };
            transitions.push(BoardAuditTransition {
                obs: obs.clone(),
                action: ActionId(action),
                reward,
                continuation: if terminal { 0.0 } else { 1.0 },
                next_obs,
                next_legal_mask: next_legal,
                reference_policy_jsd,
                tactical_position: current_has_win || opponent_has_win,
            });
            steps.push(SelfPlayStep {
                obs: obs.into(),
                action: vec![action as f32],
                policy_target: behavior_policy,
                player,
                reward,
                root_value: Some(root_value),
                terminated: terminal,
                truncated: false,
                continuation: if terminal { 0.0 } else { 1.0 },
                extras: Default::default(),
            });
            if terminal {
                let outcome = if reward > 0.5 {
                    GameOutcome::Win(player)
                } else {
                    GameOutcome::Draw
                };
                games.push(SelfPlayGame { steps, outcome });
                break;
            }
            position += 1;
        }
    }

    BoardAuditBlock { transitions, games }
}

/// 用任意冻结 model revision 给同一真实 transition block 重复评分。
pub(crate) fn score_gomoku_audit_block(
    model: &MyZeroModel,
    block: &[BoardAuditTransition],
) -> BoardProxyAuditReport {
    let records = block
        .iter()
        .map(|transition| {
            let error = model.transition_error_components(
                &transition.obs,
                transition.action,
                transition.reward,
                transition.continuation,
                transition.next_obs.as_deref(),
                transition.next_legal_mask.as_deref(),
            );
            debug_assert!(error.is_finite_nonnegative(), "error={error:?}");
            BoardProxyAuditRecord {
                error,
                reference_policy_jsd: transition.reference_policy_jsd,
                tactical_position: transition.tactical_position,
            }
        })
        .collect();
    BoardProxyAuditReport::from_records(records)
}

fn negamax_root_value(children: &[ChildStat], root_player: u8) -> f32 {
    let total_visits: u32 = children.iter().map(|child| child.visit_count).sum();
    if total_visits == 0 {
        return 0.0;
    }
    children
        .iter()
        .filter_map(|child| child_q(child, root_player).map(|q| q * child.visit_count as f32))
        .sum::<f32>()
        / total_visits as f32
}

fn visit_policy(result: &SearchResult, action_dim: usize) -> Vec<f32> {
    let mut policy = vec![0.0; action_dim];
    let total: u32 = result.children.iter().map(|child| child.visit_count).sum();
    if total == 0 {
        let candidates = result.children.len().max(1) as f32;
        for child in &result.children {
            policy[child.action_id.index()] = 1.0 / candidates;
        }
        return policy;
    }
    for child in &result.children {
        policy[child.action_id.index()] = child.visit_count as f32 / total as f32;
    }
    policy
}

fn child_q(child: &ChildStat, root_player: u8) -> Option<f32> {
    (child.visit_count > 0).then(|| {
        let child_value = child.value_sum / child.visit_count as f32;
        let perspective = if child.to_play == root_player {
            1.0
        } else {
            -1.0
        };
        child.reward + child.discount * perspective * child_value
    })
}

fn summarize_component(
    name: &'static str,
    scores: &[Option<f32>],
    records: &[BoardProxyAuditRecord],
) -> ProxyComponentAudit {
    let available: Vec<(usize, f32)> = scores
        .iter()
        .enumerate()
        .filter_map(|(index, score)| score.map(|score| (index, score)))
        .filter(|(_, score)| score.is_finite())
        .collect();
    let values: Vec<f32> = available.iter().map(|(_, score)| *score).collect();
    let reference_policy: Vec<f32> = available
        .iter()
        .map(|(index, _)| records[*index].reference_policy_jsd)
        .collect();
    let tactical_events: Vec<bool> = available
        .iter()
        .map(|(index, _)| records[*index].tactical_position)
        .collect();
    ProxyComponentAudit {
        name,
        count: values.len(),
        mean: if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f32>() / values.len() as f32
        },
        spearman_to_reference_policy: spearman_rank_correlation(&values, &reference_policy),
        tactical_position_top_decile_lift: top_fraction_event_lift(&values, &tactical_events, 0.1),
    }
}
