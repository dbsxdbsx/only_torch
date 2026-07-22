//! MCTS 主搜索循环
//!
//! Chance 合并到 action edge：展开叶子时立即按 chance prior 采样 outcome，
//! 树中仅保留 Decision 节点。多次模拟中同一 action 会被采样到不同 outcome，
//! Q 自然对分布取均值。`num_chance_outcomes == 1` 时走确定性快路径，行为不变。

use rand::{Rng, RngCore};

use super::min_max::MinMaxStats;
use super::node::{Node, NodeKind, Tree};
use super::sampled::{sample_for_expansion, sample_root_for_expansion};
use super::traits::{CandidateProvider, MctsModel, SearchPolicy};
use super::types::{ActionCandidate, CandidateSet, ChildStat, MctsConfig, SearchResult};

/// 执行 MCTS 搜索
///
/// `cfg.num_chance_outcomes > 1` 时启用 stochastic 路径：展开叶子时按 chance prior
/// 采样 outcome，树中仅保留 Decision 节点。K=1 时走确定性快路径。
pub fn mcts_search<M: MctsModel, P: SearchPolicy>(
    model: &M,
    policy: &P,
    obs: &[f32],
    cfg: &MctsConfig,
    rng: &mut dyn RngCore,
) -> SearchResult {
    // 1. root 推理 → 建根节点 + 展开子节点
    let root_out = {
        crate::prof_scope!("mcts.root_fwd");
        model.root(obs)
    };

    if root_out.candidates.is_empty() {
        return SearchResult {
            children: Vec::new(),
            recommended: super::types::ActionPayload::Discrete(0),
            recommended_id: super::types::ActionId(0),
            learn_policy: Vec::new(),
            network_value: root_out.value,
            q_range: None,
        };
    }

    let mut tree = Tree::new(root_out.state.clone(), root_out.to_play);
    let candidate_provider = ConfiguredCandidateProvider;
    let root_candidates =
        candidate_provider.expand_candidates(&root_out.candidates, cfg, true, rng);

    // 根子节点统一为 Decision（chance 合并到 action edge）
    expand_root_with_kind(
        &mut tree,
        &root_candidates.candidates,
        root_out.to_play,
        cfg.discount,
        NodeKind::Decision,
    );

    tree.nodes[tree.root].visit_count = 1;

    // 2. 注入根节点 Dirichlet 噪声
    let root_id = tree.root;
    let mut root_child_stats = collect_child_stats(&tree, root_id);
    policy.prepare_root(&mut root_child_stats, cfg, rng);
    apply_child_stats_to_tree(&mut tree, root_id, &root_child_stats);

    // 3. 模拟循环
    let mut min_max = MinMaxStats::new();
    let num_root_children = tree.nodes[tree.root].children.len();
    let mut scheduler = policy.make_root_scheduler(num_root_children, cfg);
    let use_scheduler = scheduler.is_active();
    if use_scheduler {
        scheduler.on_search_start(
            &root_child_stats,
            root_out.value,
            root_out.to_play,
            cfg,
            rng,
        );
    }

    let mut select_scratch: Vec<ChildStat> = Vec::new();

    for sim_idx in 0..cfg.num_simulations as usize {
        let forced_root = if use_scheduler {
            let root_children = collect_child_stats(&tree, tree.root);
            scheduler.next_root_child(&root_children, sim_idx, cfg, min_max.range())
        } else {
            None
        };

        let leaf_id = {
            crate::prof_scope!("mcts.select");
            select(
                &tree,
                policy,
                &min_max,
                cfg,
                forced_root,
                &mut select_scratch,
            )
        };

        if tree.nodes[leaf_id].terminal {
            crate::prof_scope!("mcts.backup");
            backup(&mut tree, leaf_id, 0.0, &mut min_max, cfg);
            continue;
        }

        let parent_id = tree.nodes[leaf_id].parent.unwrap_or(tree.root);
        let edge_idx = tree.nodes[leaf_id].action_from_parent.unwrap_or(0);
        let edge = &tree.nodes[parent_id].children[edge_idx];
        let action_id = edge.action_id;
        let action = edge.action.clone();

        let parent_state = tree.states[parent_id]
            .as_ref()
            .expect("parent state should exist");

        if model.num_chance_outcomes() <= 1 {
            // ── K=1 确定性快路径（与历史行为完全一致） ──
            let rec_out = {
                crate::prof_scope!("mcts.recurrent_fwd");
                model.recurrent(parent_state, action_id, &action)
            };
            apply_recurrent_to_leaf(
                &mut tree,
                &candidate_provider,
                leaf_id,
                &rec_out,
                cfg,
                rng,
            );
            let backup_value = if rec_out.terminal { 0.0 } else { rec_out.value };
            crate::prof_scope!("mcts.backup");
            backup(&mut tree, leaf_id, backup_value, &mut min_max, cfg);
        } else {
            // ── K>1：afterstate → 按 chance_prior 采样 outcome → next_state ──
            let dec_out = {
                crate::prof_scope!("mcts.decision_recurrent_fwd");
                model.decision_recurrent(parent_state, action_id, &action)
            };
            let chance_id = sample_chance_outcome(&dec_out.chance_prior, rng);
            let rec_out = {
                crate::prof_scope!("mcts.chance_recurrent_fwd");
                model.chance_recurrent(&dec_out.afterstate, chance_id)
            };
            apply_recurrent_to_leaf(
                &mut tree,
                &candidate_provider,
                leaf_id,
                &rec_out,
                cfg,
                rng,
            );
            let backup_value = if rec_out.terminal { 0.0 } else { rec_out.value };
            crate::prof_scope!("mcts.backup");
            backup(&mut tree, leaf_id, backup_value, &mut min_max, cfg);
        }
    }

    // 4. 收集最终根子节点统计
    let final_children = collect_child_stats(&tree, tree.root);

    // 5. 推荐动作 + 学习目标
    let rec_idx = scheduler
        .final_recommendation(&final_children, min_max.range())
        .unwrap_or_else(|| policy.recommend(&final_children, cfg, rng));
    let (recommended_id, recommended) = if rec_idx < final_children.len() {
        (
            final_children[rec_idx].action_id,
            final_children[rec_idx].action.clone(),
        )
    } else if !final_children.is_empty() {
        (
            final_children[0].action_id,
            final_children[0].action.clone(),
        )
    } else {
        let candidate = root_out
            .candidates
            .candidates
            .first()
            .cloned()
            .unwrap_or_else(|| {
                ActionCandidate::new(
                    super::types::ActionId(0),
                    super::types::ActionPayload::Discrete(0),
                    1.0,
                )
            });
        (candidate.id, candidate.payload)
    };

    let learn_policy = policy.make_targets(&final_children, cfg);

    SearchResult {
        children: final_children,
        recommended,
        recommended_id,
        learn_policy,
        network_value: root_out.value,
        q_range: min_max.range(),
    }
}

/// 将确定性 `RecurrentOut` 应用到叶子节点（K=1 快路径提取共用逻辑）。
fn apply_recurrent_to_leaf<S: Clone + 'static>(
    tree: &mut Tree<S>,
    candidate_provider: &ConfiguredCandidateProvider,
    leaf_id: usize,
    rec_out: &super::types::RecurrentOut<S>,
    cfg: &MctsConfig,
    rng: &mut dyn RngCore,
) {
    if let Some(parent_id) = tree.nodes[leaf_id].parent {
        let edge_idx = tree.nodes[leaf_id].action_from_parent.unwrap_or(0);
        if edge_idx < tree.nodes[parent_id].children.len() {
            let edge = &mut tree.nodes[parent_id].children[edge_idx];
            edge.reward = rec_out.reward;
            edge.discount = rec_out.discount;
        }
    }
    tree.nodes[leaf_id].terminal = rec_out.terminal;
    tree.nodes[leaf_id].to_play = rec_out.to_play;
    tree.states[leaf_id] = Some(rec_out.state.clone());

    if !rec_out.terminal && !rec_out.candidates.is_empty() {
        crate::prof_scope!("mcts.expand");
        let candidates =
            candidate_provider.expand_candidates(&rec_out.candidates, cfg, false, rng);
        tree.expand(leaf_id, &candidates.candidates, rec_out.to_play, rec_out.discount);
    }
}

/// 基于当前配置的候选展开策略。
#[derive(Debug, Clone, Copy, Default)]
struct ConfiguredCandidateProvider;

impl CandidateProvider for ConfiguredCandidateProvider {
    fn expand_candidates(
        &self,
        candidates: &CandidateSet,
        cfg: &MctsConfig,
        is_root: bool,
        rng: &mut dyn RngCore,
    ) -> CandidateSet {
        match cfg.sampled() {
            None => candidates.clone(),
            Some(sampled) if is_root => sample_root_for_expansion(candidates, cfg, sampled.k, rng),
            Some(sampled) => sample_for_expansion(candidates, sampled.k, rng),
        }
    }
}

/// 展开根节点的子节点，可指定子节点类型。
fn expand_root_with_kind<S: Clone + 'static>(
    tree: &mut Tree<S>,
    candidates: &[ActionCandidate],
    to_play: u8,
    discount: f32,
    child_kind: NodeKind,
) {
    let mut edges = Vec::with_capacity(candidates.len());
    for (i, candidate) in candidates.iter().enumerate() {
        let child = Node {
            parent: Some(tree.root),
            action_from_parent: Some(i),
            children: Vec::new(),
            visit_count: 0,
            terminal: false,
            to_play,
            expanded: false,
            kind: child_kind,
        };
        let child_id = tree.add_node(child, None);
        edges.push(super::node::Edge {
            action_id: candidate.id,
            action: candidate.payload.clone(),
            child: child_id,
            prior: candidate.policy_prior,
            visit_count: 0,
            value_sum: 0.0,
            reward: 0.0,
            discount,
        });
    }
    tree.nodes[tree.root].children = edges;
    tree.nodes[tree.root].expanded = true;
}

/// Selection：从根向下选到未展开叶子（所有节点均为 Decision，统一 PUCT）。
fn select<S: Clone + 'static, P: SearchPolicy>(
    tree: &Tree<S>,
    policy: &P,
    stats: &MinMaxStats,
    cfg: &MctsConfig,
    forced_root: Option<usize>,
    scratch: &mut Vec<ChildStat>,
) -> usize {
    let mut current = tree.root;
    if let Some(ci) = forced_root {
        let root = &tree.nodes[tree.root];
        if root.expanded && ci < root.children.len() {
            current = root.children[ci].child;
        }
    }
    loop {
        let node = &tree.nodes[current];
        if !node.expanded || node.children.is_empty() {
            return current;
        }

        let parent_to_play = node.to_play;
        scratch.clear();
        scratch.extend(node.children.iter().map(|edge| {
            let child_node = &tree.nodes[edge.child];
            ChildStat {
                action_id: edge.action_id,
                action: edge.action.clone(),
                visit_count: edge.visit_count,
                value_sum: edge.value_sum,
                prior: edge.prior,
                reward: edge.reward,
                to_play: child_node.to_play,
                discount: edge.discount,
            }
        }));
        let idx = policy.select_child(node.visit_count, parent_to_play, scratch, stats, cfg);
        let idx = idx.min(node.children.len().saturating_sub(1));
        current = node.children[idx].child;
    }
}

/// 按 chance prior 概率采样一个 outcome。
fn sample_chance_outcome(priors: &[f32], rng: &mut dyn RngCore) -> usize {
    let total: f32 = priors.iter().sum();
    let mut r = rng.r#gen::<f32>() * total;
    for (i, &p) in priors.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i;
        }
    }
    priors.len() - 1
}

/// Backup：从叶子向根回传价值
///
/// # v0.23+ TODO
/// - scalar↔categorical value 支持变换（MuZero 原论文用 categorical）
/// - virtual loss 支持（并行 MCTS 时防重复展开同一路径）
/// - tree reuse（搜索后不丢弃树，下一步 rebase 根节点）
fn backup<S: Clone + 'static>(
    tree: &mut Tree<S>,
    leaf_id: usize,
    leaf_value: f32,
    stats: &mut MinMaxStats,
    _cfg: &MctsConfig,
) {
    let mut current = leaf_id;
    let mut value = leaf_value;

    // 从叶子往上回传
    loop {
        // 更新当前节点
        tree.nodes[current].visit_count += 1;

        match tree.nodes[current].parent {
            Some(pid) => {
                let edge_idx = tree.nodes[current].action_from_parent.unwrap_or(0);
                let (reward, discount) = {
                    let edge = &mut tree.nodes[pid].children[edge_idx];
                    edge.visit_count += 1;
                    edge.value_sum += value;
                    (edge.reward, edge.discount)
                };
                let to_play = tree.nodes[current].to_play;
                let parent_to_play = tree.nodes[pid].to_play;
                let perspective = if to_play == parent_to_play { 1.0 } else { -1.0 };
                value = reward + discount * value * perspective;
                // MinMaxStats 用与 select 相同的 Q 定义更新，保持归一化一致
                stats.update(value);
                current = pid;
            }
            None => break,
        }
    }
}

/// 从树中收集某节点的子节点统计
fn collect_child_stats<S>(tree: &Tree<S>, node_id: usize) -> Vec<ChildStat> {
    tree.nodes[node_id]
        .children
        .iter()
        .map(|edge| {
            let child = &tree.nodes[edge.child];
            ChildStat {
                action_id: edge.action_id,
                action: edge.action.clone(),
                visit_count: edge.visit_count,
                value_sum: edge.value_sum,
                prior: edge.prior,
                reward: edge.reward,
                to_play: child.to_play,
                discount: edge.discount,
            }
        })
        .collect()
}

/// 将修改后的 ChildStat（如噪声注入后的 prior）写回树
fn apply_child_stats_to_tree<S>(tree: &mut Tree<S>, node_id: usize, stats: &[ChildStat]) {
    for (i, stat) in stats.iter().enumerate() {
        if i < tree.nodes[node_id].children.len() {
            tree.nodes[node_id].children[i].prior = stat.prior;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::mcts::types::{ActionCandidate, ActionId, ActionPayload};

    #[test]
    fn apply_child_stats_updates_edge_prior_single_source() {
        let mut tree = Tree::new(0_u32, 0);
        let candidates = vec![
            ActionCandidate::new(ActionId(0), ActionPayload::Discrete(0), 0.8),
            ActionCandidate::new(ActionId(1), ActionPayload::Discrete(1), 0.2),
        ];
        tree.expand(tree.root, &candidates, 0, 1.0);

        let mut stats = collect_child_stats(&tree, tree.root);
        stats[0].prior = 0.3;
        stats[1].prior = 0.7;
        apply_child_stats_to_tree(&mut tree, 0, &stats);

        assert!((tree.nodes[0].children[0].prior - 0.3).abs() < 1e-6);
        assert!((tree.nodes[0].children[1].prior - 0.7).abs() < 1e-6);
    }

    /// expand 后子节点隐状态应为 None（推理后才写入；与 expand_root 语义一致）
    ///
    /// 回归：旧实现预克隆父状态填充子节点（值语义错误且从未被读），
    /// 大动作空间下每次 expansion 白付 候选数 × latent 的克隆开销。
    #[test]
    fn expand_leaves_child_states_none_until_recurrent() {
        let mut tree = Tree::new(vec![1.0f32; 8], 0);
        let candidates = vec![
            ActionCandidate::new(ActionId(0), ActionPayload::Discrete(0), 0.5),
            ActionCandidate::new(ActionId(1), ActionPayload::Discrete(1), 0.3),
            ActionCandidate::new(ActionId(2), ActionPayload::Discrete(2), 0.2),
        ];
        tree.expand(tree.root, &candidates, 0, 1.0);

        // 根自身的状态保留；子节点状态为 None，直到被选为 leaf 并 recurrent 后写入
        assert!(tree.states[tree.root].is_some());
        for edge in &tree.nodes[tree.root].children {
            assert!(
                tree.states[edge.child].is_none(),
                "expand 不应预填充子节点状态"
            );
        }
    }
}
