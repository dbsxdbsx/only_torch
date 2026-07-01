//! MCTS 主搜索循环

use rand::RngCore;

use super::min_max::MinMaxStats;
use super::node::{Node, Tree};
use super::sampled::{sample_for_expansion, sample_root_for_expansion};
use super::traits::{CandidateProvider, MctsModel, SearchPolicy};
use super::types::{ActionCandidate, CandidateSet, ChildStat, MctsConfig, SearchResult};

/// 执行 MCTS 搜索
///
/// # 参数
/// - `model`: 提供 root/recurrent 推理的模型
/// - `policy`: 搜索策略 hook（选择、推荐、噪声注入）
/// - `obs`: 当前观测
/// - `cfg`: 搜索配置
///
/// # 返回
/// 搜索结果，包含子节点统计、推荐动作和学习用策略目标
pub fn mcts_search<M: MctsModel, P: SearchPolicy>(
    model: &M,
    policy: &P,
    obs: &[f32],
    cfg: &MctsConfig,
    rng: &mut dyn RngCore,
) -> SearchResult {
    // 1. root 推理 → 建根节点 + 展开子节点
    let root_out = model.root(obs);

    // 空候选 / 终局 root → 直接返回空结果
    if root_out.candidates.is_empty() {
        return SearchResult {
            children: Vec::new(),
            recommended: super::types::ActionPayload::Discrete(0),
            learn_policy: Vec::new(),
            network_value: root_out.value,
            q_range: None,
        };
    }

    let mut tree = Tree::new(root_out.state.clone(), root_out.to_play);
    let candidate_provider = ConfiguredCandidateProvider;
    let root_candidates =
        candidate_provider.expand_candidates(&root_out.candidates, cfg, true, rng);
    expand_root(
        &mut tree,
        model,
        &root_candidates.candidates,
        &root_out.state,
        root_out.to_play,
        cfg.discount,
    );

    // 根节点 backup 初始价值
    tree.nodes[tree.root].visit_count = 1;

    // 2. 注入根节点 Dirichlet 噪声
    let root_id = tree.root;
    let mut root_child_stats = collect_child_stats(&tree, root_id);
    policy.prepare_root(&mut root_child_stats, cfg, rng);
    apply_child_stats_to_tree(&mut tree, root_id, &root_child_stats);

    // 3. 模拟循环（含根调度 hook：默认 PuctScheduler 不干预、零开销）
    let mut min_max = MinMaxStats::new();
    let num_root_children = tree.nodes[tree.root].children.len();
    let mut scheduler = policy.make_root_scheduler(num_root_children, cfg);
    let use_scheduler = scheduler.is_active();
    if use_scheduler {
        scheduler.on_search_start(&root_child_stats, root_out.value, cfg, rng);
    }

    for sim_idx in 0..cfg.num_simulations as usize {
        // 根调度：Gumbel 等可强制本次模拟的根起步子节点；默认 None=走 PUCT
        let forced_root = if use_scheduler {
            let root_children = collect_child_stats(&tree, tree.root);
            scheduler.next_root_child(&root_children, sim_idx, cfg)
        } else {
            None
        };

        // selection: 从根往下选择
        let leaf_id = select(&tree, policy, &min_max, cfg, forced_root);

        // 若叶子已终止，只做 backup
        if tree.nodes[leaf_id].terminal {
            backup(&mut tree, leaf_id, 0.0, &mut min_max, cfg);
            continue;
        }

        // expansion: 获取父状态 + 动作 → recurrent → 展开
        let parent_id = tree.nodes[leaf_id].parent.unwrap_or(tree.root);
        let edge_idx = tree.nodes[leaf_id].action_from_parent.unwrap_or(0);
        let action = tree.nodes[parent_id].children[edge_idx].action.clone();

        let parent_state = tree.states[parent_id]
            .as_ref()
            .expect("parent state should exist");
        let rec_out = model.recurrent(parent_state, &action);

        // 更新叶子节点信息
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
            let candidates =
                candidate_provider.expand_candidates(&rec_out.candidates, cfg, false, rng);

            tree.expand(
                leaf_id,
                &candidates.candidates,
                &vec![rec_out.state.clone(); candidates.len()],
                rec_out.to_play,
                rec_out.discount,
            );
        }

        let backup_value = if rec_out.terminal { 0.0 } else { rec_out.value };
        backup(&mut tree, leaf_id, backup_value, &mut min_max, cfg);
    }

    // 4. 收集最终根子节点统计
    let final_children = collect_child_stats(&tree, tree.root);

    // 5. 推荐动作 + 学习目标（scheduler 可覆盖推荐，如 Gumbel 用最终幸存者）
    let rec_idx = scheduler
        .final_recommendation(&final_children)
        .unwrap_or_else(|| policy.recommend(&final_children, cfg, rng));
    let recommended = if rec_idx < final_children.len() {
        final_children[rec_idx].action.clone()
    } else if !final_children.is_empty() {
        final_children[0].action.clone()
    } else {
        root_out
            .candidates
            .candidates
            .first()
            .map(|c| c.payload.clone())
            .unwrap_or(super::types::ActionPayload::Discrete(0))
    };

    let learn_policy = policy.make_targets(&final_children, cfg);

    SearchResult {
        children: final_children,
        recommended,
        learn_policy,
        network_value: root_out.value,
        q_range: min_max.range(),
    }
}

/// 基于当前配置的候选展开策略。
///
/// 这是兼容层：后续 recipe 可直接装配不同 [`CandidateProvider`]。
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

/// 展开根节点的子节点
fn expand_root<M: MctsModel>(
    tree: &mut Tree<M::State>,
    _model: &M,
    candidates: &[ActionCandidate],
    _root_state: &M::State,
    to_play: u8,
    discount: f32,
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

/// Selection：从根沿 PUCT 策略向下选择到未展开叶子
fn select<S: Clone + 'static, P: SearchPolicy>(
    tree: &Tree<S>,
    policy: &P,
    stats: &MinMaxStats,
    cfg: &MctsConfig,
    forced_root: Option<usize>,
) -> usize {
    let mut current = tree.root;
    // 根调度 hook：若指定，强制第一步走该根子节点（其下仍走 PUCT 选择）
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
        let child_stats: Vec<ChildStat> = node
            .children
            .iter()
            .map(|edge| {
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
            })
            .collect();

        let idx = policy.select_child(node.visit_count, parent_to_play, &child_stats, stats, cfg);
        let idx = idx.min(node.children.len().saturating_sub(1));
        current = node.children[idx].child;
    }
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
        let states = vec![1_u32, 2_u32];
        tree.expand(tree.root, &candidates, &states, 0, 1.0);

        let mut stats = collect_child_stats(&tree, tree.root);
        stats[0].prior = 0.3;
        stats[1].prior = 0.7;
        apply_child_stats_to_tree(&mut tree, 0, &stats);

        assert!((tree.nodes[0].children[0].prior - 0.3).abs() < 1e-6);
        assert!((tree.nodes[0].children[1].prior - 0.7).abs() < 1e-6);
    }
}
