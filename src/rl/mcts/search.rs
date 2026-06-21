//! MCTS 主搜索循环

use rand::RngCore;

use super::min_max::MinMaxStats;
use super::node::{Node, Tree};
use super::traits::{MctsModel, SearchPolicy};
use super::types::{ChildStat, MctsConfig, SearchResult};

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
    if root_out.candidate_actions.is_empty() {
        return SearchResult {
            children: Vec::new(),
            recommended: super::types::ActionPayload::Discrete(0),
            learn_policy: Vec::new(),
            network_value: root_out.value,
        };
    }

    let mut tree = Tree::new(root_out.state.clone(), root_out.to_play);
    expand_root(
        &mut tree,
        model,
        &root_out.candidate_actions,
        &root_out.prior,
        &root_out.state,
        root_out.to_play,
        cfg.discount,
    );

    // 根节点 backup 初始价值
    tree.nodes[tree.root].visit_count = 1;
    tree.nodes[tree.root].value_sum = root_out.value;

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
        tree.nodes[leaf_id].reward = rec_out.reward;
        tree.nodes[leaf_id].terminal = rec_out.terminal;
        tree.nodes[leaf_id].to_play = rec_out.to_play;
        tree.nodes[leaf_id].discount = rec_out.discount;
        tree.states[leaf_id] = Some(rec_out.state.clone());

        if !rec_out.terminal && !rec_out.candidate_actions.is_empty() {
            // 展开子节点
            let n_children = rec_out.candidate_actions.len();
            let priors = if rec_out.prior.len() == n_children {
                rec_out.prior.clone()
            } else {
                vec![1.0 / n_children as f32; n_children]
            };

            tree.expand(
                leaf_id,
                &rec_out.candidate_actions,
                &priors,
                &vec![rec_out.state.clone(); n_children],
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
            .candidate_actions
            .first()
            .cloned()
            .unwrap_or(super::types::ActionPayload::Discrete(0))
    };

    let learn_policy = policy.make_targets(&final_children, cfg);

    SearchResult {
        children: final_children,
        recommended,
        learn_policy,
        network_value: root_out.value,
    }
}

/// 展开根节点的子节点
fn expand_root<M: MctsModel>(
    tree: &mut Tree<M::State>,
    _model: &M,
    actions: &[super::types::ActionPayload],
    priors: &[f32],
    _root_state: &M::State,
    to_play: u8,
    discount: f32,
) {
    let n = actions.len();
    let adjusted_priors = if priors.len() == n {
        priors.to_vec()
    } else {
        vec![1.0 / n as f32; n]
    };

    let mut edges = Vec::with_capacity(n);
    for (i, (action, &prior)) in actions.iter().zip(adjusted_priors.iter()).enumerate() {
        let child = Node {
            parent: Some(tree.root),
            action_from_parent: Some(i),
            children: Vec::new(),
            visit_count: 0,
            value_sum: 0.0,
            prior,
            reward: 0.0,
            terminal: false,
            to_play,
            discount,
            expanded: false,
        };
        let child_id = tree.add_node(child, None);
        edges.push(super::node::Edge {
            action: action.clone(),
            child: child_id,
            prior,
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
                    action: edge.action.clone(),
                    visit_count: child_node.visit_count,
                    value_sum: child_node.value_sum,
                    prior: child_node.prior,
                    reward: child_node.reward,
                    to_play: child_node.to_play,
                    discount: child_node.discount,
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
        let node = &tree.nodes[current];
        let parent_id = node.parent;
        let discount = node.discount;
        let reward = node.reward;
        let to_play = node.to_play;

        // 更新当前节点
        tree.nodes[current].visit_count += 1;
        tree.nodes[current].value_sum += value;

        match parent_id {
            Some(pid) => {
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
                action: edge.action.clone(),
                visit_count: child.visit_count,
                value_sum: child.value_sum,
                prior: child.prior,
                reward: child.reward,
                to_play: child.to_play,
                discount: child.discount,
            }
        })
        .collect()
}

/// 将修改后的 ChildStat（如噪声注入后的 prior）写回树
fn apply_child_stats_to_tree<S>(tree: &mut Tree<S>, node_id: usize, stats: &[ChildStat]) {
    let child_ids: Vec<usize> = tree.nodes[node_id]
        .children
        .iter()
        .map(|e| e.child)
        .collect();
    for (i, stat) in stats.iter().enumerate() {
        if i < child_ids.len() {
            tree.nodes[child_ids[i]].prior = stat.prior;
        }
    }
}
