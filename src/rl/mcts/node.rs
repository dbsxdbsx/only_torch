//! Arena-based 搜索树结构
//!
//! Stochastic MuZero 扩展：树内 decision / chance 两类节点交替。
//! Decision node 的出边是 action 候选，chance node 的出边是 chance outcome。
//! `num_chance_outcomes == 1` 时搜索循环走确定性快路径，不创建 chance 节点。
//!
//! # 后续 TODO
//! - tree reuse：搜索后不丢弃树，下一步 rebase root + 清理兄弟子树 + arena 复用
//! - 并行：Node 字段不含 Python 借用 / `Rc<RefCell>`，可条件加 `Send + Sync`
//! - 连续动作：Edge 的 ActionPayload::Continuous 不能做 map key，保持 Vec 存储

use super::types::{ActionCandidate, ActionId, ActionPayload};

/// 节点索引（arena 模式下即 Vec 下标）
pub(crate) type NodeId = usize;

/// Stochastic MuZero 节点类型。
///
/// Decision node 的出边是 action 候选（PUCT 选择）；
/// Chance node 的出边是 chance outcome（按 learned prior 比例选择）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NodeKind {
    Decision,
    Chance,
}

/// 从父节点到子节点的边
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct Edge {
    /// 策略 / target 使用的稳定动作 id（decision 边）或 chance outcome id（chance 边）
    pub action_id: ActionId,
    /// 对应动作（chance 边使用 `Discrete(chance_id)` 占位）
    pub action: ActionPayload,
    /// 子节点 ID
    pub child: NodeId,
    /// 先验概率
    pub prior: f32,
    /// 访问计数
    pub visit_count: u32,
    /// 累计价值（子节点视角）
    pub value_sum: f32,
    /// 从父到此子的即时奖励（decision→chance 边为 0，chance→decision 边为真实 reward）
    pub reward: f32,
    /// 子边 transition discount（decision→chance 边为 1.0，chance→decision 边为 γ·(1-done)）
    pub discount: f32,
}

/// 搜索树节点
#[derive(Debug, Clone)]
pub(crate) struct Node {
    /// 父节点（根为 None）
    pub parent: Option<NodeId>,
    /// 该节点在父节点 children 列表中的索引
    pub action_from_parent: Option<usize>,
    /// 子边列表
    pub children: Vec<Edge>,
    /// 状态访问计数
    pub visit_count: u32,
    /// 是否终止态
    pub terminal: bool,
    /// 当前玩家
    pub to_play: u8,
    /// 是否已展开
    pub expanded: bool,
    /// 节点类型：Decision（agent 选 action）或 Chance（环境选 outcome）
    pub kind: NodeKind,
}

/// Arena-based 搜索树
#[derive(Debug)]
pub(crate) struct Tree<S> {
    /// 节点池
    pub nodes: Vec<Node>,
    /// 与 nodes 平行的隐状态存储
    pub states: Vec<Option<S>>,
    /// 根节点 ID
    pub root: NodeId,
}

impl<S: Clone + 'static> Tree<S> {
    /// 创建只含根节点（Decision）的树
    pub fn new(root_state: S, to_play: u8) -> Self {
        let root_node = Node {
            parent: None,
            action_from_parent: None,
            children: Vec::new(),
            visit_count: 0,
            terminal: false,
            to_play,
            expanded: false,
            kind: NodeKind::Decision,
        };
        Self {
            nodes: vec![root_node],
            states: vec![Some(root_state)],
            root: 0,
        }
    }

    /// 分配新节点并返回其 ID
    pub fn add_node(&mut self, node: Node, state: Option<S>) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        self.states.push(state);
        id
    }

    /// 展开节点：为其每个候选动作创建子节点
    ///
    /// 子节点的隐状态初始为 `None`（与 `expand_root` 一致）：
    /// 状态只在该子节点被选为 leaf、经 recurrent 推理后写入（`states[leaf] = Some(...)`），
    /// 且读取只发生在 parent 侧（写必先于读）。旧实现按候选数预克隆父状态填充，
    /// 既是语义错误的值（父状态的拷贝）又从未被读——大动作空间（如棋类数百候选）
    /// 时每次 expansion 白付 `候选数 × latent` 的克隆开销。
    /// 展开 decision 节点：为每个 action 候选创建子节点。
    ///
    /// `child_kind` 决定子节点类型：确定性模式（K=1）子节点仍为 Decision；
    /// Stochastic 模式（K>1）子节点为 Chance（afterstate）。
    pub fn expand(
        &mut self,
        node_id: NodeId,
        candidates: &[ActionCandidate],
        to_play: u8,
        discount: f32,
    ) {
        self.expand_with_kind(node_id, candidates, to_play, discount, NodeKind::Decision);
    }

    /// 展开节点并指定子节点类型。
    pub fn expand_with_kind(
        &mut self,
        node_id: NodeId,
        candidates: &[ActionCandidate],
        to_play: u8,
        discount: f32,
        child_kind: NodeKind,
    ) {
        let mut edges = Vec::with_capacity(candidates.len());
        for (i, candidate) in candidates.iter().enumerate() {
            let child = Node {
                parent: Some(node_id),
                action_from_parent: Some(i),
                children: Vec::new(),
                visit_count: 0,
                terminal: false,
                to_play,
                expanded: false,
                kind: child_kind,
            };
            let child_id = self.add_node(child, None);
            edges.push(Edge {
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
        self.nodes[node_id].children = edges;
        self.nodes[node_id].expanded = true;
    }

    /// 展开 chance 节点：为每个 chance outcome 创建 Decision 子节点。
    ///
    /// `chance_priors[i]` 是 outcome `i` 的 learned 概率。
    pub fn expand_chance(
        &mut self,
        node_id: NodeId,
        chance_priors: &[f32],
        to_play: u8,
    ) {
        let mut edges = Vec::with_capacity(chance_priors.len());
        for (i, &prior) in chance_priors.iter().enumerate() {
            let child = Node {
                parent: Some(node_id),
                action_from_parent: Some(i),
                children: Vec::new(),
                visit_count: 0,
                terminal: false,
                to_play,
                expanded: false,
                kind: NodeKind::Decision,
            };
            let child_id = self.add_node(child, None);
            edges.push(Edge {
                action_id: ActionId(i),
                action: ActionPayload::Discrete(i),
                child: child_id,
                prior,
                visit_count: 0,
                value_sum: 0.0,
                reward: 0.0,
                discount: 1.0,
            });
        }
        self.nodes[node_id].children = edges;
        self.nodes[node_id].expanded = true;
    }
}
