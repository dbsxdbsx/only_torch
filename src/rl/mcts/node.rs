//! Arena-based 搜索树结构
//!
//! # v0.23+ TODO
//! - tree reuse：搜索后不丢弃树，下一步 rebase root + 清理兄弟子树 + arena 复用
//! - 并行：Node 字段不含 Python 借用 / `Rc<RefCell>`，可条件加 `Send + Sync`
//! - Stochastic MuZero：`NodeKind::{Decision, Chance}` 两类节点交替（核心扩展级）
//! - 连续动作：Edge 的 ActionPayload::Continuous 不能做 map key，保持 Vec 存储

use super::types::{ActionCandidate, ActionId, ActionPayload};

/// 节点索引（arena 模式下即 Vec 下标）
pub(crate) type NodeId = usize;

/// 从父节点到子节点的边
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct Edge {
    /// 策略 / target 使用的稳定动作 id
    pub action_id: ActionId,
    /// 对应动作
    pub action: ActionPayload,
    /// 子节点 ID
    pub child: NodeId,
    /// 先验概率
    pub prior: f32,
    /// 访问计数
    pub visit_count: u32,
    /// 累计价值（子节点视角）
    pub value_sum: f32,
    /// 从父到此子的即时奖励
    pub reward: f32,
    /// 子边 transition discount
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
    /// 创建只含根节点的树
    pub fn new(root_state: S, to_play: u8) -> Self {
        let root_node = Node {
            parent: None,
            action_from_parent: None,
            children: Vec::new(),
            visit_count: 0,
            terminal: false,
            to_play,
            expanded: false,
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
    pub fn expand(
        &mut self,
        node_id: NodeId,
        candidates: &[ActionCandidate],
        states: &[S],
        to_play: u8,
        discount: f32,
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
            };
            let state = states.get(i).cloned();
            let child_id = self.add_node(child, state);
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
}
