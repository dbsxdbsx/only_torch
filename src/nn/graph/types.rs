/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : Graph 模块的类型定义
 */

use crate::nn::NodeId;

// ==================== 节点分组标签 ====================

/// 通用节点分组标签
///
/// 当前用于概率分布的 cluster 可视化（如 Categorical、Normal），
/// 未来将统一替代 `LayerGroup` 的显式注册机制（阶段二）。
///
/// 同一 `(group_type, instance_id)` 的节点在可视化中归入同一个 cluster 子图。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeGroupTag {
    /// 分组类型名（如 "Categorical" / "Normal" / "TanhNormal"）
    pub group_type: String,
    /// 实例 ID（区分同类型的多个实例）
    pub instance_id: usize,
}

// ==================== 层/模型分组（旧体系，阶段二统一后移除）====================

/// 分组类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupKind {
    /// 层级分组（如 Linear、Conv2d）
    Layer,
    /// 模型级分组（如 Generator、Discriminator）
    Model,
}

/// 层分组信息（用于可视化时将属于同一层/模型的节点框在一起）
#[derive(Debug, Clone)]
pub struct LayerGroup {
    /// 层名称（如 "fc1", "conv1"）
    pub name: String,
    /// 层类型（如 "Linear", "Conv2d", "Model"）
    pub layer_type: String,
    /// 层的描述信息（如 "784→128"）
    pub description: String,
    /// 属于该层的节点 ID 列表（用于可视化显示）
    pub node_ids: Vec<NodeId>,
    /// 分组类型
    pub kind: GroupKind,
    /// 循环层的时间步数（None 表示非循环层）
    /// 对于变长序列，使用 `min_steps` 和 `max_steps`
    pub recurrent_steps: Option<usize>,
    /// 循环层的最小时间步数（变长序列支持）
    pub min_steps: Option<usize>,
    /// 循环层的最大时间步数（变长序列支持）
    pub max_steps: Option<usize>,
    /// 需要在可视化中隐藏的节点 ID（如循环层的非代表性展开节点）
    pub hidden_node_ids: Vec<NodeId>,
    /// 代表性节点（折叠显示，实际代表多个节点）
    /// `格式：(node_id`, 实际重复次数)
    pub folded_nodes: Vec<(NodeId, usize)>,
    /// 输出代理：(隐藏的真实输出节点, 代表性输出节点)
    /// 用于在可视化时连接代表性节点到下游（折叠层的输出流向）
    pub output_proxy: Option<(NodeId, NodeId)>,
}

/// 循环层元信息（惰性收集可视化信息）
///
/// 分为两部分：
/// - 静态信息：层创建时注册（参数节点、每步节点数）
/// - 动态信息：forward 时更新（时间步数、起始/结束节点 ID）
///
/// 只在 `save_visualization` 时才根据此元信息推断完整的展开分组，
/// 避免 forward 时的额外开销（只记录几个数值）。
#[derive(Debug, Clone)]
pub struct RecurrentLayerMeta {
    /// 层名称（如 "rnn", "lstm"）
    pub name: String,
    /// 层类型（"RNN", "LSTM", "GRU"）
    pub layer_type: String,
    /// 层描述（如 "1→16"）
    pub description: String,
    /// 参数节点 ID 列表
    pub param_node_ids: Vec<NodeId>,
    /// 每个时间步的计算节点数量（不含参数和初始状态）
    /// RNN: 6 (select, matmul, matmul, add, add, tanh)
    pub nodes_per_step: usize,
    /// 动态信息列表：每次 forward 调用都会追加（支持变长序列）
    /// 惰性推断时使用最后一次调用作为代表，其他调用的节点被隐藏
    pub unroll_infos: Vec<RecurrentUnrollInfo>,
}

/// 循环层展开的动态信息（forward 时记录）
#[derive(Debug, Clone)]
pub struct RecurrentUnrollInfo {
    /// 时间步数
    pub steps: usize,
    /// 输入节点 ID
    pub input_node_id: NodeId,
    /// 初始状态节点 ID 列表（如 [h0] 或 [h0, c0]）
    /// - RNN/GRU：只有一个 h0
    /// - LSTM：有两个 [h0, c0]
    pub init_state_node_ids: Vec<NodeId>,
    /// 第一个时间步的第一个计算节点 ID（如 Select）
    pub first_step_start_id: NodeId,
    /// 代表性输出节点列表（第一个时间步的各状态输出）
    /// - RNN/GRU：[`h_1`]
    /// - LSTM：[`h_1`, `c_1`]（h 和 c 的第一个时间步输出）
    pub repr_output_node_ids: Vec<NodeId>,
    /// 真实输出节点（最后一个时间步的隐藏状态输出，如 `h_N`）
    pub real_output_node_id: NodeId,
}

// ==================== 可视化快照 ====================

/// 可视化快照中的单个节点——从活 `NodeInner` 提取的轻量级副本
///
/// 只保留渲染 DOT 所需的最少信息，不持有 `Rc<NodeInner>` 引用。
/// 快照创建后完全独立于节点生命周期，Var 被 drop 也不影响。
#[derive(Debug, Clone)]
pub struct SnapshotNode {
    /// 节点 ID
    pub id: NodeId,
    /// 节点名称（如 "fc1_W", "add_3"）
    pub name: Option<String>,
    /// 节点类型名（如 "Parameter", "MatMul", "MSE"）
    pub type_name: String,
    /// 期望输出形状
    pub shape: Vec<usize>,
    /// 父节点 ID 列表（数据流方向：parent → self）
    pub parent_ids: Vec<NodeId>,
    /// 是否已 detach
    pub is_detached: bool,
    /// 超参数 HTML 片段（如 Dropout 的 `<BR/>(p=0.5)`，无超参数时为 None）
    pub hyperparam_html: Option<String>,
    /// 节点分组标签（如属于某个概率分布的 cluster）
    pub node_group_tag: Option<NodeGroupTag>,
}

/// 可视化拓扑快照——计算图的轻量级结构副本
///
/// 在 Var 还活着时由 `Graph::snapshot_once` 创建。
/// 存储在 `GraphInner` 中，之后随时可用于生成 DOT/PNG。
///
/// # 使用场景
/// ```ignore
/// // 训练循环内，backward 之后、Var drop 之前
/// graph.snapshot_once(&[("Actor Loss", &actor_loss), ("Critic Loss", &critic1_loss)]);
///
/// // 训练结束后，任意时刻渲染
/// graph.visualize_snapshot("examples/cartpole_sac/cartpole_sac")?;
/// ```
#[derive(Debug, Clone)]
pub struct VisualizationSnapshot {
    /// 所有可达节点（BFS 收集，顺序与遍历顺序一致）
    pub nodes: Vec<SnapshotNode>,
    /// 命名输出端点：(用户给的名称, 对应节点 ID)
    pub named_outputs: Vec<(String, NodeId)>,
}
