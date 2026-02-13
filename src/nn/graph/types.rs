/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : Graph 模块的类型定义
 */

use crate::nn::NodeId;

// ==================== 节点分组标签 ====================

/// 分组的视觉渲染风格
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupStyle {
    /// 层 cluster：实线填充、粗边框、显示描述（Linear / Conv2d 等）
    Layer,
    /// 分布 cluster：虚线、半透明紫色（Categorical / Normal / TanhNormal）
    Distribution,
    /// 循环层 cluster：三重边框、粗边框、显示步数（RNN / LSTM / GRU）
    Recurrent,
}

/// 通用节点分组标签
///
/// 统一用于所有 cluster 可视化：Layer、Distribution、Recurrent。
/// 同一 `(group_type, instance_id)` 的节点在可视化中归入同一个 cluster 子图。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeGroupTag {
    /// 分组类型名（如 "Linear" / "Categorical" / "RNN" 等）
    pub group_type: String,
    /// 实例 ID（区分同类型的多个实例）
    pub instance_id: usize,
    /// 显示名称（Layer/Recurrent: "Actor/fc1"，Distribution: None）
    /// 含 `/` 时用于推断模型层级（如 "Actor/fc1" → 模型 "Actor"、层 "fc1"）
    pub display_name: Option<String>,
    /// 描述信息（Layer: "[?, 784] -> [?, 128]"，Distribution: None）
    pub description: Option<String>,
    /// 视觉渲染风格
    pub style: GroupStyle,
    /// 是否在可视化中隐藏（RNN 步骤 1..N-1 的节点设为 true）
    pub hidden: bool,
}

// ==================== 循环层折叠渲染 ====================

/// 循环层折叠渲染元信息（仅保留时间步折叠所需的最小信息）
///
/// cluster 分组职责已迁移到 `NodeGroupTag`（通过 `NodeGroupContext::for_recurrent`），
/// 此结构仅负责折叠渲染：隐藏重复时间步、边重定向、反馈边、步数标注。
///
/// 分为两部分：
/// - 静态信息：层创建时注册（每步节点数）
/// - 动态信息：forward 时更新（时间步数、起始/结束节点 ID）
#[derive(Debug, Clone)]
pub struct RecurrentFoldingMeta {
    /// 层名称（用于关联到对应的 NodeGroupTag）
    pub name: String,
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
    /// 数据源 ID（仅 Input/TargetInput 节点，用于检测同源数据节点）
    pub data_source_id: Option<u64>,
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
