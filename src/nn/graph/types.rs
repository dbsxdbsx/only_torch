/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : Graph 模块的类型定义
 */

use crate::nn::NodeId;
use crate::tensor::Tensor;

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
    /// 格式：(node_id, 实际重复次数)
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
    /// 输入节点 ID（如 SmartInput）
    pub input_node_id: NodeId,
    /// 初始状态节点 ID 列表（如 [h0] 或 [h0, c0]）
    /// - RNN/GRU：只有一个 h0
    /// - LSTM：有两个 [h0, c0]
    pub init_state_node_ids: Vec<NodeId>,
    /// 第一个时间步的第一个计算节点 ID（如 Select）
    pub first_step_start_id: NodeId,
    /// 代表性输出节点列表（第一个时间步的各状态输出）
    /// - RNN/GRU：[h_1]
    /// - LSTM：[h_1, c_1]（h 和 c 的第一个时间步输出）
    pub repr_output_node_ids: Vec<NodeId>,
    /// 真实输出节点（最后一个时间步的隐藏状态输出，如 h_N）
    pub real_output_node_id: NodeId,
}

/// BPTT 时间步快照：存储节点在某个时间步的状态
#[derive(Clone)]
pub(crate) struct StepSnapshot {
    /// 节点的值
    pub value: Option<Tensor>,
}
