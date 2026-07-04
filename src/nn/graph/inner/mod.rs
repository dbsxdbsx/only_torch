/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : GraphInner 计算图的底层实现
 *
 * 各 impl 块分散在子模块中：
 * - core.rs: 基础操作 + forward
 * - backward.rs: VJP 反向传播
 * - mode.rs: train/eval/detach
 * - recurrent.rs: 循环机制
 * - bptt.rs: BPTT
 * - node_builders.rs: new_*_node
 * - serialization.rs: save_params/load_params 底层参数序列化
 * - model_io.rs: save_weights/load_weights 权重 I/O
 * - describe.rs: describe/summary
 * - visualization.rs: DOT 可视化
 */

mod backward;
mod core;
mod describe;
mod mode;
mod model_io;
mod node_builders;
mod serialization;
mod visualization;

// 注意：bptt.rs 和 recurrent.rs 已删除
// 新架构使用展开式 RNN，BPTT 通过标准 backward() 自动完成

use super::types::{Mode, NodeGroupTag, RecurrentFoldingMeta};
use crate::nn::nodes::{NodeId, NodeInner};
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::rc::Weak;

/// CSE 去重缓存 key（紧凑形态，避免旧 `(String, …, Option<NodeGroupTag>)`
/// 元组 key 每次建节点的 String / Tag 多重 clone 分配）
///
/// 去重充要条件与 [`crate::nn::nodes::raw_node::TraitNode::dedup_fingerprint`]
/// 文档一致；`group` 用 `(instance_id, hidden)` 等价替代完整 `NodeGroupTag`：
/// `instance_id` 全局递增唯一，同一实例内除 `hidden`（RNN 步进时切换）外
/// 其余字段（`group_type` / `display_name` / …）在 guard 创建时一次性固定。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(in crate::nn::graph) struct CseKey {
    /// 节点类型（builder 传入的 `'static` 字面量）
    pub node_type: &'static str,
    /// 父节点 ID 列表（按序）
    pub parents: Vec<NodeId>,
    /// 节点配置指纹（`dedup_fingerprint`）
    pub fingerprint: u64,
    /// 分组上下文等价键：`(instance_id, hidden)`
    pub group: Option<(usize, bool)>,
}

/// 图的完整定义（核心实现）
///
/// 这是计算图的核心实现。用户通常通过 `Graph` 句柄使用此结构，
/// 高级用户（如 NEAT）可通过 `graph.inner()` 访问底层操作。
///
/// `parameters` 是参数注册表（弱引用，不控制生命周期）
pub struct GraphInner {
    pub(in crate::nn::graph) name: String,
    /// 最后一次前向传播的 id
    pub(in crate::nn::graph) last_forward_pass_id: u64,
    /// 最后一次反向传播的 id
    pub(in crate::nn::graph) last_backward_pass_id: u64,
    pub(in crate::nn::graph) next_id: u64,
    /// 当前执行模式（Train / Inference），forward 时透传给所有节点
    pub(in crate::nn::graph) mode: Mode,

    // ========== 参数管理 ==========
    /// 参数注册表（弱引用，不控制参数生命周期）
    ///
    /// - key: 参数名称（如 "linear1.weight"）
    /// - value: 弱引用，当 Layer 销毁时自动失效
    ///
    /// 用途：
    /// - `zero_grad()`: 遍历清除所有参数梯度
    /// - `parameters()`: 获取所有存活的参数
    /// - 序列化：保存/加载命名参数
    pub(in crate::nn::graph) parameters: HashMap<String, Weak<NodeInner>>,
    /// 图级别的随机数生成器（用于参数初始化等）
    /// None 表示使用默认的 `thread_rng（非确定性`）
    pub(in crate::nn) rng: Option<StdRng>,
    /// 循环层折叠渲染元信息（仅保留折叠所需的最小信息）
    pub(in crate::nn::graph) recurrent_folding_metas: Vec<RecurrentFoldingMeta>,

    // ========== 动态图节点命名 ==========
    /// 节点类型计数器：用于同批次内区分同类型节点
    /// key: 节点类型字符串, value: 当前计数
    pub(in crate::nn::graph) node_type_counts: HashMap<String, u64>,
    /// 上次重置计数器时的 `forward_pass_id`
    /// 当 forward 完成后，下一次创建节点时会检测到 `pass_id` 变化并重置计数器
    pub(in crate::nn::graph) counts_reset_pass_id: u64,

    // ========== 节点分组上下文 ==========
    /// 当前激活的分组上下文（非栈，"外层优先"策略）
    ///
    /// 由 `NodeGroupContext` RAII guard 设置/清除。
    /// 在上下文激活期间，`create_node_inner` 会自动给新建的计算节点打上此标签。
    pub(in crate::nn::graph) node_group_context: Option<NodeGroupTag>,
    /// 下一个分组实例 ID（全局递增）
    pub(in crate::nn::graph) next_node_group_id: usize,
    /// 当前上下文是否标记 Parameter 节点（Layer/Recurrent 为 true，Distribution 为 false）
    pub(in crate::nn::graph) node_group_include_params: bool,

    // ========== CSE 去重缓存 ==========
    /// CSE（公共子表达式消除）节点去重缓存
    ///
    /// value: `Weak<NodeInner>`（不阻止节点被 Rc 回收）
    ///
    /// 随 `forward_pass_id` 变化自动清空（同 `node_type_counts` 机制）。
    pub(in crate::nn::graph) cse_cache: HashMap<CseKey, Weak<NodeInner>>,
    /// CSE 缓存上次重置时的 `forward_pass_id`
    pub(in crate::nn::graph) cse_cache_reset_pass_id: u64,

    // ========== 可视化快照 ==========
    /// 可视化拓扑快照（由 `Graph::snapshot_once` 写入）
    ///
    /// 在训练循环中 Var 还活着时拍快照，之后 Var 被 drop 也不影响。
    /// 通过 `Graph::visualize_snapshot` 从快照渲染 DOT/PNG。
    pub(in crate::nn::graph) visualization_snapshot: Option<super::types::VisualizationSnapshot>,
}

impl Default for GraphInner {
    fn default() -> Self {
        Self::new()
    }
}
