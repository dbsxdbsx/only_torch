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
 * - model_io.rs: save_model/load_model 高层模型 I/O
 * - describe.rs: describe/summary
 * - visualization.rs: DOT 可视化
 * - evolution.rs: Evolution API (骨架)
 */

mod backward;
mod bptt;
mod core;
mod describe;
mod evolution;
mod mode;
mod model_io;
mod node_builders;
mod recurrent;
mod serialization;
mod visualization;

use super::types::{LayerGroup, RecurrentLayerMeta, StepSnapshot};
use crate::nn::nodes::NodeHandle;
use crate::nn::NodeId;
use crate::tensor::Tensor;
use rand::rngs::StdRng;
use std::collections::HashMap;

/// 图的完整定义（核心实现）
///
/// 这是计算图的核心实现。用户通常通过 `Graph` 句柄使用此结构，
/// 高级用户（如 NEAT）可通过 `graph.inner()` 访问底层操作。
pub struct GraphInner {
    pub(in crate::nn::graph) name: String,
    pub(in crate::nn::graph) nodes: HashMap<NodeId, NodeHandle>,
    /// 正向边：parent_id -> child_ids（父节点指向子节点）
    pub(in crate::nn::graph) forward_edges: HashMap<NodeId, Vec<NodeId>>,
    /// 反向边：child_id -> parent_ids（子节点指向父节点）
    pub(in crate::nn::graph) backward_edges: HashMap<NodeId, Vec<NodeId>>,
    /// 最后一次前向传播的 id
    pub(in crate::nn::graph) last_forward_pass_id: u64,
    /// 最后一次反向传播的 id
    pub(in crate::nn::graph) last_backward_pass_id: u64,
    pub(in crate::nn::graph) next_id: u64,
    pub(in crate::nn::graph) is_eval_mode: bool,
    /// 图级别的随机数生成器（用于参数初始化等）
    /// None 表示使用默认的 thread_rng（非确定性）
    pub(in crate::nn::graph) rng: Option<StdRng>,
    /// 层分组信息（用于可视化）
    pub(in crate::nn::graph) layer_groups: Vec<LayerGroup>,
    /// 循环层元信息（惰性收集：只在可视化时才根据此信息推断完整分组）
    pub(in crate::nn::graph) recurrent_layer_metas: Vec<RecurrentLayerMeta>,

    // ========== 循环/记忆机制相关字段 ==========
    /// 循环边：to_node -> from_node（to 节点在 step() 时从 from 节点的上一步值读取）
    pub(in crate::nn::graph) recurrent_edges: HashMap<NodeId, NodeId>,
    /// 双缓冲：存储循环节点的上一时间步值
    pub(in crate::nn::graph) prev_values: HashMap<NodeId, Tensor>,
    /// 当前时间步（用于调试，每次 step() 递增，reset() 归零）
    pub(in crate::nn::graph) time_step: u64,

    // ========== BPTT 相关字段 ==========
    /// 时间步历史：存储每个时间步的节点快照，用于 BPTT
    /// 每个元素是一个时间步的快照：NodeId -> (value, jacobi)
    /// 只在训练模式下记录
    pub(in crate::nn::graph) step_history: Vec<HashMap<NodeId, StepSnapshot>>,

    /// BPTT 调试标志（仅用于调试）
    #[cfg(test)]
    pub(in crate::nn::graph) bptt_debug: bool,
}

impl Default for GraphInner {
    fn default() -> Self {
        Self::new()
    }
}
