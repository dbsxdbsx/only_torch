/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : Graph 模块：计算图的核心实现
 *
 * 公开 API：
 * - `Graph`: 用户级句柄（PyTorch 风格）
 * - `GraphInner`: 底层实现（高级用户/NEAT 使用）
 * - `GraphError`: 错误类型
 */

mod descriptor_rebuild;
mod error;
mod handle;
mod inner;
pub(crate) mod model_save;
pub(crate) mod onnx_error;
pub(crate) mod onnx_export;
pub mod onnx_import;
pub(crate) mod onnx_ops;
mod types;

pub use descriptor_rebuild::RebuildResult;
pub use error::{GraphError, ImageFormat, VisualizationOutput};
pub use handle::Graph;
pub use inner::GraphInner;
pub use onnx_error::OnnxError;
// 可视化分组类型导出
pub use types::{
    GroupStyle, NodeGroupTag, RecurrentFoldingMeta, RecurrentUnrollInfo, SnapshotNode,
    VisualizationSnapshot,
};

// ==================== 节点分组上下文 ====================

use crate::nn::var::Var;
use std::cell::RefCell;
use std::rc::Rc;

/// 节点分组上下文 RAII guard
///
/// 在 guard 存活期间，通过 `create_node_inner` 创建的**计算节点**会自动打上分组标签。
/// Input / Parameter / TargetInput 等数据节点不受影响。
///
/// 采用"外层优先"策略：如果已有活跃的分组上下文（如 TanhNormal 调用 Normal），
/// 内层 guard 不会覆盖外层标签，确保所有节点统一归属到最外层分布。
///
/// # 示例
/// ```ignore
/// let _guard = NodeGroupContext::new(&logits, "Categorical", instance_id);
/// let probs = logits.softmax();       // 自动标记为 Categorical
/// let log_probs = logits.log_softmax(); // 自动标记为 Categorical
/// // guard drop 后，新节点不再标记
/// ```
pub(crate) struct NodeGroupContext {
    graph: Rc<RefCell<GraphInner>>,
    /// 是否真正 push 了上下文（外层优先：已有上下文时为 false）
    did_push: bool,
    /// 之前的 include_params 值（drop 时恢复）
    prev_include_params: bool,
}

impl NodeGroupContext {
    /// Distribution 用：创建分组上下文（不标记 Parameter/Input 节点）
    ///
    /// # 参数
    /// - `var`: 任意一个属于目标 Graph 的 Var（用于获取图引用）
    /// - `group_type`: 分组类型名（如 "Categorical"）
    /// - `instance_id`: 实例 ID（由 `GraphInner::next_node_group_instance_id()` 获取）
    pub fn new(var: &Var, group_type: &str, instance_id: usize) -> Self {
        Self::create(
            var,
            group_type,
            instance_id,
            None,
            None,
            types::GroupStyle::Distribution,
            false, // 不标记 Parameter 节点
        )
    }

    /// Layer 用：创建分组上下文（标记 Parameter 节点，含名称/描述）
    ///
    /// 与 `new` 的区别：
    /// - Parameter 节点也会被自动标记（`include_params = true`）
    /// - 携带 `display_name` 和 `description`（用于可视化 cluster 标签）
    pub fn for_layer(
        var: &Var,
        layer_type: &str,
        instance_id: usize,
        qualified_name: &str,
        description: &str,
    ) -> Self {
        Self::create(
            var,
            layer_type,
            instance_id,
            Some(qualified_name),
            Some(description),
            types::GroupStyle::Layer,
            true, // 标记 Parameter 节点
        )
    }

    /// Recurrent 层用：创建分组上下文（标记 Parameter 节点，三重边框样式）
    pub fn for_recurrent(
        var: &Var,
        layer_type: &str,
        instance_id: usize,
        qualified_name: &str,
        description: &str,
    ) -> Self {
        Self::create(
            var,
            layer_type,
            instance_id,
            Some(qualified_name),
            Some(description),
            types::GroupStyle::Recurrent,
            true, // 标记 Parameter 节点
        )
    }

    /// 将已有节点纳入当前分组（用于在 guard 之前创建的 Parameter / 初始状态等节点）
    pub fn tag_existing(&self, var: &Var) {
        let tag = self.graph.borrow().node_group_context.clone();
        if let Some(tag) = tag {
            var.node().set_node_group_tag(Some(tag));
        }
    }

    /// 切换后续自动标记的 hidden 标志（RNN 步骤 1..N-1 时调用）
    pub fn set_hidden(&self, hidden: bool) {
        let mut g = self.graph.borrow_mut();
        if let Some(ref mut tag) = g.node_group_context {
            tag.hidden = hidden;
        }
    }

    // ==================== 内部实现 ====================

    fn create(
        var: &Var,
        group_type: &str,
        instance_id: usize,
        display_name: Option<&str>,
        description: Option<&str>,
        style: types::GroupStyle,
        include_params: bool,
    ) -> Self {
        let graph = var.graph();
        let (did_push, prev_include_params) = {
            let mut g = graph.borrow_mut();
            let prev = g.node_group_include_params;
            if g.node_group_context.is_none() {
                g.node_group_context = Some(NodeGroupTag {
                    group_type: group_type.to_string(),
                    instance_id,
                    display_name: display_name.map(|s| s.to_string()),
                    description: description.map(|s| s.to_string()),
                    style,
                    hidden: false,
                });
                g.node_group_include_params = include_params;
                (true, prev)
            } else {
                (false, prev) // 已在外层分组上下文中，不覆盖
            }
        };
        Self {
            graph,
            did_push,
            prev_include_params,
        }
    }
}

impl Drop for NodeGroupContext {
    fn drop(&mut self) {
        if self.did_push {
            let mut g = self.graph.borrow_mut();
            g.node_group_context = None;
            g.node_group_include_params = self.prev_include_params;
        }
    }
}
