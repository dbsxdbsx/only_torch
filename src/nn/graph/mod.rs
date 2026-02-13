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

mod error;
mod handle;
mod inner;
mod types;

pub use error::{GraphError, ImageFormat, VisualizationOutput};
pub use handle::Graph;
pub use inner::GraphInner;
// 这些类型用于可视化分组，作为公共 API 导出供外部使用
#[allow(unused_imports)]
pub use types::{
    GroupKind, LayerGroup, NodeGroupTag, RecurrentLayerMeta, RecurrentUnrollInfo, SnapshotNode,
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
}

impl NodeGroupContext {
    /// 创建分组上下文
    ///
    /// # 参数
    /// - `var`: 任意一个属于目标 Graph 的 Var（用于获取图引用）
    /// - `group_type`: 分组类型名（如 "Categorical"）
    /// - `instance_id`: 实例 ID（由 `GraphInner::next_node_group_instance_id()` 获取）
    pub fn new(var: &Var, group_type: &str, instance_id: usize) -> Self {
        let graph = var.graph();
        let did_push = {
            let mut g = graph.borrow_mut();
            if g.node_group_context.is_none() {
                g.node_group_context = Some(NodeGroupTag {
                    group_type: group_type.to_string(),
                    instance_id,
                });
                true
            } else {
                false // 已在外层分组上下文中，不覆盖
            }
        };
        Self { graph, did_push }
    }
}

impl Drop for NodeGroupContext {
    fn drop(&mut self) {
        if self.did_push {
            self.graph.borrow_mut().node_group_context = None;
        }
    }
}
