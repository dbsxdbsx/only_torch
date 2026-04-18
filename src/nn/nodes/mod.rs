// 新代码使用 NodeInner；旧的 NodeHandle / node_handle 已移除。
pub mod node_inner;
pub mod raw_node;

pub use super::GraphError;
pub use node_inner::NodeInner;
pub use raw_node::*;

/// 节点唯一标识符
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(in crate::nn) u64);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
