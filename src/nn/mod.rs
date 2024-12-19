/*
 * @Author       : 老董
 * @Date         : 2024-01-31 20:23:53
 * @LastEditors  : 老董
 * @LastEditTime : 2024-12-19 10:50:51
 * @Description  : 负责神经网络（neural network）的构建
 */

pub mod graph;
pub mod nodes;

// Re-export commonly used types
pub use graph::{Graph, GraphError};
pub use nodes::{NodeHandle, NodeId, TraitNode};
