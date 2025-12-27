/*
 * @Author       : 老董
 * @Date         : 2024-01-31 20:23:53
 * @LastEditors  : 老董
 * @LastEditTime : 2025-01-04 19:37:27
 * @Description  : 负责神经网络（neural network）的构建
 */

mod descriptor;
mod display;
mod graph;
pub mod layer;
mod nodes;
pub mod optimizer;

pub use descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor};
pub(in crate::nn) use display::format_node_display;
pub use graph::{Graph, GraphError, ImageFormat, VisualizationOutput};
pub use layer::{LinearOutput, linear};
pub use nodes::NodeId;
pub use nodes::raw_node::Reduction;

#[cfg(test)]
mod tests;
