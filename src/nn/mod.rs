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
mod module;
mod nodes;
pub mod optimizer;
mod var;
mod var_ops;

pub use descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor};
pub(in crate::nn) use display::format_node_display;
pub use graph::{Graph, GraphError, GraphHandle, GraphInner, ImageFormat, VisualizationOutput};
pub use layer::{Linear, LinearOutput, linear};
pub use module::Module;
pub use nodes::NodeId;
pub use nodes::raw_node::Reduction;
pub use optimizer::{Adamv2, OptimizerV2, SGDv2};
pub use var::{Init, Var};
pub use var_ops::{VarActivationOps, VarLossOps, VarMatrixOps, VarShapeOps};

#[cfg(test)]
mod tests;
