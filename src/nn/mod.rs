/*
 * @Author       : 老董
 * @Date         : 2024-01-31 20:23:53
 * @LastEditors  : 老董
 * @LastEditTime : 2026-02-01 00:00:00
 * @Description  : 负责神经网络（neural network）的构建
 */

// mod criterion; // 已移除，统一用 Var 方法（如 mse_loss()）
pub mod debug;
mod descriptor;
mod display;
mod graph;
pub mod layer;
// mod model_state; // 已移除
mod module;
mod nodes;
pub mod optimizer;
mod shape;
mod var;
mod var_ops;

// pub use criterion::{...}; // 已移除
pub use descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor};
pub(in crate::nn) use display::format_node_display;
pub use graph::{Graph, GraphError, GraphInner, ImageFormat, VisualizationOutput};
pub use layer::{AvgPool2d, Conv2d, Gru, Linear, Lstm, MaxPool2d, Rnn};
// pub use model_state::{ForwardInput, ForwardOutput, ModelState}; // 已移除
pub use module::Module;
pub use nodes::NodeId;
pub use nodes::node_inner::NodeInner; // 供内部模块使用
pub use nodes::raw_node::{DEFAULT_DROPOUT_P, Reduction};
pub use optimizer::{Adam, Optimizer, SGD};
pub use shape::{Dim, DynamicShape};
pub use var::{Init, IntoVar, Var};
pub use var_ops::{
    GatherIndex, VarActivationOps, VarLossOps, VarMatrixOps, VarReduceOps, VarRegularizationOps,
    VarShapeOps,
};

#[cfg(test)]
mod tests;
