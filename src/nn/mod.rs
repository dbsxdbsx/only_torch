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
pub mod distributions;
pub mod evolution;
mod graph;
pub mod layer;
// mod model_state; // 已移除
mod module;
mod nodes;
pub mod optimizer;
mod shape;
mod var;

// pub use criterion::{...}; // 已移除
pub use descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor};
pub(in crate::nn) use display::format_node_display;
pub use graph::{
    Graph, GraphError, GraphInner, ImageFormat, OnnxError, RebuildResult, SnapshotNode,
    VisualizationOutput, VisualizationSnapshot,
};
// ONNX 导入路径的可观测性入口（让用户在 from_onnx rebuild 失败时仍能拿 ImportReport）
pub use graph::onnx_import::{
    ImportReport, OnnxImportResult, RewriteRecord, load_onnx, load_onnx_from_bytes,
};
pub use layer::{
    AvgPool2d, BatchNorm, Conv2d, Embedding, GroupNorm, Gru, InstanceNorm, LayerNorm, Linear, Lstm,
    MaxPool2d, MultiHeadAttention, RMSNorm, Rnn,
};
// pub use model_state::{ForwardInput, ForwardOutput, ModelState}; // 已移除
pub use module::Module;
pub use nodes::NodeId;
pub use nodes::node_inner::NodeInner; // 供内部模块使用
pub use nodes::raw_node::{DEFAULT_DROPOUT_P, Reduction};
pub use optimizer::{Adam, CosineAnnealingLR, LambdaLR, LrScheduler, Optimizer, SGD, StepLR};
pub use shape::{Dim, DynamicShape};
pub use var::ops::{
    GatherIndex, VarActivationOps, VarFilterOps, VarLossOps, VarMatrixOps, VarReduceOps,
    VarRegularizationOps, VarSelectionOps, VarShapeOps,
};
pub use var::{Init, IntoVar, Var};

#[cfg(test)]
mod tests;
