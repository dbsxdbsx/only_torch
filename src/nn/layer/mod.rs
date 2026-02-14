/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : Layer 模块 - 便捷函数，组合 Node 构建常见网络结构
 *
 * Layer 不是新的抽象层，只是语法糖！
 * 详见 .doc/design/node_vs_layer_design.md
 */

mod attention;
mod avg_pool2d;
mod batch_norm;
mod conv2d;
mod embedding;
mod group_norm;
mod gru;
mod instance_norm;
mod layer_norm;
mod linear;
mod lstm;
mod max_pool2d;
mod rms_norm;
mod rnn;

pub use attention::MultiHeadAttention;
pub use avg_pool2d::AvgPool2d;
pub use batch_norm::BatchNorm;
pub use conv2d::Conv2d;
pub use embedding::Embedding;
pub use group_norm::GroupNorm;
pub use gru::Gru;
pub use instance_norm::InstanceNorm;
pub use layer_norm::LayerNorm;
pub use linear::Linear;
pub use lstm::Lstm;
pub use max_pool2d::MaxPool2d;
pub use rms_norm::RMSNorm;
pub use rnn::Rnn;
