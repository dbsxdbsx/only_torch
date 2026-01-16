/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : Layer 模块 - 便捷函数，组合 Node 构建常见网络结构
 *
 * Layer 不是新的抽象层，只是语法糖！
 * 详见 .doc/design/node_vs_layer_design.md
 */

mod avg_pool2d;
mod conv2d;
mod gru;
mod linear;
mod linear_v2;
mod lstm;
mod max_pool2d;
mod rnn;

pub use avg_pool2d::{AvgPool2dOutput, avg_pool2d};
pub use conv2d::{Conv2dOutput, conv2d};
pub use gru::{GruOutput, gru};
pub use linear::{LinearOutput, linear};
pub use linear_v2::Linear;
pub use lstm::{LstmOutput, lstm};
pub use max_pool2d::{MaxPool2dOutput, max_pool2d};
pub use rnn::{RnnOutput, rnn};
