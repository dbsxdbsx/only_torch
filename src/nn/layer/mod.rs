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
mod linear;
mod max_pool2d;

pub use avg_pool2d::{avg_pool2d, AvgPool2dOutput};
pub use conv2d::{conv2d, Conv2dOutput};
pub use linear::{linear, LinearOutput};
pub use max_pool2d::{max_pool2d, MaxPool2dOutput};
