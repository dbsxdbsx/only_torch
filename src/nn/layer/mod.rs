/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : Layer 模块 - 便捷函数，组合 Node 构建常见网络结构
 *
 * Layer 不是新的抽象层，只是语法糖！
 * 详见 .doc/design/node_vs_layer_design.md
 */

mod linear;

pub use linear::{LinearOutput, linear};
