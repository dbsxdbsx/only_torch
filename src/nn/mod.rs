/*
 * @Author       : 老董
 * @Date         : 2024-01-31 20:23:53
 * @LastEditors  : 老董
 * @LastEditTime : 2024-12-20 19:19:32
 * @Description  : 负责神经网络（neural network）的构建
 */

pub mod graph;
pub mod nodes;

pub use graph::{Graph, GraphError};
pub use nodes::{NodeHandle, NodeId};
