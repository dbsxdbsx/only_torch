//! # 常用接口模块
//!
//! 本模块提供一些常用的操作接口

use crate::nn::graph::DEFAULT_GRAPH;
use crate::nn::nodes::NodeEnum;

#[cfg(test)]
mod tests;

pub mod macro_for_unit_test;

// TODO: move traits to utils/traits relatively
pub mod traits {
    pub mod float;
    pub mod image;
}

pub(crate) fn add_node_to_default_graph(node: &NodeEnum) {
    DEFAULT_GRAPH.with(|graph| {
        let mut graph = graph.borrow_mut();
        graph.add_node(node);
    });
}
