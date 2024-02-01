/*
 * @Author       : 老董
 * @Date         : 2024-01-31 17:57:13
 * @LastEditors  : 老董
 * @LastEditTime : 2024-01-31 20:43:32
 * @Description  : 神经网络模型的计算图
 */

use super::nodes::{NodeEnum, TraitForNode};

pub struct Graph {
    nodes: Vec<NodeEnum>,
    // name_scope: Option<String>, // TODO: need?
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            // name_scope: None,
        }
    }

    pub fn add_node(&mut self, node: &NodeEnum) {
        self.nodes.push(node.clone()); // TODO: store ref or clone?
    }

    pub fn clear_jacobi(&mut self) {
        for node in &mut self.nodes {
            node.clear_jacobi();
        }
    }

    pub fn reset_value(&mut self) {
        for node in &mut self.nodes {
            node.reset_value(false);
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓默认计算图↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
use std::cell::RefCell;
thread_local!(pub(crate) static DEFAULT_GRAPH: RefCell<Graph> = RefCell::new(Graph::new()));
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑默认计算图↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
