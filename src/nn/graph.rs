/*
 * @Author       : 老董
 * @Date         : 2024-01-31 17:57:13
 * @LastEditors  : 老董
 * @LastEditTime : 2024-10-20 11:24:08
 * @Description  : 神经网络模型的计算图
 */

use super::nodes::{NodeEnum, TraitForNode};
use core::panic;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub struct Graph {
    nodes: HashMap<String, Rc<RefCell<NodeEnum>>>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: NodeEnum) {
        let name = node.name().to_string();
        self.nodes.insert(name, Rc::new(RefCell::new(node)));
    }

    pub fn get_node(&self, name: &str) -> Option<Rc<RefCell<NodeEnum>>> {
        self.nodes.get(name).cloned()
    }

    pub fn clear_jacobi(&mut self) {
        for node in self.nodes.values_mut() {
            node.borrow_mut().clear_jacobi();
        }
    }

    pub fn reset_value(&mut self) {
        for node in self.nodes.values_mut() {
            node.borrow_mut().reset_value(false);
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓默认计算图↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
thread_local!(pub(crate) static DEFAULT_GRAPH: RefCell<Graph> = RefCell::new(Graph::new()));
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑默认计算图↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

pub(crate) fn convert_parents(parents: &[String]) -> Vec<Rc<RefCell<NodeEnum>>> {
    parents
        .iter()
        .filter_map(|name| get_node_from_default_graph(name))
        .collect()
}

pub(crate) fn get_node_from_default_graph(name: &str) -> Option<Rc<RefCell<NodeEnum>>> {
    DEFAULT_GRAPH.with(|graph| graph.borrow().get_node(name))
}

pub(crate) fn add_node_to_default_graph(node: &NodeEnum) {
    DEFAULT_GRAPH.with(|graph| {
        let mut graph = graph.borrow_mut();
        let name = node.name().to_string();
        if graph.nodes.contains_key(&name) {
            panic!("在添加节点`{}`时，发现该节点已存在", name);
        } else {
            graph.add_node(node.clone());
        }
    });
}

pub(crate) fn update_node_in_default_graph(node: &NodeEnum) {
    DEFAULT_GRAPH.with(|graph| {
        let graph = graph.borrow();
        if let Some(existing_node) = graph.get_node(node.name()) {
            *existing_node.borrow_mut() = node.clone();
        } else {
            panic!("在更新节点`{}`时，发现该节点不存在", node.name());
        }
    });
}

pub(crate) fn generate_unique_name(prefix: &str) -> String {
    let mut max_index = 0;

    DEFAULT_GRAPH.with(|graph| {
        let graph = graph.borrow();
        for node_name in graph.nodes.keys() {
            if node_name.starts_with(prefix) {
                if let Some(index_str) = node_name
                    .strip_prefix(prefix)
                    .and_then(|s| s.strip_prefix('_'))
                {
                    if let Ok(index) = index_str.parse::<usize>() {
                        max_index = max_index.max(index);
                    }
                }
            }
        }
    });

    format!("{}_{}", prefix, max_index + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_unique_name_for_node() {
        use crate::nn::nodes::Variable;

        // 创建多个Variable节点，不指定名称
        let v1 = Variable::new(&[2, 3], false, false, None);
        let v2 = Variable::new(&[2, 3], false, false, None);
        DEFAULT_GRAPH.with(|graph| {
            let graph = graph.borrow();
            for node in graph.nodes.values() {
                println!("hi: {}", node.borrow().name());
            }
        });
        let v3 = Variable::new(&[2, 3], false, false, None);

        // 验证生成的名称是否正确
        assert_eq!(v1.name(), "<default>_variable_1");
        assert_eq!(v2.name(), "<default>_variable_2");
        assert_eq!(v3.name(), "<default>_variable_3");
    }
}
