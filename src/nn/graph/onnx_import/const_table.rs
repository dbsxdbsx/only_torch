//! 常量收集 + 元信息消费标记
//!
//! ONNX Constant 节点和 initializer 都可能携带数值常量。本模块在装配前预扫描整张
//! 图,把它们汇总到一个 `name → TensorProto` 表里,供后续折叠路径(Reshape/Resize/
//! Split 等)按 input name 直接查值,而不必走运行时常量节点。
//!
//! 同时标记哪些常量是被算子用作"元信息"消费的(如 Reshape.shape、Resize.scales、
//! Split.split),这些常量来自 initializer 时不应再被建成 Parameter 节点暴露给运行时。

use std::collections::{HashMap, HashSet};

use onnx_rs::ast::{OpType, TensorProto};

/// 收集 Constant 节点输出 + initializer 到统一查找表
///
/// 当同名时优先保留 Constant 节点的 value(因为它在 ONNX 模型里通常更明确)。
pub(super) fn build_const_table<'a>(
    graph: &'a onnx_rs::ast::Graph<'a>,
) -> HashMap<&'a str, &'a TensorProto<'a>> {
    let mut const_table: HashMap<&str, &TensorProto> = HashMap::new();

    for node in &graph.node {
        if node.op_type == OpType::Constant {
            // ONNX Constant 通过 attribute "value": TensorProto 携带常量值
            if let Some(attr) = node.attribute.iter().find(|a| a.name == "value") {
                if let Some(tensor) = &attr.t {
                    if let Some(&out_name) = node.output.first() {
                        const_table.insert(out_name, tensor);
                    }
                }
            }
        }
    }
    for init in &graph.initializer {
        const_table.entry(init.name()).or_insert(init);
    }
    const_table
}

/// 标记哪些常量被算子用作"元信息"(shape / scales / split_sizes 等)
///
/// 这些常量对应的 initializer 在装配阶段不应被建成 Parameter 节点。
/// Constant 节点的输出名同样会落在此集合,装配阶段决定是跳过还是建 Parameter
/// (取决于是否被消费)。
pub(super) fn build_consumed_meta_names<'a>(
    graph: &'a onnx_rs::ast::Graph<'a>,
) -> HashSet<&'a str> {
    let mut consumed: HashSet<&str> = HashSet::new();
    for node in &graph.node {
        match node.op_type {
            OpType::Reshape if node.input.len() >= 2 && !node.input[1].is_empty() => {
                consumed.insert(node.input[1]);
            }
            OpType::Resize | OpType::Upsample => {
                // ONNX Resize 输入布局：[X, roi, scales, sizes]
                // YOLOv5 通常 roi 为空、scales 或 sizes 二选一
                if node.input.len() >= 3 && !node.input[2].is_empty() {
                    consumed.insert(node.input[2]);
                }
                if node.input.len() >= 4 && !node.input[3].is_empty() {
                    consumed.insert(node.input[3]);
                }
            }
            OpType::Split if node.input.len() >= 2 && !node.input[1].is_empty() => {
                consumed.insert(node.input[1]);
            }
            _ => {}
        }
    }
    consumed
}
