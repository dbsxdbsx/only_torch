//! Split → N×Narrow 重写
//!
//! ONNX Split 把一个张量沿 axis 切成 N 段。only_torch 没有 Split 算子,
//! 但 Narrow(axis, start, length) 完全可以表达——一个 Split 重写为 N 个
//! 不重叠的 Narrow 节点,共享同一个父输入。
//!
//! `split_sizes` 来源优先级：
//! 1. opset 12 及以下：attribute "split"(Vec<i64>)
//! 2. opset 13 及以上：input[1] 为常量 i64 张量
//! 3. 都没有：要求 axis 维度均匀 N 等分(暂不支持,需要 input shape 信息,本轮报错)

use std::collections::HashMap;

use crate::nn::descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor};
use crate::nn::graph::onnx_error::OnnxError;
use crate::nn::graph::onnx_ops;
use onnx_rs::ast::TensorProto;

use super::util::{extract_const_i64, SymbolTable};
use super::{ImportReport, RewriteRecord};

/// 装配 Split 节点：展开为 N 个 Narrow 节点
pub(super) fn assemble_split_to_narrows<'a>(
    node: &onnx_rs::ast::Node<'a>,
    const_table: &HashMap<&'a str, &'a TensorProto<'a>>,
    symbols: &mut SymbolTable,
    descriptor: &mut GraphDescriptor,
    import_report: &mut ImportReport,
) -> Result<(), OnnxError> {
    if node.input.is_empty() {
        return Err(OnnxError::InvalidGraph(format!(
            "Split 节点 \"{}\" 缺少输入",
            node.name
        )));
    }
    let data_name = node.input[0];
    let axis = onnx_ops::find_attr_int(&node.attribute, "axis").unwrap_or(0);
    if axis < 0 {
        return Err(OnnxError::UnsupportedAttribute {
            op_type: "Split".to_string(),
            attribute: "axis".to_string(),
            reason: format!(
                "Split 节点 \"{}\": 仅支持非负 axis,得到 {axis}(请用 onnxsim 规范化)",
                node.name
            ),
        });
    }
    let axis = axis as usize;

    // split_sizes 提取
    let split_sizes_i64: Vec<i64> = if node.input.len() >= 2 && !node.input[1].is_empty() {
        // opset 13+：来自 input[1] 常量
        extract_const_i64(
            const_table,
            node.input[1],
            &format!("Split 节点 \"{}\"", node.name),
        )?
    } else {
        // opset ≤ 12：来自 attribute "split"
        let split_attr = onnx_ops::find_attr_ints(&node.attribute, "split");
        if split_attr.is_empty() {
            return Err(OnnxError::UnsupportedAttribute {
                op_type: "Split".to_string(),
                attribute: "split".to_string(),
                reason: format!(
                    "Split 节点 \"{}\": split 既无 input 也无 attribute,等分模式需 input shape 信息,\
                    本版本暂不支持。请用 onnxsim 显式指定 split_sizes",
                    node.name
                ),
            });
        }
        split_attr
    };

    if split_sizes_i64.is_empty() {
        return Err(OnnxError::UnsupportedAttribute {
            op_type: "Split".to_string(),
            attribute: "split".to_string(),
            reason: format!("Split 节点 \"{}\": split_sizes 为空", node.name),
        });
    }
    if split_sizes_i64.iter().any(|&s| s <= 0) {
        return Err(OnnxError::UnsupportedAttribute {
            op_type: "Split".to_string(),
            attribute: "split".to_string(),
            reason: format!(
                "Split 节点 \"{}\": split_sizes 含非正值 {:?}",
                node.name, split_sizes_i64
            ),
        });
    }
    let split_sizes: Vec<usize> = split_sizes_i64.iter().map(|&s| s as usize).collect();

    if split_sizes.len() != node.output.len() {
        return Err(OnnxError::InvalidGraph(format!(
            "Split 节点 \"{}\": split_sizes 长度 {} 与输出数 {} 不一致",
            node.name,
            split_sizes.len(),
            node.output.len()
        )));
    }

    let parent_id = symbols.get_or_assign(data_name);
    let mut produced_ids = Vec::with_capacity(split_sizes.len());
    let mut start = 0usize;
    for (i, &length) in split_sizes.iter().enumerate() {
        let out_name = node.output[i];
        let out_id = symbols.get_or_assign(out_name);
        let parent_shape = descriptor
            .nodes
            .iter()
            .find(|n| n.id == parent_id)
            .map(|n| n.output_shape.clone())
            .unwrap_or_default();
        // 占位输出形状：把 axis 维度替换为 length
        let mut out_shape = parent_shape;
        if axis < out_shape.len() {
            out_shape[axis] = length;
        }
        descriptor.add_node(NodeDescriptor::new(
            out_id,
            out_name,
            NodeTypeDescriptor::Narrow {
                axis,
                start,
                length,
            },
            out_shape,
            None,
            vec![parent_id],
        ));
        produced_ids.push(out_id);
        start += length;
    }

    import_report.rewritten.push(RewriteRecord {
        pattern: "split_to_narrows",
        consumed_onnx_nodes: vec![node.name.to_string()],
        produced_descriptor_nodes: produced_ids,
    });
    Ok(())
}
