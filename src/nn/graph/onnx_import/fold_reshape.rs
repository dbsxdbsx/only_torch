//! Reshape 节点常量折叠
//!
//! ONNX Reshape 的 shape 输入(`input[1]`)在常见场景里是常量(initializer 或
//! Constant 节点输出)。本模块从常量表读出 shape 值,推导 -1/0 这两个特殊维度,
//! 折叠到内部 `NodeTypeDescriptor::Reshape::target_shape` 属性。

use std::collections::HashMap;

use crate::nn::descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor};
use crate::nn::graph::onnx_error::OnnxError;
use onnx_rs::ast::TensorProto;

use super::util::{extract_const_i64, SymbolTable};
use super::{ImportReport, RewriteRecord};

/// 装配 Reshape 节点：从常量表读取 shape 输入,折叠到 `Reshape::target_shape`
pub(super) fn assemble_reshape_with_const_fold<'a>(
    node: &onnx_rs::ast::Node<'a>,
    const_table: &HashMap<&'a str, &'a TensorProto<'a>>,
    symbols: &mut SymbolTable,
    descriptor: &mut GraphDescriptor,
    import_report: &mut ImportReport,
) -> Result<(), OnnxError> {
    let data_name = node.input[0];
    let shape_name = node.input[1];

    let shape_i64 = extract_const_i64(
        const_table,
        shape_name,
        &format!("Reshape 节点 \"{}\"", node.name),
    )?;

    let parent_id = symbols.get_or_assign(data_name);
    // 查 parent 形状供 -1/0 推导(descriptor 中找不到时给空 slice,意味着无法推导)
    let parent_shape: Vec<usize> = descriptor
        .nodes
        .iter()
        .find(|n| n.id == parent_id)
        .map(|n| n.output_shape.clone())
        .unwrap_or_default();
    let target_shape = convert_onnx_shape_to_usize(
        &shape_i64,
        &parent_shape,
        &format!("Reshape 节点 \"{}\"", node.name),
    )?;
    let output_name = node.output.first().copied().unwrap_or(node.name);
    let out_id = symbols.get_or_assign(output_name);

    descriptor.add_node(
        NodeDescriptor::new(
            out_id,
            output_name,
            NodeTypeDescriptor::Reshape { target_shape: target_shape.clone() },
            target_shape,
            None,
            vec![parent_id],
        )
        .with_origin_onnx_nodes(vec![
            node.name.to_string(),
            format!("<const:{shape_name}>"),
        ]),
    );

    import_report.rewritten.push(RewriteRecord {
        pattern: "constant_fold_into_reshape",
        consumed_onnx_nodes: vec![
            node.name.to_string(),
            format!("<const:{shape_name}>"),
        ],
        produced_descriptor_nodes: vec![out_id],
    });
    Ok(())
}

/// 把 ONNX 风格的 i64 shape 转为 only_torch 的 `Vec<usize>`
///
/// ONNX shape 允许两种特殊值：
/// - `-1`(推导)：根据 `parent.numel() / 其他维度乘积` 算出,最多一个
/// - `0`(保留)：复用 `parent_shape` 对应位置的维度
///
/// 当 parent 形状中有动态维度(=0 表示未知 batch 等)时,-1/0 的推导可能失败,
/// 此时回退为 1 占位,下游 `Reshape::new` 会按动态 batch 机制处理。
pub(super) fn convert_onnx_shape_to_usize(
    raw: &[i64],
    parent_shape: &[usize],
    op_context: &str,
) -> Result<Vec<usize>, OnnxError> {
    // 第 1 趟：扫描 + 校验,统计 -1 出现次数 + 收集已知维度乘积
    let mut neg_one_idx: Option<usize> = None;
    let mut known_product: usize = 1;
    let mut has_unknown_factor = false;
    for (i, &d) in raw.iter().enumerate() {
        match d {
            -1 => {
                if neg_one_idx.is_some() {
                    return Err(OnnxError::UnsupportedAttribute {
                        op_type: op_context.to_string(),
                        attribute: "shape".to_string(),
                        reason: format!(
                            "ONNX shape {raw:?} 含多个 -1,违反 ONNX 规范"
                        ),
                    });
                }
                neg_one_idx = Some(i);
            }
            0 => {
                if i >= parent_shape.len() {
                    return Err(OnnxError::UnsupportedAttribute {
                        op_type: op_context.to_string(),
                        attribute: "shape".to_string(),
                        reason: format!(
                            "ONNX shape {raw:?} 第 {i} 维为 0(保留),但 parent 仅 {} 维",
                            parent_shape.len()
                        ),
                    });
                }
                let v = parent_shape[i];
                if v == 0 {
                    has_unknown_factor = true;
                } else {
                    known_product = known_product.saturating_mul(v);
                }
            }
            d if d > 0 => {
                known_product = known_product.saturating_mul(d as usize);
            }
            _ => {
                return Err(OnnxError::UnsupportedAttribute {
                    op_type: op_context.to_string(),
                    attribute: "shape".to_string(),
                    reason: format!("ONNX shape 含非法值 {d}"),
                });
            }
        }
    }

    // 第 2 趟：推导 -1 对应的值
    // 推导优先级：
    //   1. parent_shape 全静态 + known_product>0 → total / known
    //   2. parent_shape 含动态维度 → 占位 1(让 Reshape 动态 batch 接管)
    let parent_total: Option<usize> = if parent_shape.iter().any(|&d| d == 0) {
        None
    } else {
        Some(parent_shape.iter().product())
    };
    let inferred_neg_one = match (neg_one_idx, parent_total) {
        (Some(_), Some(total)) if known_product > 0 && total % known_product == 0 => {
            total / known_product
        }
        (Some(_), _) => {
            // 无法精确推导 → 占位 1(动态 batch 走运行时调整)
            // has_unknown_factor 用于说明不静态可推导是因为 parent 含 0
            let _ = has_unknown_factor;
            1
        }
        (None, _) => 1, // 没 -1,占位无意义
    };

    // 第 3 趟：构造输出
    let mut out = Vec::with_capacity(raw.len());
    for (i, &d) in raw.iter().enumerate() {
        out.push(match d {
            -1 => inferred_neg_one,
            0 => parent_shape[i],
            d if d > 0 => d as usize,
            _ => unreachable!("已在第 1 趟拦截"),
        });
    }
    Ok(out)
}
