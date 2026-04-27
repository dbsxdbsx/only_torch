//! Resize / Upsample 节点常量折叠
//!
//! ONNX Resize/Upsample 的 scales 输入(`input[2]`)是 f32 静态常量时,直接折叠
//! 到内部 `NodeTypeDescriptor::Upsample2d::scale_h/scale_w` 整数倍属性。
//!
//! 仅支持 NCHW 4 维 + 整数倍 nearest 模式(YOLOv5 默认模式)。

use std::collections::HashMap;

use crate::nn::descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor};
use crate::nn::graph::onnx_error::OnnxError;
use crate::nn::graph::onnx_ops;
use onnx_rs::ast::TensorProto;

use super::util::{SymbolTable, extract_const_f32, infer_output_shape_placeholder};
use super::{ImportReport, RewriteRecord};

/// 装配 Resize/Upsample 节点：从常量表读取 scales/sizes,折叠到 `Upsample2d::scale_h/scale_w`
pub(super) fn assemble_resize_with_const_fold<'a>(
    node: &onnx_rs::ast::Node<'a>,
    const_table: &HashMap<&'a str, &'a TensorProto<'a>>,
    symbols: &mut SymbolTable,
    descriptor: &mut GraphDescriptor,
    import_report: &mut ImportReport,
) -> Result<(), OnnxError> {
    // mode 校验委托给 onnx_op_to_descriptors(已检查 mode="nearest")
    let _ = onnx_ops::onnx_op_to_descriptors(&node.op_type, &node.attribute, node.name)?;

    let data_name = node.input[0];
    let op_ctx = format!("Resize 节点 \"{}\"", node.name);

    // 优先用 scales(input[2],f32),不存在则用 sizes(input[3],i64)反推
    let (scale_h, scale_w, source_const) = if node.input.len() >= 3 && !node.input[2].is_empty() {
        let scales_name = node.input[2];
        let scales = extract_const_f32(const_table, scales_name, &op_ctx)?;
        if scales.len() != 4 {
            return Err(OnnxError::UnsupportedAttribute {
                op_type: "Resize".to_string(),
                attribute: "scales".to_string(),
                reason: format!(
                    "{op_ctx}: 仅支持 4 维 NCHW(scales.len=4),实际 {}",
                    scales.len()
                ),
            });
        }
        // 通常 N、C 维 scale=1.0,H/W 维是上采样倍数
        for (i, &s) in scales.iter().enumerate().take(2) {
            if (s - 1.0).abs() > 1e-6 {
                return Err(OnnxError::UnsupportedAttribute {
                    op_type: "Resize".to_string(),
                    attribute: format!("scales[{i}]"),
                    reason: format!("{op_ctx}: N/C 维 scale 必须=1.0,得到 {s}"),
                });
            }
        }
        let sh = scales[2];
        let sw = scales[3];
        if sh.fract().abs() > 1e-6 || sw.fract().abs() > 1e-6 || sh < 1.0 || sw < 1.0 {
            return Err(OnnxError::UnsupportedAttribute {
                op_type: "Resize".to_string(),
                attribute: "scales".to_string(),
                reason: format!("{op_ctx}: 仅支持整数倍上采样(scale≥1,nearest),得到 H={sh} W={sw}"),
            });
        }
        (sh as usize, sw as usize, scales_name.to_string())
    } else if node.input.len() >= 4 && !node.input[3].is_empty() {
        // sizes 路径：从输出/输入 shape 反推 scale。本轮最小骨架不支持,
        // 用户可改用 scales 形式或用 onnxsim 转换
        return Err(OnnxError::UnsupportedAttribute {
            op_type: "Resize".to_string(),
            attribute: "sizes".to_string(),
            reason: format!("{op_ctx}: 仅支持 scales 形式,sizes 形式请用 onnxsim 转换"),
        });
    } else {
        return Err(OnnxError::UnsupportedAttribute {
            op_type: "Resize".to_string(),
            attribute: "scales/sizes".to_string(),
            reason: format!("{op_ctx}: 必须提供 scales 或 sizes 输入"),
        });
    };

    let parent_id = symbols.get_or_assign(data_name);
    let output_name = node.output.first().copied().unwrap_or(node.name);
    let out_id = symbols.get_or_assign(output_name);
    let output_shape = infer_output_shape_placeholder(
        &NodeTypeDescriptor::Upsample2d { scale_h, scale_w },
        &[parent_id],
        descriptor,
    );

    descriptor.add_node(
        NodeDescriptor::new(
            out_id,
            output_name,
            NodeTypeDescriptor::Upsample2d { scale_h, scale_w },
            output_shape,
            None,
            vec![parent_id],
        )
        .with_origin_onnx_nodes(vec![
            node.name.to_string(),
            format!("<const:{source_const}>"),
        ]),
    );

    import_report.rewritten.push(RewriteRecord {
        pattern: "constant_fold_into_resize",
        consumed_onnx_nodes: vec![node.name.to_string(), format!("<const:{source_const}>")],
        produced_descriptor_nodes: vec![out_id],
    });
    // warning: 标记折叠成 nearest Upsample2d,提示语义可能与原 ONNX 不完全一致
    // (例如 coordinate_transformation_mode/nearest_mode 的细微差异在整数倍场景可忽略)
    import_report.warnings.push(format!(
        "Resize \"{}\" 折叠为 Upsample2d (scale_h={scale_h}, scale_w={scale_w}, mode=nearest);\
         coordinate_transformation_mode/nearest_mode 子模式差异在整数倍场景可忽略",
        node.name
    ));
    Ok(())
}
