//! 装配主循环
//!
//! 负责把 ONNX 计算图按拓扑顺序装配为 `GraphDescriptor` + 权重表,过程中:
//! - 普通 1:1 算子直接映射到 NodeTypeDescriptor
//! - Conv/ConvTranspose with bias(3 输入) → 拆为 Conv + Add 双节点
//! - Gemm → 拆为 MatMul + Add 双节点
//! - Reshape / Resize / Split / Constant → 委托给对应子模块的 special path
//!
//! 重写过程会向 `ImportReport` 追加可观测记录(rewrite + warnings)。

use std::collections::{HashMap, HashSet};

use crate::nn::descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor};
use crate::nn::graph::onnx_error::OnnxError;
use crate::nn::graph::onnx_ops;
use crate::tensor::Tensor;
use onnx_rs::ast::{DataType, OpType};

use super::const_table::{build_const_table, build_consumed_meta_names};
use super::fold_reshape::assemble_reshape_with_const_fold;
use super::fold_resize::assemble_resize_with_const_fold;
use super::split_narrow::assemble_split_to_narrows;
use super::util::{
    extract_shape_from_value_info, infer_output_shape_placeholder, resolve_parents, SymbolTable,
};
use super::{ImportReport, RewriteRecord};

/// ONNX 导入结果
#[derive(Debug)]
pub struct OnnxImportResult {
    /// 图描述符(可直接用于 Graph::from_descriptor)
    pub descriptor: GraphDescriptor,
    /// 权重映射：节点 ID → Tensor
    pub weights: HashMap<u64, Tensor>,
    /// 导入过程的可观测报告：模式重写记录 + 非致命警告
    pub import_report: ImportReport,
}

pub(super) fn assemble<'a>(
    graph: &'a onnx_rs::ast::Graph<'a>,
    symbols: &mut SymbolTable,
) -> Result<OnnxImportResult, OnnxError> {
    let mut descriptor = GraphDescriptor::new(graph.name);
    let mut weights: HashMap<u64, Tensor> = HashMap::new();
    let mut import_report = ImportReport::default();

    // initializer 名称集合(这些同时出现在 input 中时不创建 BasicInput)
    let initializer_names: HashSet<&str> =
        graph.initializer.iter().map(|t| t.name()).collect();

    // ── 第 0 步：常量收集 + 元信息消费标记 ──
    let const_table = build_const_table(graph);
    let consumed_meta_names = build_consumed_meta_names(graph);

    // ── 图输入节点(排除 initializer) ──
    for input_info in &graph.input {
        if initializer_names.contains(input_info.name) {
            continue;
        }
        let id = symbols.get_or_assign(input_info.name);
        let shape = extract_shape_from_value_info(input_info);
        descriptor.add_node(NodeDescriptor::new(
            id,
            input_info.name,
            NodeTypeDescriptor::BasicInput,
            shape.clone(),
            Some(shape.iter().map(|&d| if d == 0 { None } else { Some(d) }).collect()),
            vec![],
        ));
    }

    // ── Initializer → Parameter 节点 + 权重 ──
    for init in &graph.initializer {
        // 被算子用作元信息(如 Reshape 的 shape、Resize 的 scales)的 initializer
        // 已经折叠到下游算子属性里,跳过 Parameter 节点创建
        if consumed_meta_names.contains(init.name()) {
            continue;
        }
        if init.data_type() != DataType::Float {
            return Err(OnnxError::UnsupportedDataType {
                data_type: init.data_type() as i32,
                context: format!("initializer \"{}\"", init.name()),
            });
        }
        let id = symbols.get_or_assign(init.name());
        let raw_shape: Vec<usize> = init.dims().iter().map(|&d| d as usize).collect();

        // Parameter 节点要求 ≥ 2 维;1D bias → 升维为 [1, N]
        let shape = if raw_shape.len() == 1 {
            vec![1, raw_shape[0]]
        } else {
            raw_shape.clone()
        };

        descriptor.add_node(NodeDescriptor::new(
            id,
            init.name(),
            NodeTypeDescriptor::Parameter,
            shape.clone(),
            None,
            vec![],
        ));

        let float_data = init
            .as_f32()
            .ok_or_else(|| OnnxError::WeightError {
                tensor_name: init.name().to_string(),
                reason: "无法提取 float32 数据".to_string(),
            })?;
        let tensor = Tensor::new(&float_data, &shape);
        weights.insert(id, tensor);
    }

    // ── 计算节点 ──
    for node in &graph.node {
        // ── 折叠/展开 special path ──

        // Constant 节点处理：
        // - 输出被算子用作元信息(Reshape.shape / Resize.scales / Split.split)→ 已折叠到属性,跳过
        // - 输出被普通数值算子(Mul / Add / Sub 等)当作输入 → 必须保留为 Parameter 节点
        //   持有常量值,否则下游 resolve_parents 找不到父节点 id(典型 YOLOv5 头部 Mul_204 用例)
        if node.op_type == OpType::Constant {
            let output_name = node.output.first().copied().unwrap_or("");
            if consumed_meta_names.contains(output_name) {
                continue;
            }
            let attr = node.attribute.iter().find(|a| a.name == "value");
            let tensor_proto = match attr.and_then(|a| a.t.as_ref()) {
                Some(t) => t,
                None => continue, // 没有 value 属性的 Constant 跳过
            };
            if tensor_proto.data_type() != DataType::Float {
                // 非 float 常量目前不支持作为 Parameter(int 常量罕见)
                continue;
            }
            let id = symbols.get_or_assign(output_name);
            let raw_shape: Vec<usize> = tensor_proto.dims().iter().map(|&d| d as usize).collect();
            // Parameter 节点要求 ≥ 2 维：标量 → [1,1],1D → [1, N]
            let shape = if raw_shape.is_empty() {
                vec![1, 1]
            } else if raw_shape.len() == 1 {
                vec![1, raw_shape[0]]
            } else {
                raw_shape.clone()
            };
            descriptor.add_node(NodeDescriptor::new(
                id,
                output_name,
                NodeTypeDescriptor::Parameter,
                shape.clone(),
                None,
                vec![],
            ));
            if let Some(float_data) = tensor_proto.as_f32() {
                let tensor = Tensor::new(&float_data, &shape);
                weights.insert(id, tensor);
            }
            continue;
        }

        // Reshape：从常量表读 shape input → 填到 Reshape::target_shape
        if node.op_type == OpType::Reshape && node.input.len() >= 2 {
            assemble_reshape_with_const_fold(
                node,
                &const_table,
                symbols,
                &mut descriptor,
                &mut import_report,
            )?;
            continue;
        }

        // Resize / Upsample：从常量表读 scales → 填到 Upsample2d::scale_h/scale_w
        if matches!(node.op_type, OpType::Resize | OpType::Upsample) {
            assemble_resize_with_const_fold(
                node,
                &const_table,
                symbols,
                &mut descriptor,
                &mut import_report,
            )?;
            continue;
        }

        // Split：展开为 N 个 Narrow 节点(split_sizes 来自 attribute 或常量表)
        if node.op_type == OpType::Split {
            assemble_split_to_narrows(
                node,
                &const_table,
                symbols,
                &mut descriptor,
                &mut import_report,
            )?;
            continue;
        }

        // ── 默认路径 ──
        let mapped_descriptors =
            onnx_ops::onnx_op_to_descriptors(&node.op_type, &node.attribute, node.name)?;

        if mapped_descriptors.len() == 1 {
            let parent_ids = resolve_parents(node, symbols)?;

            // Conv/ConvTranspose with bias (3 inputs) → Conv2d + Add
            let is_conv_with_bias = matches!(
                mapped_descriptors[0],
                NodeTypeDescriptor::Conv2d { .. } | NodeTypeDescriptor::ConvTranspose2d { .. }
            ) && parent_ids.len() == 3;

            if is_conv_with_bias {
                emit_conv_with_bias(
                    node,
                    &mapped_descriptors,
                    &parent_ids,
                    symbols,
                    &mut descriptor,
                    &mut weights,
                    &mut import_report,
                );
            } else {
                let output_name = node.output.first().copied().unwrap_or(node.name);
                let out_id = symbols.get_or_assign(output_name);
                let output_shape = infer_output_shape_placeholder(
                    &mapped_descriptors[0],
                    &parent_ids,
                    &descriptor,
                );

                descriptor.add_node(NodeDescriptor::new(
                    out_id,
                    output_name,
                    mapped_descriptors[0].clone(),
                    output_shape,
                    None,
                    parent_ids,
                ));
            }
        } else if mapped_descriptors.len() == 2 {
            // Gemm → MatMul + Add
            emit_gemm(
                node,
                &mapped_descriptors,
                symbols,
                &mut descriptor,
                &mut weights,
                &mut import_report,
            )?;
        }
    }

    Ok(OnnxImportResult {
        descriptor,
        weights,
        import_report,
    })
}

// ==================== Conv+bias 拆分 ====================

fn emit_conv_with_bias(
    node: &onnx_rs::ast::Node,
    mapped_descriptors: &[NodeTypeDescriptor],
    parent_ids: &[u64],
    symbols: &mut SymbolTable,
    descriptor: &mut GraphDescriptor,
    weights: &mut HashMap<u64, Tensor>,
    import_report: &mut ImportReport,
) {
    let input_id = parent_ids[0];
    let weight_id = parent_ids[1];
    let bias_id = parent_ids[2];

    // Reshape bias from [1, C] to [1, C, 1, 1] for 4D broadcasting
    let mut bias_was_reshaped = false;
    if let Some(b_node) = descriptor.nodes.iter_mut().find(|n| n.id == bias_id) {
        if b_node.output_shape.len() == 2 {
            let c = b_node.output_shape[1];
            b_node.output_shape = vec![1, c, 1, 1];
            bias_was_reshaped = true;
        }
    }
    if let Some(b_tensor) = weights.get_mut(&bias_id) {
        if b_tensor.shape().len() == 2 {
            let c = b_tensor.shape()[1];
            *b_tensor = b_tensor.reshape(&[1, c, 1, 1]);
        }
    }
    if bias_was_reshaped {
        import_report.warnings.push(format!(
            "Conv \"{}\" 的 bias 从 [1, C] 自动升维到 [1, C, 1, 1] 以匹配 4D 广播",
            node.name
        ));
    }

    let conv_name = format!("{}/conv", node.name);
    let conv_id = symbols.get_or_assign(&conv_name);
    let conv_shape = infer_output_shape_placeholder(
        &mapped_descriptors[0],
        &[input_id, weight_id],
        descriptor,
    );
    descriptor.add_node(NodeDescriptor::new(
        conv_id,
        &conv_name,
        mapped_descriptors[0].clone(),
        conv_shape,
        None,
        vec![input_id, weight_id],
    ));

    let output_name = node.output.first().copied().unwrap_or(node.name);
    let add_id = symbols.get_or_assign(output_name);
    let add_shape = infer_output_shape_placeholder(
        &NodeTypeDescriptor::Add,
        &[conv_id, bias_id],
        descriptor,
    );
    descriptor.add_node(NodeDescriptor::new(
        add_id,
        output_name,
        NodeTypeDescriptor::Add,
        add_shape,
        None,
        vec![conv_id, bias_id],
    ));

    import_report.rewritten.push(RewriteRecord {
        pattern: "conv_with_bias_to_conv_plus_add",
        consumed_onnx_nodes: vec![node.name.to_string()],
        produced_descriptor_nodes: vec![conv_id, add_id],
    });
}

// ==================== Gemm 拆分 ====================

fn emit_gemm(
    node: &onnx_rs::ast::Node,
    mapped_descriptors: &[NodeTypeDescriptor],
    symbols: &mut SymbolTable,
    descriptor: &mut GraphDescriptor,
    weights: &mut HashMap<u64, Tensor>,
    import_report: &mut ImportReport,
) -> Result<(), OnnxError> {
    if node.input.len() < 3 {
        return Err(OnnxError::InvalidGraph(format!(
            "Gemm 节点 \"{}\" 需要至少 3 个输入,实际 {}",
            node.name,
            node.input.len()
        )));
    }
    let a_id = symbols.get_or_assign(node.input[0]);
    let b_id = symbols.get_or_assign(node.input[1]);
    let c_id = symbols.get_or_assign(node.input[2]);

    // 处理 transB=1：转置权重 B 的 shape 和数据
    let trans_b = onnx_ops::find_attr_int(&node.attribute, "transB").unwrap_or(0);
    if trans_b == 1 {
        if let Some(b_node) = descriptor.nodes.iter_mut().find(|n| n.id == b_id) {
            if b_node.output_shape.len() == 2 {
                let (rows, cols) = (b_node.output_shape[0], b_node.output_shape[1]);
                b_node.output_shape = vec![cols, rows];
            }
        }
        if let Some(b_tensor) = weights.get_mut(&b_id) {
            if b_tensor.shape().len() == 2 {
                *b_tensor = b_tensor.transpose();
            }
        }
        import_report.warnings.push(format!(
            "Gemm \"{}\" 因 transB=1 对权重 B 做了转置(只对 [in, out] 风格 FC 权重生效)",
            node.name
        ));
    }

    // 中间节点：MatMul 的输出
    let matmul_name = format!("{}/matmul", node.name);
    let matmul_id = symbols.get_or_assign(&matmul_name);
    let matmul_shape =
        infer_output_shape_placeholder(&mapped_descriptors[0], &[a_id, b_id], descriptor);
    descriptor.add_node(NodeDescriptor::new(
        matmul_id,
        &matmul_name,
        mapped_descriptors[0].clone(),
        matmul_shape,
        None,
        vec![a_id, b_id],
    ));

    // 最终节点：Add(matmul_out, C)
    let output_name = node.output.first().copied().unwrap_or(node.name);
    let add_id = symbols.get_or_assign(output_name);
    let add_shape =
        infer_output_shape_placeholder(&mapped_descriptors[1], &[matmul_id, c_id], descriptor);
    descriptor.add_node(NodeDescriptor::new(
        add_id,
        output_name,
        mapped_descriptors[1].clone(),
        add_shape,
        None,
        vec![matmul_id, c_id],
    ));

    import_report.rewritten.push(RewriteRecord {
        pattern: "gemm_to_matmul_plus_add",
        consumed_onnx_nodes: vec![node.name.to_string()],
        produced_descriptor_nodes: vec![matmul_id, add_id],
    });
    Ok(())
}
