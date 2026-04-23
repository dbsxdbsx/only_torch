/*
 * @Author       : 老董
 * @Date         : 2026-04-18
 * @Description  : ONNX 导入流水线：.onnx → GraphDescriptor + 权重
 *
 * 四层流水线：
 * 1. 解析层：读取 .onnx 二进制 → onnx_rs::ast::Model
 * 2. 符号表层：为每个 tensor name 分配唯一 u64 ID
 * 3. 算子映射层：ONNX Node → NodeTypeDescriptor
 * 4. 装配层：组装 GraphDescriptor + 权重 HashMap
 */

use std::collections::HashMap;
use std::path::Path;

use crate::nn::descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor};
use crate::nn::graph::onnx_error::OnnxError;
use crate::nn::graph::onnx_ops;
use crate::tensor::Tensor;
use onnx_rs::ast::{DataType, Dimension, TypeValue};

/// 支持的 ONNX opset 版本范围
const MIN_OPSET_VERSION: i64 = 13;
const MAX_OPSET_VERSION: i64 = 21;

/// ONNX 导入结果
#[derive(Debug)]
pub struct OnnxImportResult {
    /// 图描述符（可直接用于 Graph::from_descriptor）
    pub descriptor: GraphDescriptor,
    /// 权重映射：节点 ID → Tensor
    pub weights: HashMap<u64, Tensor>,
}

/// 从 .onnx 文件加载为 GraphDescriptor + 权重
pub fn load_onnx<P: AsRef<Path>>(path: P) -> Result<OnnxImportResult, OnnxError> {
    let bytes = std::fs::read(path)?;
    load_onnx_from_bytes(&bytes)
}

/// 从内存中的 .onnx 字节流加载
pub fn load_onnx_from_bytes(bytes: &[u8]) -> Result<OnnxImportResult, OnnxError> {
    // ── 第 1 层：解析 ──
    let model =
        onnx_rs::parse(bytes).map_err(|e| OnnxError::ParseError(format!("{e}")))?;

    // 验证 opset 版本
    validate_opset(&model)?;

    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| OnnxError::InvalidGraph("模型不含计算图".to_string()))?;

    // ── 第 2 层：符号表 ──
    let mut symbols = SymbolTable::new();
    symbols.register_graph(graph);

    // ── 第 3 & 4 层：算子映射 + 装配 ──
    assemble(graph, &mut symbols)
}

// ==================== 第 1 层：opset 验证 ====================

fn validate_opset(model: &onnx_rs::ast::Model) -> Result<(), OnnxError> {
    let opset_version = model
        .opset_import
        .iter()
        .find(|op| op.domain.is_empty() || op.domain == "ai.onnx")
        .map(|op| op.version)
        .unwrap_or(0);

    if opset_version < MIN_OPSET_VERSION || opset_version > MAX_OPSET_VERSION {
        return Err(OnnxError::UnsupportedOpsetVersion {
            version: opset_version,
            min_supported: MIN_OPSET_VERSION,
            max_supported: MAX_OPSET_VERSION,
        });
    }
    Ok(())
}

// ==================== 第 2 层：符号表 ====================

/// 管理 ONNX tensor name → 内部 u64 ID 的映射
struct SymbolTable {
    name_to_id: HashMap<String, u64>,
    next_id: u64,
}

impl SymbolTable {
    fn new() -> Self {
        Self {
            name_to_id: HashMap::new(),
            next_id: 1,
        }
    }

    fn get_or_assign(&mut self, name: &str) -> u64 {
        if let Some(&id) = self.name_to_id.get(name) {
            id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.name_to_id.insert(name.to_string(), id);
            id
        }
    }

    /// 预注册图中所有 tensor name（输入、输出、initializer、中间值）
    fn register_graph(&mut self, graph: &onnx_rs::ast::Graph) {
        for input in &graph.input {
            self.get_or_assign(input.name);
        }
        for init in &graph.initializer {
            self.get_or_assign(init.name());
        }
        for node in &graph.node {
            for &out in &node.output {
                if !out.is_empty() {
                    self.get_or_assign(out);
                }
            }
        }
        for output in &graph.output {
            self.get_or_assign(output.name);
        }
    }
}

// ==================== 第 3 & 4 层：装配 ====================

fn assemble(
    graph: &onnx_rs::ast::Graph,
    symbols: &mut SymbolTable,
) -> Result<OnnxImportResult, OnnxError> {
    let mut descriptor = GraphDescriptor::new(graph.name);
    let mut weights: HashMap<u64, Tensor> = HashMap::new();

    // initializer 名称集合（这些同时出现在 input 中时不创建 BasicInput）
    let initializer_names: std::collections::HashSet<&str> =
        graph.initializer.iter().map(|t| t.name()).collect();

    // ── 图输入节点（排除 initializer） ──
    for input_info in &graph.input {
        if initializer_names.contains(input_info.name) {
            continue; // initializer 单独处理
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
        if init.data_type() != DataType::Float {
            return Err(OnnxError::UnsupportedDataType {
                data_type: init.data_type() as i32,
                context: format!("initializer \"{}\"", init.name()),
            });
        }
        let id = symbols.get_or_assign(init.name());
        let raw_shape: Vec<usize> = init.dims().iter().map(|&d| d as usize).collect();

        // Parameter 节点要求 2-4 维；1D bias → 升维为 [1, N]
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
                let input_id = parent_ids[0];
                let weight_id = parent_ids[1];
                let bias_id = parent_ids[2];

                // Reshape bias from [1, C] to [1, C, 1, 1] for 4D broadcasting
                if let Some(b_node) = descriptor.nodes.iter_mut().find(|n| n.id == bias_id) {
                    if b_node.output_shape.len() == 2 {
                        let c = b_node.output_shape[1];
                        b_node.output_shape = vec![1, c, 1, 1];
                    }
                }
                if let Some(b_tensor) = weights.get_mut(&bias_id) {
                    if b_tensor.shape().len() == 2 {
                        let c = b_tensor.shape()[1];
                        *b_tensor = b_tensor.reshape(&[1, c, 1, 1]);
                    }
                }

                let conv_name = format!("{}/conv", node.name);
                let conv_id = symbols.get_or_assign(&conv_name);
                let conv_shape = infer_output_shape_placeholder(
                    &mapped_descriptors[0],
                    &[input_id, weight_id],
                    &descriptor,
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
                    &descriptor,
                );
                descriptor.add_node(NodeDescriptor::new(
                    add_id,
                    output_name,
                    NodeTypeDescriptor::Add,
                    add_shape,
                    None,
                    vec![conv_id, bias_id],
                ));
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
            // ONNX Gemm: Y = alpha * A @ B^T + beta * C (when transB=1)
            if node.input.len() < 3 {
                return Err(OnnxError::InvalidGraph(format!(
                    "Gemm 节点 \"{}\" 需要至少 3 个输入，实际 {}",
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
            }

            // 中间节点：MatMul 的输出
            let matmul_name = format!("{}/matmul", node.name);
            let matmul_id = symbols.get_or_assign(&matmul_name);
            let matmul_shape =
                infer_output_shape_placeholder(&mapped_descriptors[0], &[a_id, b_id], &descriptor);
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
                infer_output_shape_placeholder(&mapped_descriptors[1], &[matmul_id, c_id], &descriptor);
            descriptor.add_node(NodeDescriptor::new(
                add_id,
                output_name,
                mapped_descriptors[1].clone(),
                add_shape,
                None,
                vec![matmul_id, c_id],
            ));
        }
    }

    Ok(OnnxImportResult {
        descriptor,
        weights,
    })
}

/// 将 ONNX 节点的输入名称解析为内部 ID
fn resolve_parents(
    node: &onnx_rs::ast::Node,
    symbols: &mut SymbolTable,
) -> Result<Vec<u64>, OnnxError> {
    let mut parent_ids = Vec::new();
    for &input_name in &node.input {
        if input_name.is_empty() {
            continue; // ONNX 允许空字符串表示"可选输入未提供"
        }
        let id = symbols.get_or_assign(input_name);
        parent_ids.push(id);
    }
    Ok(parent_ids)
}

/// 从 ValueInfo 中提取形状信息
fn extract_shape_from_value_info(vi: &onnx_rs::ast::ValueInfo) -> Vec<usize> {
    if let Some(type_proto) = &vi.r#type {
        if let Some(TypeValue::Tensor(tensor_type)) = &type_proto.value {
            if let Some(shape) = &tensor_type.shape {
                return shape
                    .dim
                    .iter()
                    .map(|d| match &d.value {
                        Dimension::Value(v) => {
                            if *v > 0 { *v as usize } else { 0 } // 0 或负值 = 动态维度
                        }
                        Dimension::Param(_) => 0, // 符号维度 = 动态
                    })
                    .collect();
            }
        }
    }
    vec![] // 无类型信息
}

/// 占位形状推导（最终精确推导由 Graph::from_descriptor 负责）
fn infer_output_shape_placeholder(
    _node_type: &NodeTypeDescriptor,
    parent_ids: &[u64],
    descriptor: &GraphDescriptor,
) -> Vec<usize> {
    // 简单策略：继承第一个父节点的形状
    if let Some(&first_parent) = parent_ids.first() {
        if let Some(parent_node) = descriptor.nodes.iter().find(|n| n.id == first_parent) {
            return parent_node.output_shape.clone();
        }
    }
    vec![]
}

