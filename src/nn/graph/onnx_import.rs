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

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::nn::descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor};
use crate::nn::graph::onnx_error::OnnxError;
use crate::nn::graph::onnx_ops;
use crate::tensor::Tensor;
use onnx_rs::ast::{DataType, Dimension, OpType, TensorProto, TypeValue};

/// 支持的 ONNX opset 版本范围
///
/// 下限放宽到 12 以兼容 VinXiangQi 等 YOLOv5 老版本导出
/// （opset 12 引入了 Constant/Split/Pow 的稳定形式，本 import 已覆盖）
const MIN_OPSET_VERSION: i64 = 12;
const MAX_OPSET_VERSION: i64 = 21;

/// ONNX 导入结果
#[derive(Debug)]
pub struct OnnxImportResult {
    /// 图描述符（可直接用于 Graph::from_descriptor）
    pub descriptor: GraphDescriptor,
    /// 权重映射：节点 ID → Tensor
    pub weights: HashMap<u64, Tensor>,
    /// 导入过程的可观测报告：模式重写记录 + 非致命警告
    pub import_report: ImportReport,
}

/// ONNX 导入过程的可观测报告（最小骨架版）
///
/// 当前仅含：
/// - `rewritten`：所有命中的模式重写记录（如 Conv+bias 拆分、Split→Narrow 等）
/// - `warnings`：非致命警告（如属性默认值兜底、未使用的 initializer 等）
///
/// **范围控制**（参见 `chinese_chess_yolo_example_b4f3a201.plan.md` §3.4）：
/// 不含 `folded`/`shape_inference`/`provenance`/`origin_onnx_nodes` 等扩展字段，
/// 等真正撞到对应需求时再补，避免范围蔓延。
#[derive(Debug, Default, Clone)]
pub struct ImportReport {
    /// 已应用的模式重写记录（按命中顺序排列）
    pub rewritten: Vec<RewriteRecord>,
    /// 非致命警告
    pub warnings: Vec<String>,
}

/// 单条 pattern rewrite 记录
///
/// 描述一次 ONNX → only_torch 节点重写的输入/输出对应关系，
/// 便于上层调试"为什么 ONNX 节点数和 only_torch 节点数不一致"。
#[derive(Debug, Clone)]
pub struct RewriteRecord {
    /// 模式名（如 `"conv_with_bias_to_conv_plus_add"`、`"split_to_narrows"`、
    /// `"constant_fold_into_reshape"`）
    pub pattern: &'static str,
    /// 该重写"消化"了哪些 ONNX 原始节点（按 ONNX `node.name` 收集）
    pub consumed_onnx_nodes: Vec<String>,
    /// 该重写在 only_torch 内"产出"了哪些 descriptor 节点 ID
    pub produced_descriptor_nodes: Vec<u64>,
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

fn assemble<'a>(
    graph: &'a onnx_rs::ast::Graph<'a>,
    symbols: &mut SymbolTable,
) -> Result<OnnxImportResult, OnnxError> {
    let mut descriptor = GraphDescriptor::new(graph.name);
    let mut weights: HashMap<u64, Tensor> = HashMap::new();
    let mut import_report = ImportReport::default();

    // initializer 名称集合（这些同时出现在 input 中时不创建 BasicInput）
    let initializer_names: HashSet<&str> =
        graph.initializer.iter().map(|t| t.name()).collect();

    // ── 第 0 步：常量折叠预处理 ──
    // 收集所有可作为元信息源的常量：
    //   1. ONNX `Constant` 节点的 value 属性（输出名 → TensorProto）
    //   2. graph.initializer（initializer 既可能是真权重，也可能是 shape/scales 等元信息）
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

    // 标记哪些 const_table 中的常量是被算子用作"元信息"消费的
    // （这些常量来自 initializer 时，不应再被创建为 Parameter 节点暴露给运行时）
    let mut consumed_meta_names: HashSet<&str> = HashSet::new();
    for node in &graph.node {
        match node.op_type {
            OpType::Reshape if node.input.len() >= 2 && !node.input[1].is_empty() => {
                consumed_meta_names.insert(node.input[1]);
            }
            OpType::Resize | OpType::Upsample => {
                // ONNX Resize 输入布局：[X, roi, scales, sizes]
                // YOLOv5 通常 roi 为空、scales 或 sizes 二选一
                if node.input.len() >= 3 && !node.input[2].is_empty() {
                    consumed_meta_names.insert(node.input[2]);
                }
                if node.input.len() >= 4 && !node.input[3].is_empty() {
                    consumed_meta_names.insert(node.input[3]);
                }
            }
            OpType::Split if node.input.len() >= 2 && !node.input[1].is_empty() => {
                consumed_meta_names.insert(node.input[1]);
            }
            _ => {}
        }
    }

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
        // 被算子用作元信息（如 Reshape 的 shape、Resize 的 scales）的 initializer
        // 已经折叠到下游算子属性里，跳过 Parameter 节点创建
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
        // ── 折叠/展开 special path ──

        // Constant 节点本身在 only_torch 内消失（值已折叠到下游算子属性）
        if node.op_type == OpType::Constant {
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

        // Split：展开为 N 个 Narrow 节点（split_sizes 来自 attribute 或常量表）
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

                import_report.rewritten.push(RewriteRecord {
                    pattern: "conv_with_bias_to_conv_plus_add",
                    consumed_onnx_nodes: vec![node.name.to_string()],
                    produced_descriptor_nodes: vec![conv_id, add_id],
                });
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

            import_report.rewritten.push(RewriteRecord {
                pattern: "gemm_to_matmul_plus_add",
                consumed_onnx_nodes: vec![node.name.to_string()],
                produced_descriptor_nodes: vec![matmul_id, add_id],
            });
        }
    }

    Ok(OnnxImportResult {
        descriptor,
        weights,
        import_report,
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

// ==================== 常量折叠 / Split 重写 special path ====================

/// 从常量表中提取 i64 向量（用于 shape / split_sizes 等）
fn extract_const_i64<'a>(
    const_table: &HashMap<&'a str, &'a TensorProto<'a>>,
    name: &str,
    op_context: &str,
) -> Result<Vec<i64>, OnnxError> {
    let tensor = const_table.get(name).ok_or_else(|| OnnxError::InvalidGraph(
        format!(
            "{op_context}: 输入 \"{name}\" 既不是 Constant 节点输出也不是 initializer，\
            无法折叠为静态属性。建议用 onnxsim 预处理把动态形状固化"
        ),
    ))?;
    if tensor.data_type() != DataType::Int64 {
        return Err(OnnxError::UnsupportedDataType {
            data_type: tensor.data_type() as i32,
            context: format!(
                "{op_context}: 期望 int64 元信息常量 \"{name}\"，但得到 {:?}",
                tensor.data_type()
            ),
        });
    }
    Ok(tensor
        .as_i64()
        .ok_or_else(|| OnnxError::WeightError {
            tensor_name: name.to_string(),
            reason: format!("{op_context}: 无法从常量提取 int64 数据"),
        })?
        .into_owned())
}

/// 从常量表中提取 f32 向量（用于 Resize 的 scales）
fn extract_const_f32<'a>(
    const_table: &HashMap<&'a str, &'a TensorProto<'a>>,
    name: &str,
    op_context: &str,
) -> Result<Vec<f32>, OnnxError> {
    let tensor = const_table.get(name).ok_or_else(|| OnnxError::InvalidGraph(
        format!(
            "{op_context}: 输入 \"{name}\" 既不是 Constant 节点输出也不是 initializer，\
            无法折叠为静态属性。建议用 onnxsim 预处理"
        ),
    ))?;
    if tensor.data_type() != DataType::Float {
        return Err(OnnxError::UnsupportedDataType {
            data_type: tensor.data_type() as i32,
            context: format!(
                "{op_context}: 期望 float32 元信息常量 \"{name}\"，但得到 {:?}",
                tensor.data_type()
            ),
        });
    }
    Ok(tensor
        .as_f32()
        .ok_or_else(|| OnnxError::WeightError {
            tensor_name: name.to_string(),
            reason: format!("{op_context}: 无法从常量提取 float32 数据"),
        })?
        .into_owned())
}

/// 把 ONNX 风格的 i64 shape 转为 only_torch 的 `Vec<usize>`
///
/// ONNX shape 允许两种特殊值：
/// - `-1`（推导）：根据 `parent.numel() / 其他维度乘积` 算出，最多一个
/// - `0`（保留）：复用 `parent_shape` 对应位置的维度
///
/// 当 parent 形状中有动态维度（=0 表示未知 batch 等）时，-1/0 的推导可能失败，
/// 此时回退为 1 占位，下游 `Reshape::new` 会按动态 batch 机制处理。
fn convert_onnx_shape_to_usize(
    raw: &[i64],
    parent_shape: &[usize],
    op_context: &str,
) -> Result<Vec<usize>, OnnxError> {
    // 第 1 趟：扫描 + 校验，统计 -1 出现次数 + 收集已知维度乘积
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
                            "ONNX shape {raw:?} 含多个 -1，违反 ONNX 规范"
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
                            "ONNX shape {raw:?} 第 {i} 维为 0（保留），但 parent 仅 {} 维",
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
    //   2. parent_shape 含动态维度 → 占位 1（让 Reshape 动态 batch 接管）
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
            // 无法精确推导 → 占位 1（动态 batch 走运行时调整）
            // has_unknown_factor 用于说明不静态可推导是因为 parent 含 0
            let _ = has_unknown_factor;
            1
        }
        (None, _) => 1, // 没 -1，占位无意义
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

/// 装配 Reshape 节点：从常量表读取 shape 输入，折叠到 `Reshape::target_shape`
fn assemble_reshape_with_const_fold<'a>(
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
    // 查 parent 形状供 -1/0 推导（descriptor 中找不到时给空 slice，意味着无法推导）
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

    descriptor.add_node(NodeDescriptor::new(
        out_id,
        output_name,
        NodeTypeDescriptor::Reshape { target_shape: target_shape.clone() },
        target_shape,
        None,
        vec![parent_id],
    ));

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

/// 装配 Resize/Upsample 节点：从常量表读取 scales/sizes，折叠到 `Upsample2d::scale_h/scale_w`
///
/// 仅支持 NCHW 4 维 + 整数倍 nearest 模式（YOLOv5 默认模式）。
fn assemble_resize_with_const_fold<'a>(
    node: &onnx_rs::ast::Node<'a>,
    const_table: &HashMap<&'a str, &'a TensorProto<'a>>,
    symbols: &mut SymbolTable,
    descriptor: &mut GraphDescriptor,
    import_report: &mut ImportReport,
) -> Result<(), OnnxError> {
    // mode 校验委托给 onnx_op_to_descriptors（已检查 mode="nearest"）
    let _ = onnx_ops::onnx_op_to_descriptors(&node.op_type, &node.attribute, node.name)?;

    let data_name = node.input[0];
    let op_ctx = format!("Resize 节点 \"{}\"", node.name);

    // 优先用 scales（input[2]，f32），不存在则用 sizes（input[3]，i64）反推
    let (scale_h, scale_w, source_const) = if node.input.len() >= 3 && !node.input[2].is_empty() {
        let scales_name = node.input[2];
        let scales = extract_const_f32(const_table, scales_name, &op_ctx)?;
        if scales.len() != 4 {
            return Err(OnnxError::UnsupportedAttribute {
                op_type: "Resize".to_string(),
                attribute: "scales".to_string(),
                reason: format!(
                    "{op_ctx}: 仅支持 4 维 NCHW（scales.len=4），实际 {}",
                    scales.len()
                ),
            });
        }
        // 通常 N、C 维 scale=1.0，H/W 维是上采样倍数
        for (i, &s) in scales.iter().enumerate().take(2) {
            if (s - 1.0).abs() > 1e-6 {
                return Err(OnnxError::UnsupportedAttribute {
                    op_type: "Resize".to_string(),
                    attribute: format!("scales[{i}]"),
                    reason: format!("{op_ctx}: N/C 维 scale 必须=1.0，得到 {s}"),
                });
            }
        }
        let sh = scales[2];
        let sw = scales[3];
        if sh.fract().abs() > 1e-6 || sw.fract().abs() > 1e-6 || sh < 1.0 || sw < 1.0 {
            return Err(OnnxError::UnsupportedAttribute {
                op_type: "Resize".to_string(),
                attribute: "scales".to_string(),
                reason: format!(
                    "{op_ctx}: 仅支持整数倍上采样（scale≥1，nearest），得到 H={sh} W={sw}"
                ),
            });
        }
        (sh as usize, sw as usize, scales_name.to_string())
    } else if node.input.len() >= 4 && !node.input[3].is_empty() {
        // sizes 路径：从输出/输入 shape 反推 scale。本轮最小骨架不支持，
        // 用户可改用 scales 形式或用 onnxsim 转换
        return Err(OnnxError::UnsupportedAttribute {
            op_type: "Resize".to_string(),
            attribute: "sizes".to_string(),
            reason: format!(
                "{op_ctx}: 仅支持 scales 形式，sizes 形式请用 onnxsim 转换"
            ),
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

    descriptor.add_node(NodeDescriptor::new(
        out_id,
        output_name,
        NodeTypeDescriptor::Upsample2d { scale_h, scale_w },
        output_shape,
        None,
        vec![parent_id],
    ));

    import_report.rewritten.push(RewriteRecord {
        pattern: "constant_fold_into_resize",
        consumed_onnx_nodes: vec![
            node.name.to_string(),
            format!("<const:{source_const}>"),
        ],
        produced_descriptor_nodes: vec![out_id],
    });
    Ok(())
}

/// 装配 Split 节点：展开为 N 个 Narrow 节点
///
/// `split_sizes` 来源优先级：
/// 1. opset 12 及以下：attribute "split"（Vec<i64>）
/// 2. opset 13 及以上：input[1] 为常量 i64 张量
/// 3. 都没有：要求 axis 维度均匀 N 等分（暂不支持，需要 input shape 信息，本轮报错）
fn assemble_split_to_narrows<'a>(
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
                "Split 节点 \"{}\": 仅支持非负 axis，得到 {axis}（请用 onnxsim 规范化）",
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
                    "Split 节点 \"{}\": split 既无 input 也无 attribute，等分模式需 input shape 信息，\
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

