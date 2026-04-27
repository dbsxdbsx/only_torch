/*
 * @Author       : 老董
 * @Date         : 2026-04-18
 * @Description  : ONNX 导出流水线：GraphDescriptor + 权重 → .onnx
 *
 * 三层流水线：
 * 1. 分类层：对 GraphDescriptor 中每个节点进行导出分类
 * 2. 构建层：组装 ONNX Graph（input/output/initializer/node）
 * 3. 编码层：onnx_rs::encode → 字节流
 */

use std::collections::HashMap;
use std::path::Path;

use crate::nn::descriptor::GraphDescriptor;
use crate::nn::graph::onnx_error::OnnxError;
use crate::nn::graph::onnx_ops::{self, ExportCategory};
use crate::tensor::Tensor;

pub(crate) const EXPORT_OPSET_VERSION: i64 = 17;

/// 将 GraphDescriptor + 权重导出为 .onnx 文件
pub fn save_onnx<P: AsRef<Path>>(
    path: P,
    descriptor: &GraphDescriptor,
    weights: &HashMap<String, Tensor>,
) -> Result<(), OnnxError> {
    let bytes = export_to_bytes(descriptor, weights)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

/// 将 GraphDescriptor + 权重导出为 ONNX 字节流
pub fn export_to_bytes(
    descriptor: &GraphDescriptor,
    weights: &HashMap<String, Tensor>,
) -> Result<Vec<u8>, OnnxError> {
    // 第 1 层：分类过滤
    let plan = build_plan(descriptor)?;

    // 第 2 层：预分配所有字符串
    let pool = build_string_pool(descriptor, &plan);

    // 第 3 层：构建 AST + 编码
    let bytes = encode_model(descriptor, &plan, weights, &pool);
    Ok(bytes)
}

// ==================== 第 1 层：分类过滤 ====================

struct NodePlan {
    name: String,
    id: u64,
    parents: Vec<u64>,
    output_shape: Vec<usize>,
    category: ExportCategory,
}

fn build_plan(desc: &GraphDescriptor) -> Result<Vec<NodePlan>, OnnxError> {
    let mut nodes = Vec::new();

    for node in &desc.nodes {
        let category = onnx_ops::descriptor_to_export_category(&node.node_type);

        match &category {
            ExportCategory::TrainingOnly => continue,
            ExportCategory::Unsupported(reason) => {
                return Err(OnnxError::TrainingNodeInExportPath {
                    node_type: reason.clone(),
                    node_name: node.name.clone(),
                });
            }
            _ => {}
        }

        nodes.push(NodePlan {
            name: node.name.clone(),
            id: node.id,
            parents: node.parents.clone(),
            output_shape: node.output_shape.clone(),
            category,
        });
    }

    Ok(nodes)
}

// ==================== 第 2 层：字符串池 ====================

/// 使用 Box<str> 保证地址稳定（Vec realloc 不影响 Box 堆指针）
struct StringPool {
    strings: Vec<Box<str>>,
}

impl StringPool {
    fn new() -> Self {
        Self {
            strings: Vec::new(),
        }
    }

    fn intern(&mut self, s: &str) {
        if !self.strings.iter().any(|existing| &**existing == s) {
            self.strings.push(s.into());
        }
    }

    fn get(&self, s: &str) -> &str {
        self.strings
            .iter()
            .find(|existing| &***existing == s)
            .map(|b| &**b)
            .unwrap_or("")
    }
}

fn build_string_pool(desc: &GraphDescriptor, plan: &[NodePlan]) -> StringPool {
    let mut pool = StringPool::new();

    pool.intern(&desc.name);
    pool.intern("batch");

    for node in plan {
        pool.intern(&node.name);
        pool.intern(&format!("node_{}", node.name));

        if let ExportCategory::Operator(ref op) = node.category {
            pool.intern(op.op_type);
            for &(attr_name, _) in &op.float_attrs {
                pool.intern(attr_name);
            }
            for &(attr_name, _) in &op.int_attrs {
                pool.intern(attr_name);
            }
            for (attr_name, _) in &op.int_list_attrs {
                pool.intern(attr_name);
            }
        }
    }

    pool
}

// ==================== 第 3 层：编码 ====================

fn encode_model(
    desc: &GraphDescriptor,
    plan: &[NodePlan],
    weights: &HashMap<String, Tensor>,
    pool: &StringPool,
) -> Vec<u8> {
    let id_to_name: HashMap<u64, &str> = plan.iter().map(|n| (n.id, pool.get(&n.name))).collect();

    let all_parent_ids: std::collections::HashSet<u64> = plan
        .iter()
        .flat_map(|n| n.parents.iter().copied())
        .collect();

    // 构建各组件（全部只读借用 pool）
    let initializers = build_initializers(plan, weights, pool);
    let graph_inputs = build_graph_inputs(plan, pool);
    let graph_outputs = build_graph_outputs(plan, &all_parent_ids, pool);
    let compute_nodes = build_compute_nodes(plan, &id_to_name, pool);

    let graph = onnx_rs::ast::Graph {
        node: compute_nodes,
        name: pool.get(&desc.name),
        initializer: initializers,
        input: graph_inputs,
        output: graph_outputs,
        ..Default::default()
    };

    let model = onnx_rs::ast::Model {
        ir_version: 8,
        opset_import: vec![onnx_rs::ast::OperatorSetId {
            domain: "",
            version: EXPORT_OPSET_VERSION,
        }],
        producer_name: "only_torch",
        producer_version: env!("CARGO_PKG_VERSION"),
        graph: Some(graph),
        ..Default::default()
    };

    onnx_rs::encode(&model)
}

fn build_initializers<'a>(
    plan: &[NodePlan],
    weights: &HashMap<String, Tensor>,
    pool: &'a StringPool,
) -> Vec<onnx_rs::ast::TensorProto<'a>> {
    plan.iter()
        .filter(|n| {
            matches!(
                n.category,
                ExportCategory::Initializer | ExportCategory::StateInitializer
            )
        })
        .map(|n| {
            let name = pool.get(&n.name);
            let dims: Vec<i64> = n.output_shape.iter().map(|&d| d as i64).collect();
            let data = if let Some(tensor) = weights.get(&n.name) {
                tensor.flatten_view().to_vec()
            } else {
                vec![0.0f32; n.output_shape.iter().product()]
            };
            onnx_rs::ast::TensorProto::from_f32(name, dims, data)
        })
        .collect()
}

fn build_graph_inputs<'a>(
    plan: &[NodePlan],
    pool: &'a StringPool,
) -> Vec<onnx_rs::ast::ValueInfo<'a>> {
    plan.iter()
        .filter(|n| matches!(n.category, ExportCategory::GraphInput))
        .map(|n| make_value_info(pool.get(&n.name), &n.output_shape, pool))
        .collect()
}

fn build_graph_outputs<'a>(
    plan: &[NodePlan],
    all_parent_ids: &std::collections::HashSet<u64>,
    pool: &'a StringPool,
) -> Vec<onnx_rs::ast::ValueInfo<'a>> {
    plan.iter()
        .filter(|n| matches!(n.category, ExportCategory::Operator(_)))
        .filter(|n| !all_parent_ids.contains(&n.id))
        .map(|n| make_value_info(pool.get(&n.name), &n.output_shape, pool))
        .collect()
}

fn build_compute_nodes<'a>(
    plan: &[NodePlan],
    id_to_name: &HashMap<u64, &'a str>,
    pool: &'a StringPool,
) -> Vec<onnx_rs::ast::Node<'a>> {
    let mut result = Vec::new();

    for node in plan {
        if let ExportCategory::Operator(ref export_op) = node.category {
            let inputs: Vec<&'a str> = node
                .parents
                .iter()
                .filter_map(|pid| id_to_name.get(pid).copied())
                .collect();

            let output_name = pool.get(&node.name);
            let node_name = pool.get(&format!("node_{}", node.name));

            let op_type = onnx_rs::ast::OpType::from(pool.get(export_op.op_type));

            let attrs = build_attributes(export_op, pool);

            result.push(onnx_rs::ast::Node {
                input: inputs,
                output: vec![output_name],
                name: node_name,
                op_type,
                domain: "",
                attribute: attrs,
                doc_string: "",
                overload: "",
                metadata_props: vec![],
            });
        }
    }

    result
}

fn build_attributes<'a>(
    export_op: &onnx_ops::OnnxExportOp,
    pool: &'a StringPool,
) -> Vec<onnx_rs::ast::Attribute<'a>> {
    use onnx_rs::ast::{Attribute, AttributeType};

    let mut attrs = Vec::new();

    for &(name, value) in &export_op.float_attrs {
        attrs.push(Attribute {
            name: pool.get(name),
            r#type: AttributeType::Float,
            f: value,
            ..Default::default()
        });
    }

    for &(name, value) in &export_op.int_attrs {
        attrs.push(Attribute {
            name: pool.get(name),
            r#type: AttributeType::Int,
            i: value,
            ..Default::default()
        });
    }

    for (name, values) in &export_op.int_list_attrs {
        attrs.push(Attribute {
            name: pool.get(name),
            r#type: AttributeType::Ints,
            ints: values.clone(),
            ..Default::default()
        });
    }

    attrs
}

fn make_value_info<'a>(
    name: &'a str,
    shape: &[usize],
    pool: &'a StringPool,
) -> onnx_rs::ast::ValueInfo<'a> {
    use onnx_rs::ast::*;

    let dims: Vec<TensorShapeDimension<'a>> = shape
        .iter()
        .enumerate()
        .map(|(i, &d)| TensorShapeDimension {
            value: if d == 0 && i == 0 {
                Dimension::Param(pool.get("batch"))
            } else {
                Dimension::Value(d as i64)
            },
            denotation: "",
        })
        .collect();

    ValueInfo {
        name,
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(TensorTypeProto {
                elem_type: DataType::Float,
                shape: Some(TensorShape { dim: dims }),
            })),
            denotation: "",
        }),
        doc_string: "",
        metadata_props: vec![],
    }
}
