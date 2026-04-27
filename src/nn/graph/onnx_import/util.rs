//! ONNX 导入流水线共享工具：opset 校验、SymbolTable、形状辅助、常量提取

use std::collections::HashMap;

use crate::nn::descriptor::{GraphDescriptor, NodeTypeDescriptor};
use crate::nn::graph::onnx_error::OnnxError;
use onnx_rs::ast::{DataType, Dimension, TensorProto, TypeValue};

/// 支持的 ONNX opset 版本范围
///
/// 下限放宽到 12 以兼容 VinXiangQi 等 YOLOv5 老版本导出
/// （opset 12 引入了 Constant/Split/Pow 的稳定形式，本 import 已覆盖）
pub(super) const MIN_OPSET_VERSION: i64 = 12;
pub(super) const MAX_OPSET_VERSION: i64 = 21;

// ==================== opset 验证 ====================

pub(super) fn validate_opset(model: &onnx_rs::ast::Model) -> Result<(), OnnxError> {
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

// ==================== 符号表 ====================

/// 管理 ONNX tensor name → 内部 u64 ID 的映射
pub struct SymbolTable {
    name_to_id: HashMap<String, u64>,
    next_id: u64,
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            name_to_id: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn get_or_assign(&mut self, name: &str) -> u64 {
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
    pub fn register_graph(&mut self, graph: &onnx_rs::ast::Graph) {
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

// ==================== 形状辅助 ====================

/// 将 ONNX 节点的输入名称解析为内部 ID
pub(super) fn resolve_parents(
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
pub(super) fn extract_shape_from_value_info(vi: &onnx_rs::ast::ValueInfo) -> Vec<usize> {
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

/// 占位形状推导（最终精确推导由 `Graph::from_descriptor` 负责）
///
/// 对大多数算子继承第一个父节点的形状（最简启发式）。但对 **Concat / Permute**
/// 这两个高频形状变换算子做精确推导，因为下游 Reshape 的 -1 推导依赖父形状的
/// 元素总数，placeholder 不准会让 -1 算成错值（如 YOLOv5 head 的
/// `Concat(axis=4) → Reshape([1,-1,20])` 链路）。
///
/// **不在此处处理**的算子（Conv/MaxPool 等几何变换）：
/// - 它们的输出 shape 跟 stride/padding/kernel 强相关，复杂度高
/// - rebuild 时会精确算
/// - 通常不直接喂给 Reshape，影响面有限
pub(super) fn infer_output_shape_placeholder(
    node_type: &NodeTypeDescriptor,
    parent_ids: &[u64],
    descriptor: &GraphDescriptor,
) -> Vec<usize> {
    let lookup = |id: u64| -> Vec<usize> {
        descriptor
            .nodes
            .iter()
            .find(|n| n.id == id)
            .map(|n| n.output_shape.clone())
            .unwrap_or_default()
    };

    // 高频形状变换算子：精确推导
    match node_type {
        NodeTypeDescriptor::Concat { axis } => {
            // 沿 axis 累加所有父节点的对应维度
            let parent_shapes: Vec<Vec<usize>> = parent_ids.iter().map(|&id| lookup(id)).collect();
            if parent_shapes.is_empty() || parent_shapes[0].is_empty() {
                return vec![];
            }
            let mut out = parent_shapes[0].clone();
            if *axis >= out.len() {
                // axis 越界，退化为第一个父
                return out;
            }
            let mut sum = 0usize;
            for ps in &parent_shapes {
                if ps.len() != out.len() {
                    // shape rank 不一致，退化
                    return out;
                }
                sum += ps.get(*axis).copied().unwrap_or(0);
            }
            out[*axis] = sum;
            return out;
        }
        NodeTypeDescriptor::Permute { dims } => {
            let parent_shape = parent_ids.first().map(|&id| lookup(id)).unwrap_or_default();
            if parent_shape.is_empty() || dims.len() != parent_shape.len() {
                return parent_shape;
            }
            return dims
                .iter()
                .map(|&d| parent_shape.get(d).copied().unwrap_or(0))
                .collect();
        }
        _ => {}
    }

    // 默认：继承第一个父节点的形状
    if let Some(&first_parent) = parent_ids.first() {
        return lookup(first_parent);
    }
    vec![]
}

// ==================== 常量提取 ====================

/// 从常量表中提取 i64 向量（用于 shape / split_sizes 等）
pub(super) fn extract_const_i64<'a>(
    const_table: &HashMap<&'a str, &'a TensorProto<'a>>,
    name: &str,
    op_context: &str,
) -> Result<Vec<i64>, OnnxError> {
    let tensor = const_table.get(name).ok_or_else(|| {
        OnnxError::InvalidGraph(format!(
            "{op_context}: 输入 \"{name}\" 既不是 Constant 节点输出也不是 initializer，\
            无法折叠为静态属性。建议用 onnxsim 预处理把动态形状固化"
        ))
    })?;
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
pub(super) fn extract_const_f32<'a>(
    const_table: &HashMap<&'a str, &'a TensorProto<'a>>,
    name: &str,
    op_context: &str,
) -> Result<Vec<f32>, OnnxError> {
    let tensor = const_table.get(name).ok_or_else(|| {
        OnnxError::InvalidGraph(format!(
            "{op_context}: 输入 \"{name}\" 既不是 Constant 节点输出也不是 initializer，\
            无法折叠为静态属性。建议用 onnxsim 预处理"
        ))
    })?;
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
