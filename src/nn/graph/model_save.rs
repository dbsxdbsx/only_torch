/*
 * @Author       : 老董
 * @Description  : 统一 .otm 模型格式 — 通用 save/load
 *
 * .otm (Only Torch Model) 文件格式 v2：
 *   [4 bytes]  Magic: "OTMD"
 *   [4 bytes]  Format version: u32 (2)
 *   [4 bytes]  Metadata JSON length: u32 (N)
 *   [N bytes]  Metadata JSON (UTF-8)
 *   [8 bytes]  Weight data length: u64 (M)
 *   [M bytes]  Weight data (参数名索引的二进制格式)
 *
 * Metadata JSON 包含 GraphDescriptor（始终存在）和可选的 evolution 字段。
 * 权重格式按参数名索引，与 save_params 二进制格式一致（无独立 magic/version）。
 */

use super::descriptor_rebuild::RebuildResult;
use super::error::GraphError;
use super::handle::Graph;
use crate::nn::descriptor::GraphDescriptor;
use crate::nn::var::Var;
use crate::tensor::Tensor;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

// ==================== 文件格式常量 ====================

/// .otm 文件魔数
pub(crate) const OTM_MAGIC: &[u8; 4] = b"OTMD";
/// .otm 文件格式版本（v2：统一格式）
pub(crate) const OTM_FORMAT_VERSION: u32 = 2;

// ==================== Metadata 结构体 ====================

/// .otm 文件的 JSON 元数据
#[derive(Serialize, Deserialize)]
pub(crate) struct OtmMetadata {
    /// 文件格式版本
    pub format_version: u32,
    /// 生产者版本（cargo pkg version）
    pub producer_version: String,
    /// 模型名称
    pub model_name: String,
    /// 图描述符（始终存在）
    pub graph: GraphDescriptor,
    /// 演化元数据（仅演化模型存在）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evolution: Option<serde_json::Value>,
}

// ==================== 通用二进制参数读写 ====================

/// 将参数写入 writer（不含 magic/version，纯数据）
///
/// 格式：
///   [4 bytes] param_count: u32
///   for each param:
///     [4 bytes] name_len: u32
///     [name_len bytes] name (UTF-8)
///     [4 bytes] shape_dims: u32
///     [shape_dims * 4 bytes] each dim: u32
///     [product(shape) * 4 bytes] f32 data (little-endian)
pub(crate) fn write_params(
    writer: &mut impl Write,
    params: &[(String, std::rc::Rc<crate::nn::nodes::NodeInner>)],
) -> Result<(), GraphError> {
    writer
        .write_all(&(params.len() as u32).to_le_bytes())
        .map_err(io_err)?;

    for (name, node) in params {
        let value = node
            .value()
            .ok_or_else(|| GraphError::ComputationError(format!("参数 '{name}' 没有值")))?;
        let shape = value.shape();
        let data = value.data_as_slice();

        // 写名称
        let name_bytes = name.as_bytes();
        writer
            .write_all(&(name_bytes.len() as u32).to_le_bytes())
            .map_err(io_err)?;
        writer.write_all(name_bytes).map_err(io_err)?;

        // 写形状
        writer
            .write_all(&(shape.len() as u32).to_le_bytes())
            .map_err(io_err)?;
        for &dim in shape {
            writer
                .write_all(&(dim as u32).to_le_bytes())
                .map_err(io_err)?;
        }

        // 写数据
        for &val in data {
            writer.write_all(&val.to_le_bytes()).map_err(io_err)?;
        }
    }
    Ok(())
}

/// 从 reader 读取参数（不含 magic/version）
pub(crate) fn read_params(reader: &mut impl Read) -> Result<HashMap<String, Tensor>, GraphError> {
    let mut count_bytes = [0u8; 4];
    reader.read_exact(&mut count_bytes).map_err(io_err)?;
    let count = u32::from_le_bytes(count_bytes) as usize;

    let mut params = HashMap::with_capacity(count);
    for _ in 0..count {
        // 读名称
        let mut name_len_bytes = [0u8; 4];
        reader.read_exact(&mut name_len_bytes).map_err(io_err)?;
        let name_len = u32::from_le_bytes(name_len_bytes) as usize;

        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes).map_err(io_err)?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| GraphError::ComputationError(format!("参数名编码无效: {e}")))?;

        // 读形状
        let mut shape_dims_bytes = [0u8; 4];
        reader.read_exact(&mut shape_dims_bytes).map_err(io_err)?;
        let shape_dims = u32::from_le_bytes(shape_dims_bytes) as usize;

        let mut shape = Vec::with_capacity(shape_dims);
        for _ in 0..shape_dims {
            let mut dim_bytes = [0u8; 4];
            reader.read_exact(&mut dim_bytes).map_err(io_err)?;
            shape.push(u32::from_le_bytes(dim_bytes) as usize);
        }

        // 读数据
        let data_len: usize = shape.iter().product();
        let mut data = Vec::with_capacity(data_len);
        for _ in 0..data_len {
            let mut val_bytes = [0u8; 4];
            reader.read_exact(&mut val_bytes).map_err(io_err)?;
            data.push(f32::from_le_bytes(val_bytes));
        }

        params.insert(name, Tensor::new(&data, &shape));
    }
    Ok(params)
}

// ==================== 通用 .otm 文件 I/O ====================

/// 写入 .otm 文件（供 Graph::save_model 和 EvolutionResult::save 共用）
pub(crate) fn write_otm_file<P: AsRef<Path>>(
    path: P,
    metadata: &OtmMetadata,
    params: &[(String, std::rc::Rc<crate::nn::nodes::NodeInner>)],
) -> Result<(), GraphError> {
    let path = path.as_ref().with_extension("otm");

    // 确保父目录存在
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).map_err(|e| {
                GraphError::ComputationError(format!("无法创建目录 {}: {e}", parent.display()))
            })?;
        }
    }

    let file = File::create(&path).map_err(|e| {
        GraphError::ComputationError(format!("无法创建文件 {}: {e}", path.display()))
    })?;
    let mut writer = BufWriter::new(file);

    // 1. Magic + Version
    writer.write_all(OTM_MAGIC).map_err(io_err)?;
    writer
        .write_all(&OTM_FORMAT_VERSION.to_le_bytes())
        .map_err(io_err)?;

    // 2. Metadata JSON
    let json_bytes = serde_json::to_vec_pretty(metadata)
        .map_err(|e| GraphError::ComputationError(format!("序列化 metadata 失败: {e}")))?;
    writer
        .write_all(&(json_bytes.len() as u32).to_le_bytes())
        .map_err(io_err)?;
    writer.write_all(&json_bytes).map_err(io_err)?;

    // 3. Weight data — 先写入临时 buffer 以计算长度
    let mut weight_buf = Vec::new();
    write_params(&mut weight_buf, params)?;
    writer
        .write_all(&(weight_buf.len() as u64).to_le_bytes())
        .map_err(io_err)?;
    writer.write_all(&weight_buf).map_err(io_err)?;

    writer.flush().map_err(io_err)?;
    Ok(())
}

/// 读取 .otm 文件，返回 (metadata, 参数表)
pub(crate) fn read_otm_file<P: AsRef<Path>>(
    path: P,
) -> Result<(OtmMetadata, HashMap<String, Tensor>), GraphError> {
    let path = path.as_ref().with_extension("otm");

    let file = File::open(&path).map_err(|e| {
        GraphError::ComputationError(format!("无法打开文件 {}: {e}", path.display()))
    })?;
    let mut reader = BufReader::new(file);

    // 1. 验证 magic
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(io_err)?;
    if &magic != OTM_MAGIC {
        return Err(GraphError::ComputationError(
            "无效的 .otm 文件：魔数不匹配".to_string(),
        ));
    }

    // 2. 验证版本
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes).map_err(io_err)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != OTM_FORMAT_VERSION {
        return Err(GraphError::ComputationError(format!(
            "不支持的 .otm 版本: {version}（当前支持版本 {OTM_FORMAT_VERSION}）。\
            如果是旧版演化模型文件（v1），请用旧版本加载后重新保存。"
        )));
    }

    // 3. 读 metadata JSON
    let mut json_len_bytes = [0u8; 4];
    reader.read_exact(&mut json_len_bytes).map_err(io_err)?;
    let json_len = u32::from_le_bytes(json_len_bytes) as usize;

    let mut json_bytes = vec![0u8; json_len];
    reader.read_exact(&mut json_bytes).map_err(io_err)?;
    let metadata: OtmMetadata = serde_json::from_slice(&json_bytes)
        .map_err(|e| GraphError::ComputationError(format!("解析 metadata JSON 失败: {e}")))?;

    // 4. 读权重数据
    let mut weight_len_bytes = [0u8; 8];
    reader.read_exact(&mut weight_len_bytes).map_err(io_err)?;
    let weight_len = u64::from_le_bytes(weight_len_bytes) as usize;

    let mut weight_bytes = vec![0u8; weight_len];
    reader.read_exact(&mut weight_bytes).map_err(io_err)?;
    let params = read_params(&mut weight_bytes.as_slice())?;

    Ok((metadata, params))
}

/// 将参数表中的权重加载到图的参数注册表中
pub(crate) fn apply_params_to_graph(
    graph: &Graph,
    params: &HashMap<String, Tensor>,
) -> Result<(), GraphError> {
    let inner = graph.inner();
    for (name, tensor) in params {
        if let Some(node) = inner.get_parameter(name) {
            node.set_value(Some(tensor))?;
        }
        // 跳过图中不存在的参数（容许权重文件有多余参数）
    }
    Ok(())
}

// ==================== Graph 用户 API ====================

impl Graph {
    /// 保存模型到 .otm 文件（拓扑 + 权重）
    ///
    /// `path` 不含文件后缀，自动添加 `.otm`。
    /// `outputs` 是模型的输出 Var，用于提取计算图拓扑。
    ///
    /// # 示例
    /// ```ignore
    /// let output = model.forward(&input)?;
    /// graph.save_model("models/my_model", &[&output])?;
    /// // 生成：models/my_model.otm
    /// ```
    pub fn save_model<P: AsRef<Path>>(&self, path: P, outputs: &[&Var]) -> Result<(), GraphError> {
        let desc = Var::vars_to_graph_descriptor(outputs, "model");
        let metadata = OtmMetadata {
            format_version: OTM_FORMAT_VERSION,
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            model_name: "model".to_string(),
            graph: desc,
            evolution: None,
        };

        let params = self.inner().get_all_parameters();
        write_otm_file(path, &metadata, &params)
    }

    /// 从 .otm 文件加载模型（重建拓扑 + 恢复权重）
    ///
    /// `path` 不含文件后缀，自动添加 `.otm`。
    /// 返回 `RebuildResult`，包含重建后的图、输入/输出 Var。
    /// 加载后自动设为 eval 模式。
    ///
    /// # 示例
    /// ```ignore
    /// let result = Graph::load_model("models/my_model")?;
    /// // 喂入数据
    /// result.inputs[0].1.set_value(&data)?;
    /// result.graph.forward(&result.outputs[0])?;
    /// let prediction = result.outputs[0].value()?;
    /// ```
    pub fn load_model<P: AsRef<Path>>(path: P) -> Result<RebuildResult, GraphError> {
        let (metadata, params) = read_otm_file(path)?;

        // 从 GraphDescriptor 重建计算图
        let result = Graph::from_descriptor(&metadata.graph)?;

        // 加载权重
        apply_params_to_graph(&result.graph, &params)?;

        // 默认 eval 模式（推理场景）
        result.graph.eval();
        Ok(result)
    }

    /// 从 .onnx 文件导入模型（重建拓扑 + 恢复权重）
    ///
    /// 返回 `RebuildResult`，包含重建后的图、输入/输出 Var。
    /// 加载后自动设为 eval 模式。
    ///
    /// # 示例
    /// ```ignore
    /// let result = Graph::from_onnx("models/resnet18.onnx")?;
    /// result.inputs[0].1.set_value(&image_tensor)?;
    /// result.graph.forward(&result.outputs[0])?;
    /// let prediction = result.outputs[0].value()?;
    /// ```
    pub fn from_onnx<P: AsRef<Path>>(path: P) -> Result<RebuildResult, GraphError> {
        let import_result = super::onnx_import::load_onnx(path)
            .map_err(|e| GraphError::ComputationError(format!("ONNX 导入失败: {e}")))?;

        Self::from_onnx_result(import_result)
    }

    /// 从内存中的 .onnx 字节流导入模型
    ///
    /// # 示例
    /// ```ignore
    /// let bytes = std::fs::read("model.onnx")?;
    /// let result = Graph::from_onnx_bytes(&bytes)?;
    /// ```
    pub fn from_onnx_bytes(bytes: &[u8]) -> Result<RebuildResult, GraphError> {
        let import_result = super::onnx_import::load_onnx_from_bytes(bytes)
            .map_err(|e| GraphError::ComputationError(format!("ONNX 导入失败: {e}")))?;

        Self::from_onnx_result(import_result)
    }

    fn from_onnx_result(
        import_result: super::onnx_import::OnnxImportResult,
    ) -> Result<RebuildResult, GraphError> {
        let mut result = Graph::from_descriptor(&import_result.descriptor)?;

        // ONNX 权重按 descriptor node ID 索引 → 转换为按名称索引
        let name_params: HashMap<String, Tensor> = import_result
            .descriptor
            .nodes
            .iter()
            .filter_map(|n| {
                import_result
                    .weights
                    .get(&n.id)
                    .map(|t| (n.name.clone(), t.clone()))
            })
            .collect();
        apply_params_to_graph(&result.graph, &name_params)?;

        result.graph.eval();
        // 透传 ONNX 导入报告供上层观测（rewrite 记录 + 警告）
        result.import_report = Some(import_result.import_report);
        Ok(result)
    }

    /// 导出模型为 .onnx 文件
    ///
    /// `path` 需包含 `.onnx` 后缀。
    /// `outputs` 是模型的输出 Var，用于提取计算图拓扑。
    ///
    /// # 示例
    /// ```ignore
    /// let output = model.forward(&input)?;
    /// graph.export_onnx("models/my_model.onnx", &[&output])?;
    /// ```
    pub fn export_onnx<P: AsRef<Path>>(&self, path: P, outputs: &[&Var]) -> Result<(), GraphError> {
        let desc = Var::vars_to_graph_descriptor(outputs, "model");
        let weights = self.collect_weight_map();
        super::onnx_export::save_onnx(path, &desc, &weights)
            .map_err(|e| GraphError::ComputationError(format!("ONNX 导出失败: {e}")))
    }

    /// 导出模型为 ONNX 字节流（内存中，不写文件）
    pub fn export_onnx_bytes(&self, outputs: &[&Var]) -> Result<Vec<u8>, GraphError> {
        let desc = Var::vars_to_graph_descriptor(outputs, "model");
        let weights = self.collect_weight_map();
        super::onnx_export::export_to_bytes(&desc, &weights)
            .map_err(|e| GraphError::ComputationError(format!("ONNX 导出失败: {e}")))
    }

    /// 从注册参数表中收集权重 → HashMap<String, Tensor>
    fn collect_weight_map(&self) -> HashMap<String, Tensor> {
        let inner = self.inner();
        let params = inner.get_all_parameters();
        let mut weight_map = HashMap::new();
        for (name, node) in params {
            if let Some(tensor) = node.value() {
                weight_map.insert(name, tensor);
            }
        }
        weight_map
    }
}

// ==================== 辅助函数 ====================

fn io_err(e: std::io::Error) -> GraphError {
    GraphError::ComputationError(format!("I/O 错误: {e}"))
}
