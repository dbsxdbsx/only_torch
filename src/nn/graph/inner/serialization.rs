/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : GraphInner 底层参数序列化（save_params/load_params）
 *
 * 职责：纯二进制序列化，只处理参数的读写
 *
 * 与 model_io.rs 的区别：
 * - serialization.rs：底层二进制序列化（只处理参数的原始读写）
 * - model_io.rs：高层模型 I/O（生成/解析 GraphDescriptor + 调用底层序列化）
 */

use super::super::error::GraphError;
use super::GraphInner;
use crate::nn::nodes::NodeType;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

impl GraphInner {
    /// 参数文件魔数
    const PARAMS_MAGIC: &'static [u8; 4] = b"OTPR";
    /// 参数文件版本
    const PARAMS_VERSION: u32 = 1;

    /// 保存所有可训练参数到二进制文件
    pub fn save_params<P: AsRef<Path>>(&self, path: P) -> Result<(), GraphError> {
        let file = File::create(path.as_ref())
            .map_err(|e| GraphError::ComputationError(format!("无法创建参数文件: {e}")))?;
        let mut writer = BufWriter::new(file);

        let param_nodes: Vec<_> = self
            .nodes
            .iter()
            .filter_map(|(&id, node)| match node.node_type() {
                NodeType::Parameter(_) => Some((id, node)),
                _ => None,
            })
            .collect();

        writer
            .write_all(Self::PARAMS_MAGIC)
            .map_err(|e| GraphError::ComputationError(format!("写入魔数失败: {e}")))?;
        writer
            .write_all(&Self::PARAMS_VERSION.to_le_bytes())
            .map_err(|e| GraphError::ComputationError(format!("写入版本失败: {e}")))?;
        writer
            .write_all(&(param_nodes.len() as u32).to_le_bytes())
            .map_err(|e| GraphError::ComputationError(format!("写入参数数量失败: {e}")))?;

        for (_id, node) in &param_nodes {
            let name = node.name();
            let value = node
                .value()
                .ok_or_else(|| GraphError::ComputationError(format!("参数 {name} 没有值")))?;
            let shape = value.shape();
            let data = value.data_as_slice();

            let name_bytes = name.as_bytes();
            writer
                .write_all(&(name_bytes.len() as u32).to_le_bytes())
                .map_err(|e| GraphError::ComputationError(format!("写入名称长度失败: {e}")))?;
            writer
                .write_all(name_bytes)
                .map_err(|e| GraphError::ComputationError(format!("写入名称失败: {e}")))?;

            writer
                .write_all(&(shape.len() as u32).to_le_bytes())
                .map_err(|e| GraphError::ComputationError(format!("写入形状维度失败: {e}")))?;
            for &dim in shape {
                writer
                    .write_all(&(dim as u32).to_le_bytes())
                    .map_err(|e| GraphError::ComputationError(format!("写入形状失败: {e}")))?;
            }

            for &val in data {
                writer
                    .write_all(&val.to_le_bytes())
                    .map_err(|e| GraphError::ComputationError(format!("写入数据失败: {e}")))?;
            }
        }

        writer
            .flush()
            .map_err(|e| GraphError::ComputationError(format!("刷新缓冲区失败: {e}")))?;

        Ok(())
    }

    /// 从二进制文件加载参数
    pub fn load_params<P: AsRef<Path>>(&mut self, path: P) -> Result<(), GraphError> {
        let file = File::open(path.as_ref())
            .map_err(|e| GraphError::ComputationError(format!("无法打开参数文件: {e}")))?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|e| GraphError::ComputationError(format!("读取魔数失败: {e}")))?;
        if &magic != Self::PARAMS_MAGIC {
            return Err(GraphError::ComputationError(
                "无效的参数文件：这不是 only_torch 格式的参数文件。请确保使用 save_params() 保存的文件。".to_string(),
            ));
        }

        let mut version_bytes = [0u8; 4];
        reader
            .read_exact(&mut version_bytes)
            .map_err(|e| GraphError::ComputationError(format!("读取版本失败: {e}")))?;
        let version = u32::from_le_bytes(version_bytes);
        if version != Self::PARAMS_VERSION {
            return Err(GraphError::ComputationError(format!(
                "不支持的参数文件版本: {}",
                version
            )));
        }

        let mut count_bytes = [0u8; 4];
        reader
            .read_exact(&mut count_bytes)
            .map_err(|e| GraphError::ComputationError(format!("读取参数数量失败: {e}")))?;
        let param_count = u32::from_le_bytes(count_bytes);

        let name_to_id: HashMap<String, crate::nn::NodeId> = self
            .nodes
            .iter()
            .filter_map(|(&id, node)| match node.node_type() {
                NodeType::Parameter(_) => Some((node.name().to_string(), id)),
                _ => None,
            })
            .collect();

        for _ in 0..param_count {
            let mut name_len_bytes = [0u8; 4];
            reader
                .read_exact(&mut name_len_bytes)
                .map_err(|e| GraphError::ComputationError(format!("读取名称长度失败: {e}")))?;
            let name_len = u32::from_le_bytes(name_len_bytes) as usize;

            let mut name_bytes = vec![0u8; name_len];
            reader
                .read_exact(&mut name_bytes)
                .map_err(|e| GraphError::ComputationError(format!("读取名称失败: {e}")))?;
            let name = String::from_utf8(name_bytes)
                .map_err(|e| GraphError::ComputationError(format!("名称编码无效: {e}")))?;

            let mut shape_dims_bytes = [0u8; 4];
            reader
                .read_exact(&mut shape_dims_bytes)
                .map_err(|e| GraphError::ComputationError(format!("读取形状维度失败: {e}")))?;
            let shape_dims = u32::from_le_bytes(shape_dims_bytes) as usize;

            let mut shape = Vec::with_capacity(shape_dims);
            for _ in 0..shape_dims {
                let mut dim_bytes = [0u8; 4];
                reader
                    .read_exact(&mut dim_bytes)
                    .map_err(|e| GraphError::ComputationError(format!("读取形状失败: {e}")))?;
                shape.push(u32::from_le_bytes(dim_bytes) as usize);
            }

            let data_len: usize = shape.iter().product();
            let mut data = Vec::with_capacity(data_len);
            for _ in 0..data_len {
                let mut val_bytes = [0u8; 4];
                reader
                    .read_exact(&mut val_bytes)
                    .map_err(|e| GraphError::ComputationError(format!("读取数据失败: {e}")))?;
                data.push(f32::from_le_bytes(val_bytes));
            }

            if let Some(&node_id) = name_to_id.get(&name) {
                let tensor = Tensor::new(&data, &shape);
                self.set_node_value(node_id, Some(&tensor))?;
            }
        }

        Ok(())
    }
}
