/*
 * @Author       : 老董
 * @Date         : 2026-03-09
 * @Description  : 演化模型的持久化（.otm 文件格式）
 *
 * .otm (Only Torch Model) 是自包含的模型文件格式：
 * - 拓扑（NetworkGenome 完整序列化）
 * - 权重（二进制 f32 数据）
 * - 元数据（fitness、generations、status 等）
 *
 * 文件结构：
 *   [4 bytes]  Magic: "OTMD"
 *   [4 bytes]  Format version: u32 (1)
 *   [4 bytes]  Metadata JSON length: u32 (N)
 *   [N bytes]  Metadata JSON (UTF-8)
 *   [remaining] Weight snapshots: bincode 序列化的 HashMap<u64, Vec<Tensor>>
 */

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::tensor::Tensor;

use super::error::EvolutionError;
use super::gene::NetworkGenome;
use super::task::FitnessScore;
use super::{EvolutionResult, EvolutionStatus};

// ==================== 文件格式常量 ====================

/// .otm 文件魔数
const OTM_MAGIC: &[u8; 4] = b"OTMD";
/// .otm 文件格式版本
const OTM_VERSION: u32 = 1;

// ==================== Metadata 结构体 ====================

/// .otm 文件的 JSON 元数据部分
#[derive(Serialize, Deserialize)]
struct OtmMetadata {
    /// 文件格式版本
    format_version: u32,
    /// 生产者版本（cargo pkg version）
    producer_version: String,
    /// 完整的网络基因组（含拓扑 + 训练配置，不含权重）
    genome: GenomeSerialized,
    /// 适应度分数
    fitness: FitnessScore,
    /// 演化经历的代数
    generations: usize,
    /// 停止原因
    status: EvolutionStatus,
    /// 人类可读的架构描述
    architecture_summary: String,
}

/// NetworkGenome 的序列化表示（不含 weight_snapshots，权重单独存二进制）
#[derive(Serialize, Deserialize)]
struct GenomeSerialized {
    layers: Vec<super::gene::LayerGene>,
    skip_edges: Vec<super::gene::SkipEdge>,
    input_dim: usize,
    output_dim: usize,
    #[serde(default)]
    seq_len: Option<usize>,
    training_config: super::gene::TrainingConfig,
    generated_by: String,
    next_innovation: u64,
}

impl From<&NetworkGenome> for GenomeSerialized {
    fn from(genome: &NetworkGenome) -> Self {
        Self {
            layers: genome.layers.clone(),
            skip_edges: genome.skip_edges.clone(),
            input_dim: genome.input_dim,
            output_dim: genome.output_dim,
            seq_len: genome.seq_len,
            training_config: genome.training_config.clone(),
            generated_by: genome.generated_by.clone(),
            next_innovation: genome.next_innovation,
        }
    }
}

impl GenomeSerialized {
    /// 从序列化表示 + 权重快照重建完整 NetworkGenome
    fn into_genome(
        self,
        weight_snapshots: std::collections::HashMap<u64, Vec<Tensor>>,
    ) -> NetworkGenome {
        NetworkGenome::from_parts(
            self.layers,
            self.skip_edges,
            self.input_dim,
            self.output_dim,
            self.seq_len,
            self.training_config,
            self.generated_by,
            self.next_innovation,
            weight_snapshots,
        )
    }
}

// ==================== EvolutionResult save/load ====================

impl EvolutionResult {
    /// 保存演化结果到 .otm 文件
    ///
    /// 生成一个自包含的模型文件，包含拓扑、权重和元数据。
    /// `path` 不含文件后缀，自动添加 `.otm`。
    ///
    /// # 示例
    /// ```ignore
    /// let result = Evolution::supervised(train, test, TaskMetric::Accuracy)
    ///     .run()?;
    /// result.save("models/my_evolved")?;
    /// // 生成：models/my_evolved.otm
    /// ```
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), EvolutionError> {
        let path = path.as_ref().with_extension("otm");

        // 确保父目录存在
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    EvolutionError::IoError(format!("无法创建目录 {}: {e}", parent.display()))
                })?;
            }
        }

        let file = File::create(&path).map_err(|e| {
            EvolutionError::IoError(format!("无法创建文件 {}: {e}", path.display()))
        })?;
        let mut writer = BufWriter::new(file);

        // 1. 写 magic + version
        writer.write_all(OTM_MAGIC).map_err(io_write_err)?;
        writer
            .write_all(&OTM_VERSION.to_le_bytes())
            .map_err(io_write_err)?;

        // 2. 序列化 metadata JSON
        let metadata = OtmMetadata {
            format_version: OTM_VERSION,
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            genome: GenomeSerialized::from(&self.genome),
            fitness: self.fitness.clone(),
            generations: self.generations,
            status: self.status.clone(),
            architecture_summary: self.architecture_summary.clone(),
        };
        let json_bytes = serde_json::to_vec_pretty(&metadata).map_err(|e| {
            EvolutionError::IoError(format!("序列化 metadata 失败: {e}"))
        })?;

        // 3. 写 JSON 长度 + JSON 数据
        writer
            .write_all(&(json_bytes.len() as u32).to_le_bytes())
            .map_err(io_write_err)?;
        writer.write_all(&json_bytes).map_err(io_write_err)?;

        // 4. 写权重快照（bincode 序列化 HashMap<u64, Vec<Tensor>>）
        let weights = self.genome.weight_snapshots();
        let weight_bytes = bincode::serialize(weights).map_err(|e| {
            EvolutionError::IoError(format!("序列化权重失败: {e}"))
        })?;
        writer
            .write_all(&(weight_bytes.len() as u64).to_le_bytes())
            .map_err(io_write_err)?;
        writer.write_all(&weight_bytes).map_err(io_write_err)?;

        writer.flush().map_err(io_write_err)?;
        Ok(())
    }

    /// 从 .otm 文件加载演化结果
    ///
    /// 返回一个完整的 `EvolutionResult`，可直接用于推理和可视化。
    /// `path` 不含文件后缀，自动添加 `.otm`。
    ///
    /// # 示例
    /// ```ignore
    /// let model = EvolutionResult::load("models/my_evolved")?;
    /// let pred = model.predict(&input)?;
    /// let vis = model.visualize("output/loaded")?;
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, EvolutionError> {
        let path = path.as_ref().with_extension("otm");

        let file = File::open(&path).map_err(|e| {
            EvolutionError::IoError(format!("无法打开文件 {}: {e}", path.display()))
        })?;
        let mut reader = BufReader::new(file);

        // 1. 读取并验证 magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(io_read_err)?;
        if &magic != OTM_MAGIC {
            return Err(EvolutionError::IoError(
                "无效的 .otm 文件：魔数不匹配。请确保使用 EvolutionResult::save() 保存的文件。"
                    .to_string(),
            ));
        }

        // 2. 读取并验证版本
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes).map_err(io_read_err)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != OTM_VERSION {
            return Err(EvolutionError::IoError(format!(
                "不支持的 .otm 文件版本: {version}（当前支持版本 {OTM_VERSION}）"
            )));
        }

        // 3. 读取 metadata JSON
        let mut json_len_bytes = [0u8; 4];
        reader.read_exact(&mut json_len_bytes).map_err(io_read_err)?;
        let json_len = u32::from_le_bytes(json_len_bytes) as usize;

        let mut json_bytes = vec![0u8; json_len];
        reader.read_exact(&mut json_bytes).map_err(io_read_err)?;

        let metadata: OtmMetadata = serde_json::from_slice(&json_bytes).map_err(|e| {
            EvolutionError::IoError(format!("解析 metadata JSON 失败: {e}"))
        })?;

        // 4. 读取权重快照
        let mut weight_len_bytes = [0u8; 8];
        reader
            .read_exact(&mut weight_len_bytes)
            .map_err(io_read_err)?;
        let weight_len = u64::from_le_bytes(weight_len_bytes) as usize;

        let mut weight_bytes = vec![0u8; weight_len];
        reader.read_exact(&mut weight_bytes).map_err(io_read_err)?;

        let weight_snapshots: std::collections::HashMap<u64, Vec<Tensor>> =
            bincode::deserialize(&weight_bytes).map_err(|e| {
                EvolutionError::IoError(format!("反序列化权重失败: {e}"))
            })?;

        // 5. 重建 genome → build graph → restore weights
        let genome = metadata.genome.into_genome(weight_snapshots);

        let mut rng = StdRng::seed_from_u64(0); // 确定性 seed 用于 build
        let build = genome.build(&mut rng)?;
        genome.restore_weights(&build)?;

        // 6. 自动快照（供 visualize 使用）
        build.graph.snapshot_once_from(&[&build.output]);

        Ok(EvolutionResult {
            build,
            fitness: metadata.fitness,
            generations: metadata.generations,
            architecture_summary: metadata.architecture_summary,
            status: metadata.status,
            genome,
        })
    }
}

// ==================== 辅助函数 ====================

fn io_write_err(e: std::io::Error) -> EvolutionError {
    EvolutionError::IoError(format!("写入文件失败: {e}"))
}

fn io_read_err(e: std::io::Error) -> EvolutionError {
    EvolutionError::IoError(format!("读取文件失败: {e}"))
}
