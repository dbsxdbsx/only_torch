/*
 * @Author       : 老董
 * @Date         : 2026-03-09
 * @Description  : 演化模型的持久化（统一 .otm v2 格式）
 *
 * 使用统一的 .otm 格式保存演化模型：
 * - graph 字段：GraphDescriptor（始终存在）
 * - evolution 字段：genome + fitness + generations + status（仅演化模型）
 * - 权重按参数名索引（与通用 save_model 格式一致）
 *
 * 加载时通过 genome.build() 重建图，再按参数名恢复权重。
 */

use std::path::Path;

use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::nn::graph::model_save::{self, OTM_FORMAT_VERSION, OtmMetadata};
use crate::nn::var::Var;

use super::error::EvolutionError;
use super::gene::NetworkGenome;
use super::task::FitnessScore;
use super::{EvolutionResult, EvolutionStatus};

// ==================== 演化专属序列化结构体 ====================

/// 演化元数据（存入 OtmMetadata.evolution 字段）
#[derive(Serialize, Deserialize)]
struct EvolutionMeta {
    genome: GenomeSerialized,
    fitness: FitnessScore,
    generations: usize,
    status: EvolutionStatus,
    architecture_summary: String,
}

/// NetworkGenome 的序列化表示（不含 weight_snapshots，权重在二进制段按参数名保存）
///
/// 演化保存路径要求：
/// - Flat/Spatial/Sequential 三类 genome 均须为 NodeLevel 格式（`is_node_level=true`）
/// - 加载旧格式（`is_node_level=false`）会返回明确错误
#[derive(Serialize, Deserialize)]
#[cfg_attr(test, derive(Debug))]
pub(crate) struct GenomeSerialized {
    #[serde(default)]
    layers: Vec<super::gene::LayerGene>,
    #[serde(default)]
    skip_edges: Vec<super::gene::SkipEdge>,
    input_dim: usize,
    output_dim: usize,
    #[serde(default)]
    seq_len: Option<usize>,
    #[serde(default)]
    input_spatial: Option<(usize, usize)>,
    training_config: super::gene::TrainingConfig,
    generated_by: String,
    next_innovation: u64,
    /// NodeLevel 节点列表（NodeLevel 专属，履历文件可贪心为空）
    #[serde(default)]
    nodes: Vec<super::node_gene::NodeGene>,
    /// 是否为 NodeLevel 表示
    #[serde(default)]
    is_node_level: bool,
}

impl From<&NetworkGenome> for GenomeSerialized {
    fn from(genome: &NetworkGenome) -> Self {
        // 所有类型的 genome 均由演化主循环在 run() 中迁移到 NodeLevel。
        // 若此处触发 panic，说明调用方绕过了迁移步骤，属于编程错误。
        if genome.is_layer_level() {
            panic!(
                "尝试以 LayerLevel 格式保存 genome（input_dim={}, output_dim={}, seq_len={:?}），\
                 这是编程错误：演化主循环应已在 run() 中将其迁移到 NodeLevel",
                genome.input_dim, genome.output_dim, genome.seq_len
            );
        }

        if genome.is_node_level() {
            Self {
                layers: Vec::new(),
                skip_edges: Vec::new(),
                input_dim: genome.input_dim,
                output_dim: genome.output_dim,
                seq_len: genome.seq_len,
                input_spatial: genome.input_spatial,
                training_config: genome.training_config.clone(),
                generated_by: genome.generated_by.clone(),
                next_innovation: genome.peek_next_innovation(),
                nodes: genome.nodes().to_vec(),
                is_node_level: true,
            }
        } else {
            Self {
                layers: genome.layers().to_vec(),
                skip_edges: genome.skip_edges().to_vec(),
                input_dim: genome.input_dim,
                output_dim: genome.output_dim,
                seq_len: genome.seq_len,
                input_spatial: genome.input_spatial,
                training_config: genome.training_config.clone(),
                generated_by: genome.generated_by.clone(),
                next_innovation: genome.peek_next_innovation(),
                nodes: Vec::new(),
                is_node_level: false,
            }
        }
    }
}

impl GenomeSerialized {
    /// 从序列化表示重建 NetworkGenome（weight_snapshots 为空，权重由参数名加载）
    ///
    /// # 错误
    /// 若文件中包含旧格式（LayerLevel）的任意 genome，返回错误消息。
    pub(crate) fn into_genome(self) -> Result<NetworkGenome, String> {
        use super::gene::GenomeRepr;

        // 所有类型（Flat/Spatial/Sequential）的演化 genome 均须为 NodeLevel 格式
        if !self.is_node_level {
            return Err(format!(
                "该 .otm 文件中包含旧格式（LayerLevel）的演化基因组（seq_len={:?}），\
                 当前版本已停止支持此格式。请使用当前版本重新运行演化以生成新格式文件。",
                self.seq_len
            ));
        }

        // NodeLevel：直接构建 NodeLevel 表示
        let mut genome = NetworkGenome::from_parts(
            Vec::new(),
            Vec::new(),
            self.input_dim,
            self.output_dim,
            self.seq_len,
            self.input_spatial,
            self.training_config,
            self.generated_by,
            self.next_innovation,
            std::collections::HashMap::new(),
        );
        // 覆盖为 NodeLevel repr
        genome.repr = GenomeRepr::NodeLevel {
            nodes: self.nodes,
            next_innovation: self.next_innovation,
            weight_snapshots: std::collections::HashMap::new(),
        };

        Ok(genome)
    }
}

// ==================== EvolutionResult save/load ====================

impl EvolutionResult {
    /// 保存演化结果到 .otm 文件（统一 v2 格式）
    ///
    /// 生成一个自包含的模型文件，包含拓扑（GraphDescriptor）、演化元数据和权重。
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
        // 从构建好的图中提取 GraphDescriptor
        let desc = Var::vars_to_graph_descriptor(&[&self.build.output], "evolution");

        // 序列化演化专属元数据
        let evo_meta = EvolutionMeta {
            genome: GenomeSerialized::from(&self.genome),
            fitness: self.fitness.clone(),
            generations: self.generations,
            status: self.status.clone(),
            architecture_summary: self.architecture_summary.clone(),
        };
        let evolution_json = serde_json::to_value(&evo_meta)
            .map_err(|e| EvolutionError::IoError(format!("序列化演化元数据失败: {e}")))?;

        let metadata = OtmMetadata {
            format_version: OTM_FORMAT_VERSION,
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            model_name: "evolution".to_string(),
            graph: desc,
            evolution: Some(evolution_json),
        };

        // 使用统一的写入逻辑（按参数名保存权重）
        let params = self.build.graph.inner().get_all_parameters();
        model_save::write_otm_file(path, &metadata, &params)
            .map_err(|e| EvolutionError::IoError(e.to_string()))
    }

    /// 导出演化结果为 .onnx 文件（推理用，不含训练节点）
    ///
    /// `path` 需包含 `.onnx` 后缀。
    ///
    /// # 示例
    /// ```ignore
    /// let result = Evolution::supervised(train, test, TaskMetric::Accuracy)
    ///     .run()?;
    /// result.export_onnx("models/evolved_model.onnx")?;
    /// ```
    pub fn export_onnx<P: AsRef<Path>>(&self, path: P) -> Result<(), EvolutionError> {
        self.build
            .graph
            .export_onnx(path, &[&self.build.output])
            .map_err(|e| EvolutionError::IoError(format!("ONNX 导出失败: {e}")))
    }

    /// 从 .otm 文件加载演化结果
    ///
    /// 读取统一 v2 格式，通过 genome.build() 重建图，再按参数名恢复权重。
    /// `path` 不含文件后缀，自动添加 `.otm`。
    ///
    /// # 示例
    /// ```ignore
    /// let model = EvolutionResult::load("models/my_evolved")?;
    /// let pred = model.predict(&input)?;
    /// let vis = model.visualize("output/loaded")?;
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, EvolutionError> {
        let (metadata, params) =
            model_save::read_otm_file(path).map_err(|e| EvolutionError::IoError(e.to_string()))?;

        // 解析演化专属元数据
        let evolution_json = metadata.evolution.ok_or_else(|| {
            EvolutionError::IoError(
                "该 .otm 文件不包含演化元数据（evolution 字段缺失）。\
                如果这是手动模型，请使用 Graph::load_model() 加载。"
                    .to_string(),
            )
        })?;
        let evo_meta: EvolutionMeta = serde_json::from_value(evolution_json)
            .map_err(|e| EvolutionError::IoError(format!("解析演化元数据失败: {e}")))?;

        // 从 genome 重建图（genome 不含 weight_snapshots，权重由参数名加载）
        let genome = evo_meta
            .genome
            .into_genome()
            .map_err(|e| EvolutionError::IoError(e))?;
        let mut rng = StdRng::seed_from_u64(0);
        let build = genome.build(&mut rng)?;

        // 按参数名恢复权重
        model_save::apply_params_to_graph(&build.graph, &params)
            .map_err(|e| EvolutionError::IoError(e.to_string()))?;

        // 自动快照（供 visualize 使用）
        build.graph.snapshot_once_from(&[&build.output]);

        Ok(EvolutionResult {
            build,
            fitness: evo_meta.fitness,
            generations: evo_meta.generations,
            architecture_summary: evo_meta.architecture_summary,
            status: evo_meta.status,
            genome,
            pareto_front: Vec::new(),
            pareto_genomes: Vec::new(),
        })
    }
}
