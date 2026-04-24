/*
 * @Author       : 老董
 * @Date         : 2026-03-25
 * @Description  : 旧层级基因组 → 节点级基因组迁移器
 *
 * 核心功能：
 * - InnovationCounter : 统一创新号分配入口（单调递增）
 * - expand_*         : 各 LayerConfig 变体的节点展开函数
 * - migrate_network_genome : 将完整 NetworkGenome 展开为 Vec<NodeGene>
 *
 * 设计原则：
 * - 只读取现有 NetworkGenome（LayerGene + SkipEdge），不修改它
 * - 展开结果可直接传入 GenomeAnalysis::compute() 验证合法性
 * - Rnn/Lstm/Gru 不在此版本展开（标记为 deferred，原因见注释）
 * - SkipEdge 转换为显式的 Add/Concat/Maximum 聚合节点
 */

use std::collections::HashMap;

use crate::nn::descriptor::NodeTypeDescriptor;

use super::gene::{
    ActivationType, AggregateStrategy, INPUT_INNOVATION, LayerConfig, NetworkGenome, PoolType,
    ResolvedDim,
};
use super::node_gene::NodeGene;

// ==================== InnovationCounter ====================

/// 统一创新号分配器
///
/// `NodeGene` 携带 `innovation_number` 字段；本结构提供统一的分配入口，避免各处手工维护计数器。
/// mutation、builder 等模块共用同一计数器语义，保证全局单调递增。
#[derive(Debug, Clone)]
pub struct InnovationCounter(u64);

impl InnovationCounter {
    /// 从指定起始值创建计数器
    pub fn new(start: u64) -> Self {
        Self(start)
    }

    /// 分配下一个创新号（单调递增）
    pub fn next(&mut self) -> u64 {
        let id = self.0;
        self.0 += 1;
        id
    }

    /// 当前下一个将分配的值（不消耗）
    pub fn peek(&self) -> u64 {
        self.0
    }
}

// ==================== 错误类型 ====================

/// 迁移错误
#[derive(Debug)]
pub enum MigrationError {
    /// 维度推导失败（旧 genome 本身不合法）
    DimensionError(String),
    /// 基因组包含无效的节点配置（如 Conv2d 输出尺寸为零）
    InvalidGenome(String),
}

impl std::fmt::Display for MigrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionError(msg) => write!(f, "维度推导失败：{msg}"),
            Self::InvalidGenome(msg) => write!(f, "无效基因组：{msg}"),
        }
    }
}

impl std::error::Error for MigrationError {}

// ==================== 迁移输出 ====================

/// 迁移结果
pub struct MigrationOutput {
    /// 展开后的节点列表（拓扑序，叶节点在前）
    pub nodes: Vec<NodeGene>,
    /// 最终输出节点的创新号（用于构图时标识输出）
    pub output_innovation: u64,
    /// 未展开的层描述（Rnn/Lstm/Gru 等暂不支持的变体）
    pub deferred: Vec<String>,
    /// 下一个可用创新号（用于继续分配）
    pub next_innovation: u64,
}

// ==================== ActivationType → NodeTypeDescriptor 映射 ====================

/// 将演化系统的 ActivationType 映射到 NodeTypeDescriptor
pub fn activation_to_node_type(act: &ActivationType) -> NodeTypeDescriptor {
    match act {
        ActivationType::ReLU => NodeTypeDescriptor::ReLU,
        ActivationType::LeakyReLU { alpha } => NodeTypeDescriptor::LeakyReLU { alpha: *alpha },
        ActivationType::Tanh => NodeTypeDescriptor::Tanh,
        ActivationType::Sigmoid => NodeTypeDescriptor::Sigmoid,
        ActivationType::GELU => NodeTypeDescriptor::Gelu,
        ActivationType::SiLU => NodeTypeDescriptor::Swish,
        ActivationType::Softplus => NodeTypeDescriptor::SoftPlus,
        ActivationType::ReLU6 => NodeTypeDescriptor::ReLU6,
        ActivationType::ELU { alpha } => NodeTypeDescriptor::Elu { alpha: *alpha },
        ActivationType::SELU => NodeTypeDescriptor::Selu,
        ActivationType::Mish => NodeTypeDescriptor::Mish,
        ActivationType::HardSwish => NodeTypeDescriptor::HardSwish,
        ActivationType::HardSigmoid => NodeTypeDescriptor::HardSigmoid,
    }
}

// ==================== 单层展开函数 ====================

/// 展开 Linear 层 → Parameter(W) + MatMul + Parameter(b) + Add
///
/// 形状约定（batch=1）：
/// - W: [in_dim, out_features]
/// - MatMul: [1, out_features]
/// - b:  [1, out_features]
/// - Add: [1, out_features]
///
/// 所有节点共享同一个 block_id。
pub fn expand_linear(
    input_id: u64,
    in_dim: usize,
    out_features: usize,
    block_id: u64,
    counter: &mut InnovationCounter,
) -> Vec<NodeGene> {
    let w_id = counter.next();
    let mm_id = counter.next();
    let b_id = counter.next();
    let add_id = counter.next();

    let bid = Some(block_id);
    vec![
        // W: [in_dim, out_features]
        NodeGene::new(
            w_id,
            NodeTypeDescriptor::Parameter,
            vec![in_dim, out_features],
            vec![],
            bid,
        ),
        // MatMul: input[1,in_dim] × W[in_dim,out_features] → [1,out_features]
        NodeGene::new(
            mm_id,
            NodeTypeDescriptor::MatMul,
            vec![1, out_features],
            vec![input_id, w_id],
            bid,
        ),
        // b: [1, out_features]
        NodeGene::new(
            b_id,
            NodeTypeDescriptor::Parameter,
            vec![1, out_features],
            vec![],
            bid,
        ),
        // Add: [1, out_features]
        NodeGene::new(
            add_id,
            NodeTypeDescriptor::Add,
            vec![1, out_features],
            vec![mm_id, b_id],
            bid,
        ),
    ]
}

/// 展开 Activation 层 → 单个激活节点
///
/// 激活节点没有参数，`block_id = None`（细粒度独立节点语义）。
pub fn expand_activation(
    input_id: u64,
    input_shape: Vec<usize>,
    activation_type: &ActivationType,
    counter: &mut InnovationCounter,
) -> Vec<NodeGene> {
    let id = counter.next();
    let nt = activation_to_node_type(activation_type);
    vec![NodeGene::new(id, nt, input_shape, vec![input_id], None)]
}

/// 展开 Conv2d 层 → Parameter(kernel) + Conv2d + Parameter(bias) + Add
/// 形状约定：
/// - kernel: [out_channels, in_channels, k, k]
/// - conv_out: [1, out_channels, H_out, W_out]（same padding，stride=1）
/// - bias: [1, out_channels, 1, 1]
/// - add_out: [1, out_channels, H_out, W_out]
///
/// 所有节点共享同一个 block_id。
pub fn expand_conv2d(
    input_id: u64,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    input_spatial: (usize, usize),
    block_id: u64,
    counter: &mut InnovationCounter,
) -> Vec<NodeGene> {
    let k = kernel_size;
    let padding = k / 2; // same padding
    let (h, w) = input_spatial;
    let h_out = (h + 2 * padding - k) / 1 + 1;
    let w_out = (w + 2 * padding - k) / 1 + 1;

    let kernel_id = counter.next();
    let conv_id = counter.next();
    let bias_id = counter.next();
    let add_id = counter.next();

    let bid = Some(block_id);
    vec![
        // kernel: [out_channels, in_channels, k, k]
        NodeGene::new(
            kernel_id,
            NodeTypeDescriptor::Parameter,
            vec![out_channels, in_channels, k, k],
            vec![],
            bid,
        ),
        // Conv2d: parents=[input, kernel]
        NodeGene::new(
            conv_id,
            NodeTypeDescriptor::Conv2d {
                stride: (1, 1),
                padding: (padding, padding),
                dilation: (1, 1),
            },
            vec![1, out_channels, h_out, w_out],
            vec![input_id, kernel_id],
            bid,
        ),
        NodeGene::new(
            bias_id,
            NodeTypeDescriptor::Parameter,
            vec![1, out_channels, 1, 1],
            vec![],
            bid,
        ),
        NodeGene::new(
            add_id,
            NodeTypeDescriptor::Add,
            vec![1, out_channels, h_out, w_out],
            vec![conv_id, bias_id],
            bid,
        ),
    ]
}

/// 展开 Pool2d 层 → 单个池化节点（无参数，block_id = None）
pub fn expand_pool2d(
    input_id: u64,
    pool_type: PoolType,
    kernel_size: usize,
    stride: usize,
    input_spatial: (usize, usize),
    in_channels: usize,
    counter: &mut InnovationCounter,
) -> Vec<NodeGene> {
    let (h, w) = input_spatial;
    let h_out = if h >= kernel_size {
        (h - kernel_size) / stride + 1
    } else {
        1
    };
    let w_out = if w >= kernel_size {
        (w - kernel_size) / stride + 1
    } else {
        1
    };
    let output_shape = vec![1, in_channels, h_out, w_out];

    let id = counter.next();
    let nt = match pool_type {
        PoolType::Max => NodeTypeDescriptor::MaxPool2d {
            kernel_size: (kernel_size, kernel_size),
            stride: (stride, stride),
            padding: (0, 0),
            ceil_mode: false,
        },
        PoolType::Avg => NodeTypeDescriptor::AvgPool2d {
            kernel_size: (kernel_size, kernel_size),
            stride: (stride, stride),
        },
    };
    vec![NodeGene::new(id, nt, output_shape, vec![input_id], None)]
}

/// 展开 Flatten 层 → 单个 Flatten 节点
///
/// 使用 `keep_first_dim = true`（保留 batch 维）。
pub fn expand_flatten(
    input_id: u64,
    in_channels: usize,
    input_spatial: Option<(usize, usize)>,
    counter: &mut InnovationCounter,
) -> Vec<NodeGene> {
    let id = counter.next();
    let output_shape = match input_spatial {
        Some((h, w)) => vec![1, in_channels * h * w],
        None => vec![1, in_channels], // already flat, identity-like
    };
    vec![NodeGene::new(
        id,
        NodeTypeDescriptor::Flatten {
            keep_first_dim: true,
        },
        output_shape,
        vec![input_id],
        None,
    )]
}

/// 展开 Dropout 层 → 单个 Dropout 节点
pub fn expand_dropout(
    input_id: u64,
    input_shape: Vec<usize>,
    p: f32,
    counter: &mut InnovationCounter,
) -> Vec<NodeGene> {
    let id = counter.next();
    vec![NodeGene::new(
        id,
        NodeTypeDescriptor::Dropout { p },
        input_shape,
        vec![input_id],
        None,
    )]
}

/// 展开 BatchNorm → gamma(Parameter) + BatchNormOp + Multiply + beta(Parameter) + Add（5 节点）
///
/// 形状约定：
/// - Flat 域 `[1, features]`：gamma/beta = `[1, features]`
/// - Spatial 域 `[1, C, H, W]`：gamma/beta = `[1, C, 1, 1]`（通道维广播）
///
/// 所有节点共享同一个 block_id。输出形状 = 输入形状（shape passthrough）。
pub fn expand_batch_norm(
    input_id: u64,
    input_shape: Vec<usize>,
    num_features: usize,
    block_id: u64,
    counter: &mut InnovationCounter,
) -> Vec<NodeGene> {
    let gamma_id = counter.next();
    let bn_id = counter.next();
    let mul_id = counter.next();
    let beta_id = counter.next();
    let add_id = counter.next();

    let bid = Some(block_id);

    // gamma/beta 形状：Flat [1, C]，Spatial [1, C, 1, 1]
    let param_shape = if input_shape.len() == 4 {
        vec![1, num_features, 1, 1]
    } else {
        vec![1, num_features]
    };

    vec![
        NodeGene::new(gamma_id, NodeTypeDescriptor::Parameter, param_shape.clone(), vec![], bid),
        NodeGene::new(
            bn_id,
            NodeTypeDescriptor::BatchNormOp {
                eps: 1e-5,
                momentum: 0.1,
                num_features,
            },
            input_shape.clone(),
            vec![input_id],
            bid,
        ),
        NodeGene::new(mul_id, NodeTypeDescriptor::Multiply, input_shape.clone(), vec![bn_id, gamma_id], bid),
        NodeGene::new(beta_id, NodeTypeDescriptor::Parameter, param_shape, vec![], bid),
        NodeGene::new(add_id, NodeTypeDescriptor::Add, input_shape, vec![mul_id, beta_id], bid),
    ]
}

/// 展开 LayerNorm → gamma(Parameter) + LayerNormOp + Multiply + beta(Parameter) + Add（5 节点）
///
/// 对最后 1 个维度归一化（normalized_dims=1）。gamma/beta 形状 = `[1, features]`。
/// 所有节点共享同一个 block_id。输出形状 = 输入形状。
pub fn expand_layer_norm(
    input_id: u64,
    input_shape: Vec<usize>,
    num_features: usize,
    block_id: u64,
    counter: &mut InnovationCounter,
) -> Vec<NodeGene> {
    let gamma_id = counter.next();
    let ln_id = counter.next();
    let mul_id = counter.next();
    let beta_id = counter.next();
    let add_id = counter.next();

    let bid = Some(block_id);
    let param_shape = vec![1, num_features];

    vec![
        NodeGene::new(gamma_id, NodeTypeDescriptor::Parameter, param_shape.clone(), vec![], bid),
        NodeGene::new(
            ln_id,
            NodeTypeDescriptor::LayerNormOp {
                normalized_dims: 1,
                eps: 1e-5,
            },
            input_shape.clone(),
            vec![input_id],
            bid,
        ),
        NodeGene::new(mul_id, NodeTypeDescriptor::Multiply, input_shape.clone(), vec![ln_id, gamma_id], bid),
        NodeGene::new(beta_id, NodeTypeDescriptor::Parameter, param_shape, vec![], bid),
        NodeGene::new(add_id, NodeTypeDescriptor::Add, input_shape, vec![mul_id, beta_id], bid),
    ]
}

/// 展开 RMSNorm → gamma(Parameter) + RMSNormOp + Multiply（3 节点）
///
/// RMSNorm 无 beta 参数（无偏移），比 LayerNorm 更简洁。
/// gamma 形状 = `[1, features]`。所有节点共享同一个 block_id。
pub fn expand_rms_norm(
    input_id: u64,
    input_shape: Vec<usize>,
    num_features: usize,
    block_id: u64,
    counter: &mut InnovationCounter,
) -> Vec<NodeGene> {
    let gamma_id = counter.next();
    let rn_id = counter.next();
    let mul_id = counter.next();

    let bid = Some(block_id);
    let param_shape = vec![1, num_features];

    vec![
        NodeGene::new(gamma_id, NodeTypeDescriptor::Parameter, param_shape, vec![], bid),
        NodeGene::new(
            rn_id,
            NodeTypeDescriptor::RMSNormOp {
                normalized_dims: 1,
                eps: 1e-5,
            },
            input_shape.clone(),
            vec![input_id],
            bid,
        ),
        NodeGene::new(mul_id, NodeTypeDescriptor::Multiply, input_shape, vec![rn_id, gamma_id], bid),
    ]
}

/// 展开 RNN 层 → 3 个权重参数节点 + 1 个 CellRnn 复合节点（共 4 个节点）
///
/// 父节点顺序：`[input_id, w_ih_id, w_hh_id, b_h_id]`
///
/// 形状约定（batch=1）：
/// - w_ih: `[in_dim, hidden_size]`
/// - w_hh: `[hidden_size, hidden_size]`
/// - b_h:  `[1, hidden_size]`
/// - CellRnn 输出: `[1, hidden_size]`（!return_sequences）或 `[1, seq_len, hidden_size]`
pub fn expand_rnn(
    input_id: u64,
    in_dim: usize,
    hidden_size: usize,
    return_sequences: bool,
    seq_len: usize,
    block_id: u64,
    counter: &mut InnovationCounter,
) -> Vec<NodeGene> {
    let bid = Some(block_id);

    let w_ih_id = counter.next();
    let w_hh_id = counter.next();
    let b_h_id = counter.next();
    let cell_id = counter.next();

    let cell_shape = if return_sequences {
        vec![1, seq_len.max(1), hidden_size]
    } else {
        vec![1, hidden_size]
    };

    vec![
        NodeGene::new(
            w_ih_id,
            NodeTypeDescriptor::Parameter,
            vec![in_dim, hidden_size],
            vec![],
            bid,
        ),
        NodeGene::new(
            w_hh_id,
            NodeTypeDescriptor::Parameter,
            vec![hidden_size, hidden_size],
            vec![],
            bid,
        ),
        NodeGene::new(
            b_h_id,
            NodeTypeDescriptor::Parameter,
            vec![1, hidden_size],
            vec![],
            bid,
        ),
        NodeGene::new(
            cell_id,
            NodeTypeDescriptor::CellRnn {
                input_size: in_dim,
                hidden_size,
                return_sequences,
                seq_len,
            },
            cell_shape,
            vec![input_id, w_ih_id, w_hh_id, b_h_id],
            bid,
        ),
    ]
}

/// 展开 LSTM 层 → 12 个权重参数节点 + 1 个 CellLstm 复合节点（共 13 个节点）
///
/// 父节点顺序：`[input_id, w_ii, w_hi, b_i, w_if, w_hf, b_f, w_ig, w_hg, b_g, w_io, w_ho, b_o]`
///
/// 形状约定（4 门：i/f/g/o，各自 w_ih+w_hh+b）：
/// - w_i*: `[in_dim, hidden_size]`, w_h*: `[hidden_size, hidden_size]`, b_*: `[1, hidden_size]`
#[allow(clippy::too_many_arguments)]
pub fn expand_lstm(
    input_id: u64,
    in_dim: usize,
    hidden_size: usize,
    return_sequences: bool,
    seq_len: usize,
    block_id: u64,
    counter: &mut InnovationCounter,
) -> Vec<NodeGene> {
    let bid = Some(block_id);

    // 输入门 (i)
    let w_ii_id = counter.next();
    let w_hi_id = counter.next();
    let b_i_id = counter.next();
    // 遗忘门 (f)
    let w_if_id = counter.next();
    let w_hf_id = counter.next();
    let b_f_id = counter.next();
    // 细胞门 (g)
    let w_ig_id = counter.next();
    let w_hg_id = counter.next();
    let b_g_id = counter.next();
    // 输出门 (o)
    let w_io_id = counter.next();
    let w_ho_id = counter.next();
    let b_o_id = counter.next();
    let cell_id = counter.next();

    let cell_shape = if return_sequences {
        vec![1, seq_len.max(1), hidden_size]
    } else {
        vec![1, hidden_size]
    };

    let wih = vec![in_dim, hidden_size];
    let whh = vec![hidden_size, hidden_size];
    let wb = vec![1, hidden_size];

    vec![
        NodeGene::new(
            w_ii_id,
            NodeTypeDescriptor::Parameter,
            wih.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            w_hi_id,
            NodeTypeDescriptor::Parameter,
            whh.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            b_i_id,
            NodeTypeDescriptor::Parameter,
            wb.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            w_if_id,
            NodeTypeDescriptor::Parameter,
            wih.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            w_hf_id,
            NodeTypeDescriptor::Parameter,
            whh.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            b_f_id,
            NodeTypeDescriptor::Parameter,
            wb.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            w_ig_id,
            NodeTypeDescriptor::Parameter,
            wih.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            w_hg_id,
            NodeTypeDescriptor::Parameter,
            whh.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            b_g_id,
            NodeTypeDescriptor::Parameter,
            wb.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            w_io_id,
            NodeTypeDescriptor::Parameter,
            wih.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            w_ho_id,
            NodeTypeDescriptor::Parameter,
            whh.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            b_o_id,
            NodeTypeDescriptor::Parameter,
            wb.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            cell_id,
            NodeTypeDescriptor::CellLstm {
                input_size: in_dim,
                hidden_size,
                return_sequences,
                seq_len,
            },
            cell_shape,
            vec![
                input_id, w_ii_id, w_hi_id, b_i_id, w_if_id, w_hf_id, b_f_id, w_ig_id, w_hg_id,
                b_g_id, w_io_id, w_ho_id, b_o_id,
            ],
            bid,
        ),
    ]
}

/// 展开 GRU 层 → 9 个权重参数节点 + 1 个 CellGru 复合节点（共 10 个节点）
///
/// 父节点顺序：`[input_id, w_ir, w_hr, b_r, w_iz, w_hz, b_z, w_in, w_hn, b_n]`
///
/// 形状约定（3 门：r/z/n，各自 w_ih+w_hh+b）：
/// - w_i*: `[in_dim, hidden_size]`, w_h*: `[hidden_size, hidden_size]`, b_*: `[1, hidden_size]`
#[allow(clippy::too_many_arguments)]
pub fn expand_gru(
    input_id: u64,
    in_dim: usize,
    hidden_size: usize,
    return_sequences: bool,
    seq_len: usize,
    block_id: u64,
    counter: &mut InnovationCounter,
) -> Vec<NodeGene> {
    let bid = Some(block_id);

    // 重置门 (r)
    let w_ir_id = counter.next();
    let w_hr_id = counter.next();
    let b_r_id = counter.next();
    // 更新门 (z)
    let w_iz_id = counter.next();
    let w_hz_id = counter.next();
    let b_z_id = counter.next();
    // 候选隐状态门 (n)
    let w_in_id = counter.next();
    let w_hn_id = counter.next();
    let b_n_id = counter.next();
    let cell_id = counter.next();

    let cell_shape = if return_sequences {
        vec![1, seq_len.max(1), hidden_size]
    } else {
        vec![1, hidden_size]
    };

    let wih = vec![in_dim, hidden_size];
    let whh = vec![hidden_size, hidden_size];
    let wb = vec![1, hidden_size];

    vec![
        NodeGene::new(
            w_ir_id,
            NodeTypeDescriptor::Parameter,
            wih.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            w_hr_id,
            NodeTypeDescriptor::Parameter,
            whh.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            b_r_id,
            NodeTypeDescriptor::Parameter,
            wb.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            w_iz_id,
            NodeTypeDescriptor::Parameter,
            wih.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            w_hz_id,
            NodeTypeDescriptor::Parameter,
            whh.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            b_z_id,
            NodeTypeDescriptor::Parameter,
            wb.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            w_in_id,
            NodeTypeDescriptor::Parameter,
            wih.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            w_hn_id,
            NodeTypeDescriptor::Parameter,
            whh.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            b_n_id,
            NodeTypeDescriptor::Parameter,
            wb.clone(),
            vec![],
            bid,
        ),
        NodeGene::new(
            cell_id,
            NodeTypeDescriptor::CellGru {
                input_size: in_dim,
                hidden_size,
                return_sequences,
                seq_len,
            },
            cell_shape,
            vec![
                input_id, w_ir_id, w_hr_id, b_r_id, w_iz_id, w_hz_id, b_z_id, w_in_id, w_hn_id,
                b_n_id,
            ],
            bid,
        ),
    ]
}

// ==================== 主迁移函数 ====================

/// 计算指定位置的循环层是否需要 `return_sequences = true`
///
/// 逻辑：向后扫描 `resolved`，跳过 Activation/Dropout，若下一个实质性层也是循环层则返回 `true`。
fn compute_needs_return_seq(idx: usize, resolved: &[ResolvedDim], genome: &NetworkGenome) -> bool {
    for dim in &resolved[idx + 1..] {
        let layer = genome
            .layers()
            .iter()
            .find(|l| l.innovation_number == dim.innovation_number && l.enabled);
        if let Some(layer) = layer {
            match &layer.layer_config {
                LayerConfig::Activation { .. } | LayerConfig::Dropout { .. } => continue,
                cfg => {
                    return matches!(
                        cfg,
                        LayerConfig::Rnn { .. }
                            | LayerConfig::Lstm { .. }
                            | LayerConfig::Gru { .. }
                    );
                }
            }
        }
    }
    false
}

/// 将旧层级 `NetworkGenome` 展开为节点级 `Vec<NodeGene>`
///
/// # 展开规则
/// - Linear → `expand_linear` (4 节点：W/MatMul/b/Add)
/// - Activation → `expand_activation` (1 节点)
/// - Conv2d → `expand_conv2d` (4 节点：kernel/Conv2d/bias/Add)
/// - Pool2d → `expand_pool2d` (1 节点)
/// - Flatten → `expand_flatten` (1 节点)
/// - Dropout → `expand_dropout` (1 节点)
/// - Rnn/Lstm/Gru → **deferred**（不展开，记入 output.deferred）
///
/// # SkipEdge 处理
/// - Add → 插入 `NodeTypeDescriptor::Add` 聚合节点
/// - Concat → 插入 `NodeTypeDescriptor::Concat` 聚合节点
/// - Max → 插入 `NodeTypeDescriptor::Maximum` 聚合节点
/// - Mean → 插入 `Stack(axis=0)` + `Mean(axis=0)` 保留真实平均语义
///
/// # 创新号
/// 新 NodeGene 的创新号从 1 开始（0 保留给虚拟输入 INPUT_INNOVATION）。
pub fn migrate_network_genome(genome: &NetworkGenome) -> Result<MigrationOutput, MigrationError> {
    let resolved = genome
        .resolve_dimensions()
        .map_err(|e| MigrationError::DimensionError(e.to_string()))?;

    let spatial_map = genome.compute_spatial_map();

    let mut counter = InnovationCounter::new(1); // 0 是 INPUT_INNOVATION
    let mut block_counter: u64 = 0;
    let mut nodes: Vec<NodeGene> = Vec::new();
    let mut deferred: Vec<String> = Vec::new();

    // 旧 innovation → 展开后最后一个 NodeGene 的 innovation（用于 skip edge 查找源节点）
    let mut innov_map: HashMap<u64, u64> = HashMap::new();
    innov_map.insert(INPUT_INNOVATION, INPUT_INNOVATION);

    // 当前主路径输出节点的 innovation
    let mut current_output_id: u64 = INPUT_INNOVATION;

    for (i, dim) in resolved.iter().enumerate() {
        let layer = genome
            .layers()
            .iter()
            .find(|l| l.innovation_number == dim.innovation_number && l.enabled)
            .expect("resolve_dimensions 返回的创新号必须对应启用层");

        // 求本层输入的空间尺寸
        let input_spatial: Option<(usize, usize)> = if i == 0 {
            genome.input_spatial
        } else {
            let prev_innov = resolved[i - 1].innovation_number;
            spatial_map.get(&prev_innov).copied().flatten()
        };

        // 检查是否有到本层的 skip edges
        let incoming: Vec<_> = genome
            .skip_edges()
            .iter()
            .filter(|e| e.enabled && e.to_innovation == layer.innovation_number)
            .collect();

        // 若有 skip edges，先插入聚合节点
        let effective_input_id = if incoming.is_empty() {
            current_output_id
        } else {
            let strategy = &incoming[0].strategy;
            let mut agg_parents = vec![current_output_id];

            for skip in &incoming {
                if let Some(&src_id) = innov_map.get(&skip.from_innovation) {
                    if !agg_parents.contains(&src_id) {
                        agg_parents.push(src_id);
                    }
                }
            }

            match strategy {
                AggregateStrategy::Add => {
                    let agg_id = counter.next();
                    nodes.push(NodeGene::new(
                        agg_id,
                        NodeTypeDescriptor::Add,
                        vec![1, dim.in_dim],
                        agg_parents,
                        None,
                    ));
                    agg_id
                }
                AggregateStrategy::Mean => {
                    let stack_id = counter.next();
                    let mean_id = counter.next();
                    let parent_count = agg_parents.len();
                    nodes.push(NodeGene::new(
                        stack_id,
                        NodeTypeDescriptor::Stack { axis: 0 },
                        vec![parent_count, 1, dim.in_dim],
                        agg_parents,
                        None,
                    ));
                    nodes.push(NodeGene::new(
                        mean_id,
                        NodeTypeDescriptor::Mean { axis: Some(0) },
                        vec![1, dim.in_dim],
                        vec![stack_id],
                        None,
                    ));
                    mean_id
                }
                AggregateStrategy::Max => {
                    let agg_id = counter.next();
                    nodes.push(NodeGene::new(
                        agg_id,
                        NodeTypeDescriptor::Maximum,
                        vec![1, dim.in_dim],
                        agg_parents,
                        None,
                    ));
                    agg_id
                }
                AggregateStrategy::Concat { dim: concat_dim } => {
                    let agg_id = counter.next();
                    // 近似：使用 dim.in_dim 作为 concat 输出（GenomeAnalysis 会验证实际维度）
                    let concat_out = dim.in_dim;
                    let axis = if *concat_dim < 0 {
                        1usize
                    } else {
                        *concat_dim as usize
                    };
                    nodes.push(NodeGene::new(
                        agg_id,
                        NodeTypeDescriptor::Concat { axis },
                        vec![1, concat_out],
                        agg_parents,
                        None,
                    ));
                    agg_id
                }
            }
        };

        // 分配 block_id（每个模板展开一个 block）
        let bid = block_counter;
        block_counter += 1;

        // 循环层需要额外计算 return_sequences 和 seq_len
        let (return_sequences, eff_seq_len) = match &layer.layer_config {
            LayerConfig::Rnn { .. } | LayerConfig::Lstm { .. } | LayerConfig::Gru { .. } => {
                let rs = compute_needs_return_seq(i, &resolved, genome);
                (rs, genome.seq_len.unwrap_or(0))
            }
            _ => (false, 0),
        };

        // 展开层配置
        let result = expand_layer_config(
            &layer.layer_config,
            effective_input_id,
            dim.in_dim,
            dim.out_dim,
            input_spatial,
            bid,
            &mut counter,
            return_sequences,
            eff_seq_len,
        );

        match result {
            Ok(new_nodes) => {
                let last_id = new_nodes
                    .last()
                    .map(|n| n.innovation_number)
                    .unwrap_or(effective_input_id);
                innov_map.insert(layer.innovation_number, last_id);
                current_output_id = last_id;
                nodes.extend(new_nodes);
            }
            Err(reason) => {
                // 不可展开的层（Rnn/Lstm/Gru 等）：标记 deferred，透传输出
                deferred.push(format!(
                    "层 {} ({}) 未展开：{}",
                    layer.innovation_number, layer.layer_config, reason
                ));
                innov_map.insert(layer.innovation_number, current_output_id);
            }
        }
    }

    Ok(MigrationOutput {
        output_innovation: current_output_id,
        next_innovation: counter.peek(),
        nodes,
        deferred,
    })
}

/// 将单个 LayerConfig 展开为 NodeGene 列表
///
/// 返回 `Err(reason)` 表示该变体暂不支持展开（标记为 deferred）。
fn expand_layer_config(
    config: &LayerConfig,
    input_id: u64,
    in_dim: usize,
    _out_dim: usize,
    input_spatial: Option<(usize, usize)>,
    block_id: u64,
    counter: &mut InnovationCounter,
    return_sequences: bool,
    seq_len: usize,
) -> Result<Vec<NodeGene>, String> {
    match config {
        LayerConfig::Linear { out_features } => Ok(expand_linear(
            input_id,
            in_dim,
            *out_features,
            block_id,
            counter,
        )),

        LayerConfig::Activation { activation_type } => {
            // 激活函数透传维度，输入形状 = [1, in_dim]
            Ok(expand_activation(
                input_id,
                vec![1, in_dim],
                activation_type,
                counter,
            ))
        }

        LayerConfig::Dropout { p } => Ok(expand_dropout(input_id, vec![1, in_dim], *p, counter)),

        LayerConfig::Flatten => Ok(expand_flatten(input_id, in_dim, input_spatial, counter)),

        LayerConfig::Conv2d {
            out_channels,
            kernel_size,
        } => {
            let spatial = input_spatial
                .ok_or_else(|| "Conv2d 需要空间输入（input_spatial 不能为 None）".to_string())?;
            Ok(expand_conv2d(
                input_id,
                in_dim, // in_dim 在空间模式下是 in_channels
                *out_channels,
                *kernel_size,
                spatial,
                block_id,
                counter,
            ))
        }

        LayerConfig::Pool2d {
            pool_type,
            kernel_size,
            stride,
        } => {
            let spatial = input_spatial
                .ok_or_else(|| "Pool2d 需要空间输入（input_spatial 不能为 None）".to_string())?;
            Ok(expand_pool2d(
                input_id,
                *pool_type,
                *kernel_size,
                *stride,
                spatial,
                in_dim, // channels
                counter,
            ))
        }

        LayerConfig::Rnn { hidden_size } => Ok(expand_rnn(
            input_id,
            in_dim,
            *hidden_size,
            return_sequences,
            seq_len,
            block_id,
            counter,
        )),

        LayerConfig::Lstm { hidden_size } => Ok(expand_lstm(
            input_id,
            in_dim,
            *hidden_size,
            return_sequences,
            seq_len,
            block_id,
            counter,
        )),

        LayerConfig::Gru { hidden_size } => Ok(expand_gru(
            input_id,
            in_dim,
            *hidden_size,
            return_sequences,
            seq_len,
            block_id,
            counter,
        )),
    }
}

// ==================== FM 分解迁移 ====================

use crate::nn::evolution::fm_ops::next_fm_id;

/// 将 NodeLevel 基因组中的 Conv2d 模板块分解为 FM 节点和边
///
/// 每个 Conv2d 模板块 (kernel[out_ch,in_ch,k,k], Conv2d_op, bias, Add) 被替换为：
/// - in_ch 个输入 FM 节点（如果上游不是 FM 则新建）
/// - out_ch 个输出 FM 节点（各含 Add 聚合 + bias）
/// - out_ch × in_ch 条 FM 边（kernel[1,1,k,k] + Conv2d op）
///
/// 不修改权重快照（结构性迁移，权重在训练时重新初始化）。
pub fn migrate_conv2d_to_feature_maps(
    nodes: &mut Vec<NodeGene>,
    counter: &mut InnovationCounter,
) {
    use std::collections::HashMap;

    // 1. 找到所有 Conv2d 模板块
    let conv_blocks = find_conv2d_blocks(nodes);
    if conv_blocks.is_empty() {
        return;
    }

    let mut fm_counter = next_fm_id(nodes);
    let mut node_map: HashMap<u64, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.innovation_number, i))
        .collect();

    for block in &conv_blocks {
        decompose_conv2d_block(
            nodes,
            block,
            counter,
            &mut fm_counter,
            &mut node_map,
        );
    }

    // 清理已禁用的旧模板块节点
    nodes.retain(|n| n.enabled);
}

/// Conv2d 模板块信息
struct Conv2dBlock {
    kernel_id: u64,
    conv_op_id: u64,
    bias_id: Option<u64>,
    add_id: Option<u64>,
    /// 原 block_id
    block_id: u64,
    /// Conv2d 参数
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    /// 输入通道数
    in_channels: usize,
    /// 输出通道数
    out_channels: usize,
    /// 核大小
    kernel_h: usize,
    kernel_w: usize,
    /// 输入节点 innovation
    input_id: u64,
    /// 输出空间尺寸
    output_h: usize,
    output_w: usize,
    /// 输入空间尺寸
    input_h: usize,
    input_w: usize,
}

/// 查找基因组中所有 Conv2d 模板块
fn find_conv2d_blocks(nodes: &[NodeGene]) -> Vec<Conv2dBlock> {
    use std::collections::HashMap;

    let node_map: HashMap<u64, &NodeGene> = nodes
        .iter()
        .filter(|n| n.enabled)
        .map(|n| (n.innovation_number, n))
        .collect();

    let mut blocks = Vec::new();

    // 找所有 Conv2d op 节点
    for n in nodes.iter().filter(|n| n.enabled) {
        let (stride, padding, dilation) = match &n.node_type {
            NodeTypeDescriptor::Conv2d { stride, padding, dilation } => (*stride, *padding, *dilation),
            _ => continue,
        };

        let block_id = match n.block_id {
            Some(bid) => bid,
            None => continue, // 非模板块
        };

        // 已经是 FM 边（fm_id = None 但上游是 FM 节点）的跳过
        if n.parents.iter().any(|&pid| {
            node_map.get(&pid).map_or(false, |p| p.fm_id.is_some())
        }) {
            continue;
        }

        if n.parents.len() < 2 {
            continue;
        }

        let input_id = n.parents[0];
        let kernel_id = n.parents[1];

        // 验证 kernel 是 Parameter
        let kernel_node = match node_map.get(&kernel_id) {
            Some(kn) if kn.is_parameter() => kn,
            _ => continue,
        };

        let kernel_shape = &kernel_node.output_shape;
        if kernel_shape.len() != 4 {
            continue;
        }

        let (out_channels, in_channels, kernel_h, kernel_w) =
            (kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3]);

        let conv_output_shape = &n.output_shape;
        if conv_output_shape.len() < 4 {
            continue;
        }
        let (output_h, output_w) = (conv_output_shape[2], conv_output_shape[3]);

        // 推导输入空间尺寸
        let eff_kh = dilation.0 * (kernel_h - 1) + 1;
        let eff_kw = dilation.1 * (kernel_w - 1) + 1;
        let input_h = (output_h - 1) * stride.0 + eff_kh - 2 * padding.0;
        let input_w = (output_w - 1) * stride.1 + eff_kw - 2 * padding.1;

        // 查找同 block_id 的 bias 和 add 节点
        let mut bias_id = None;
        let mut add_id = None;
        for m in nodes.iter().filter(|m| m.enabled && m.block_id == Some(block_id)) {
            if m.is_parameter() && m.innovation_number != kernel_id {
                bias_id = Some(m.innovation_number);
            }
            if matches!(m.node_type, NodeTypeDescriptor::Add)
                && m.innovation_number != n.innovation_number
            {
                add_id = Some(m.innovation_number);
            }
        }

        blocks.push(Conv2dBlock {
            kernel_id,
            conv_op_id: n.innovation_number,
            bias_id,
            add_id,
            block_id,
            stride,
            padding,
            dilation,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            input_id,
            output_h,
            output_w,
            input_h,
            input_w,
        });
    }

    blocks
}

/// 将单个 Conv2d 模板块分解为 FM 节点和边
fn decompose_conv2d_block(
    nodes: &mut Vec<NodeGene>,
    block: &Conv2dBlock,
    counter: &mut InnovationCounter,
    fm_counter: &mut u64,
    _node_map: &mut std::collections::HashMap<u64, usize>,
) {
    // 1. 创建输入 FM 节点（每个输入通道一个）
    let mut input_fm_ids = Vec::new();
    for _ic in 0..block.in_channels {
        let fm_id = *fm_counter;
        *fm_counter += 1;
        let node_id = counter.next();

        // 输入 FM 仅是一个标识节点（Select 从原始输入中取出单通道）
        // fm_id 标记，shape = [1, 1, H_in, W_in]
        let mut fm_node = NodeGene::new(
            node_id,
            NodeTypeDescriptor::Identity,
            vec![1, 1, block.input_h, block.input_w],
            vec![block.input_id],
            None,
        );
        fm_node.fm_id = Some(fm_id);
        nodes.push(fm_node);
        input_fm_ids.push((fm_id, node_id));
    }

    // 2. 创建输出 FM 节点和 FM 边
    let mut output_fm_output_ids = Vec::new();
    for _oc in 0..block.out_channels {
        let fm_id = *fm_counter;
        *fm_counter += 1;

        // 为每条输入边创建 kernel + Conv2d op
        let mut edge_output_ids = Vec::new();
        for &(_in_fm_id, in_fm_node_id) in &input_fm_ids {
            let edge_block_id = counter.next();

            // kernel [1, 1, kH, kW]
            let kernel_id = counter.next();
            nodes.push(NodeGene::new(
                kernel_id,
                NodeTypeDescriptor::Parameter,
                vec![1, 1, block.kernel_h, block.kernel_w],
                vec![],
                Some(edge_block_id),
            ));

            // Conv2d op
            let conv_id = counter.next();
            nodes.push(NodeGene::new(
                conv_id,
                NodeTypeDescriptor::Conv2d {
                    stride: block.stride,
                    padding: block.padding,
                    dilation: block.dilation,
                },
                vec![1, 1, block.output_h, block.output_w],
                vec![in_fm_node_id, kernel_id],
                Some(edge_block_id),
            ));

            edge_output_ids.push(conv_id);
        }

        // 聚合所有输入边的输出
        let agg_id = if edge_output_ids.len() == 1 {
            edge_output_ids[0]
        } else {
            // 二叉 Add 树聚合
            let mut current_ids = edge_output_ids;
            while current_ids.len() > 1 {
                let mut next_level = Vec::new();
                for pair in current_ids.chunks(2) {
                    if pair.len() == 2 {
                        let add_id = counter.next();
                        let mut add_node = NodeGene::new(
                            add_id,
                            NodeTypeDescriptor::Add,
                            vec![1, 1, block.output_h, block.output_w],
                            vec![pair[0], pair[1]],
                            None,
                        );
                        add_node.fm_id = Some(fm_id);
                        nodes.push(add_node);
                        next_level.push(add_id);
                    } else {
                        next_level.push(pair[0]);
                    }
                }
                current_ids = next_level;
            }
            current_ids[0]
        };

        // FM 输出节点标记
        if let Some(node) = nodes.iter_mut().find(|n| n.innovation_number == agg_id) {
            node.fm_id = Some(fm_id);
        }

        output_fm_output_ids.push(agg_id);
    }

    // 3. 创建最终的 Concat 节点将所有输出 FM 拼接回 [N, out_ch, H', W']
    let concat_id = counter.next();
    nodes.push(NodeGene::new(
        concat_id,
        NodeTypeDescriptor::Concat { axis: 1 },
        vec![1, block.out_channels, block.output_h, block.output_w],
        output_fm_output_ids.clone(),
        None,
    ));

    // 4. 将下游节点的 parents 从原 Conv2d 块输出重定向到 concat 输出
    let original_output_id = block.add_id.unwrap_or(block.conv_op_id);
    for n in nodes.iter_mut() {
        if n.enabled {
            for pid in n.parents.iter_mut() {
                if *pid == original_output_id {
                    *pid = concat_id;
                }
            }
        }
    }

    // 5. 禁用原模板块节点
    for n in nodes.iter_mut() {
        if n.block_id == Some(block.block_id) && n.fm_id.is_none() {
            n.enabled = false;
        }
    }
}
