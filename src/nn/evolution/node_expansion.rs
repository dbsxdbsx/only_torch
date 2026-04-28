/*
 * @Author       : 老董
 * @Date         : 2026-03-25
 * @Description  : NodeLevel 基因组的节点展开工具
 *
 * 本模块只服务 NodeLevel 构造、变异与可视化 block 语义：
 * - InnovationCounter : 统一创新号分配入口（单调递增）
 * - expand_*         : 将高层 block 意图展开为 NodeGene 列表
 * - decompose_conv2d_to_feature_maps : 将 Conv2d block 重写为 FM 子图
 */

use crate::nn::descriptor::NodeTypeDescriptor;

use super::fm_ops::next_fm_id;
use super::gene::{ActivationType, PoolType};
use super::node_gene::NodeGene;

// ==================== InnovationCounter ====================

/// 统一创新号分配器。
#[derive(Debug, Clone)]
pub struct InnovationCounter(u64);

impl InnovationCounter {
    /// 从指定起始值创建计数器。
    pub fn new(start: u64) -> Self {
        Self(start)
    }

    /// 分配下一个创新号（单调递增）。
    pub fn next(&mut self) -> u64 {
        let id = self.0;
        self.0 += 1;
        id
    }

    /// 当前下一个将分配的值（不消耗）。
    pub fn peek(&self) -> u64 {
        self.0
    }
}

// ==================== 错误类型 ====================

/// NodeLevel 展开或图描述转换错误。
#[derive(Debug)]
pub enum NodeExpansionError {
    /// 维度推导失败。
    DimensionError(String),
    /// 基因组包含无效的节点配置。
    InvalidGenome(String),
}

impl std::fmt::Display for NodeExpansionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionError(msg) => write!(f, "维度推导失败：{msg}"),
            Self::InvalidGenome(msg) => write!(f, "无效基因组：{msg}"),
        }
    }
}

impl std::error::Error for NodeExpansionError {}

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

/// 展开 DeformableConv2d 层 → offset predictor + DeformableConv2d + bias。
///
/// v1 采用 offset-only 结构，offset predictor 与主卷积使用同样的 same-padding
/// kernel，输出 offset 形状为 `[1, 2 * deformable_groups * k * k, H, W]`。
pub fn expand_deformable_conv2d(
    input_id: u64,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    input_spatial: (usize, usize),
    deformable_groups: usize,
    block_id: u64,
    counter: &mut InnovationCounter,
) -> Vec<NodeGene> {
    let k = kernel_size;
    let padding = k / 2;
    let (h, w) = input_spatial;
    let h_out = (h + 2 * padding - k) + 1;
    let w_out = (w + 2 * padding - k) + 1;
    let offset_channels = 2 * deformable_groups * k * k;

    let offset_kernel_id = counter.next();
    let offset_conv_id = counter.next();
    let offset_bias_id = counter.next();
    let offset_add_id = counter.next();
    let kernel_id = counter.next();
    let deform_id = counter.next();
    let bias_id = counter.next();
    let add_id = counter.next();
    let bid = Some(block_id);

    vec![
        NodeGene::new(
            offset_kernel_id,
            NodeTypeDescriptor::Parameter,
            vec![offset_channels, in_channels, k, k],
            vec![],
            bid,
        ),
        NodeGene::new(
            offset_conv_id,
            NodeTypeDescriptor::Conv2d {
                stride: (1, 1),
                padding: (padding, padding),
                dilation: (1, 1),
            },
            vec![1, offset_channels, h_out, w_out],
            vec![input_id, offset_kernel_id],
            bid,
        ),
        NodeGene::new(
            offset_bias_id,
            NodeTypeDescriptor::Parameter,
            vec![1, offset_channels, 1, 1],
            vec![],
            bid,
        ),
        NodeGene::new(
            offset_add_id,
            NodeTypeDescriptor::Add,
            vec![1, offset_channels, h_out, w_out],
            vec![offset_conv_id, offset_bias_id],
            bid,
        ),
        NodeGene::new(
            kernel_id,
            NodeTypeDescriptor::Parameter,
            vec![out_channels, in_channels, k, k],
            vec![],
            bid,
        ),
        NodeGene::new(
            deform_id,
            NodeTypeDescriptor::DeformableConv2d {
                stride: (1, 1),
                padding: (padding, padding),
                dilation: (1, 1),
                deformable_groups,
            },
            vec![1, out_channels, h_out, w_out],
            vec![input_id, kernel_id, offset_add_id],
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
            vec![deform_id, bias_id],
            bid,
        ),
    ]
}

/// 展开 ConvTranspose2d 层 → Parameter(kernel) + ConvTranspose2d + Parameter(bias) + Add
///
/// 形状约定：
/// - kernel: [in_channels, out_channels, k, k]
/// - deconv_out: [1, out_channels, H_out, W_out]
/// - bias: [1, out_channels, 1, 1]
///
/// 所有节点共享同一个 block_id。
pub fn expand_conv_transpose2d(
    input_id: u64,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    input_spatial: (usize, usize),
    block_id: u64,
    counter: &mut InnovationCounter,
) -> Vec<NodeGene> {
    let k = kernel_size;
    let (h, w) = input_spatial;
    let h_out = ((h.saturating_sub(1)) * stride + k + output_padding)
        .saturating_sub(2 * padding)
        .max(1);
    let w_out = ((w.saturating_sub(1)) * stride + k + output_padding)
        .saturating_sub(2 * padding)
        .max(1);

    let kernel_id = counter.next();
    let deconv_id = counter.next();
    let bias_id = counter.next();
    let add_id = counter.next();

    let bid = Some(block_id);
    vec![
        NodeGene::new(
            kernel_id,
            NodeTypeDescriptor::Parameter,
            vec![in_channels, out_channels, k, k],
            vec![],
            bid,
        ),
        NodeGene::new(
            deconv_id,
            NodeTypeDescriptor::ConvTranspose2d {
                stride: (stride, stride),
                padding: (padding, padding),
                output_padding: (output_padding, output_padding),
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
            vec![deconv_id, bias_id],
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
        NodeGene::new(
            gamma_id,
            NodeTypeDescriptor::Parameter,
            param_shape.clone(),
            vec![],
            bid,
        ),
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
        NodeGene::new(
            mul_id,
            NodeTypeDescriptor::Multiply,
            input_shape.clone(),
            vec![bn_id, gamma_id],
            bid,
        ),
        NodeGene::new(
            beta_id,
            NodeTypeDescriptor::Parameter,
            param_shape,
            vec![],
            bid,
        ),
        NodeGene::new(
            add_id,
            NodeTypeDescriptor::Add,
            input_shape,
            vec![mul_id, beta_id],
            bid,
        ),
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
        NodeGene::new(
            gamma_id,
            NodeTypeDescriptor::Parameter,
            param_shape.clone(),
            vec![],
            bid,
        ),
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
        NodeGene::new(
            mul_id,
            NodeTypeDescriptor::Multiply,
            input_shape.clone(),
            vec![ln_id, gamma_id],
            bid,
        ),
        NodeGene::new(
            beta_id,
            NodeTypeDescriptor::Parameter,
            param_shape,
            vec![],
            bid,
        ),
        NodeGene::new(
            add_id,
            NodeTypeDescriptor::Add,
            input_shape,
            vec![mul_id, beta_id],
            bid,
        ),
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
        NodeGene::new(
            gamma_id,
            NodeTypeDescriptor::Parameter,
            param_shape,
            vec![],
            bid,
        ),
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
        NodeGene::new(
            mul_id,
            NodeTypeDescriptor::Multiply,
            input_shape,
            vec![rn_id, gamma_id],
            bid,
        ),
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

pub fn decompose_conv2d_to_feature_maps(
    nodes: &mut Vec<NodeGene>,
    counter: &mut InnovationCounter,
) {
    use std::collections::HashMap;

    // 1. 找到所有 Conv2d 层块
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
        decompose_conv2d_block(nodes, block, counter, &mut fm_counter, &mut node_map);
    }

    // 清理已禁用的旧层块节点
    nodes.retain(|n| n.enabled);
}

/// Conv2d 层块信息
struct Conv2dBlock {
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

/// 查找基因组中所有 Conv2d 层块
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
            NodeTypeDescriptor::Conv2d {
                stride,
                padding,
                dilation,
            } => (*stride, *padding, *dilation),
            _ => continue,
        };

        let block_id = match n.block_id {
            Some(bid) => bid,
            None => continue, // 非层块
        };

        // 已经是 FM 边（fm_id = None 但上游是 FM 节点）的跳过
        if n.parents
            .iter()
            .any(|&pid| node_map.get(&pid).map_or(false, |p| p.fm_id.is_some()))
        {
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

        let (out_channels, in_channels, kernel_h, kernel_w) = (
            kernel_shape[0],
            kernel_shape[1],
            kernel_shape[2],
            kernel_shape[3],
        );

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
        for m in nodes
            .iter()
            .filter(|m| m.enabled && m.block_id == Some(block_id))
        {
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

/// 将单个 Conv2d 层块分解为 FM 节点和边
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

        let output_id = if block.bias_id.is_some() {
            let bias_block_id = counter.next();
            let bias_id = counter.next();
            nodes.push(NodeGene::new(
                bias_id,
                NodeTypeDescriptor::Parameter,
                vec![1, 1, 1, 1],
                vec![],
                Some(bias_block_id),
            ));

            let add_id = counter.next();
            let mut bias_add = NodeGene::new(
                add_id,
                NodeTypeDescriptor::Add,
                vec![1, 1, block.output_h, block.output_w],
                vec![agg_id, bias_id],
                Some(bias_block_id),
            );
            bias_add.fm_id = Some(fm_id);
            nodes.push(bias_add);
            add_id
        } else {
            agg_id
        };

        // FM 输出节点标记
        if let Some(node) = nodes.iter_mut().find(|n| n.innovation_number == output_id) {
            node.fm_id = Some(fm_id);
        }

        output_fm_output_ids.push(output_id);
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

    // 5. 禁用原层块节点
    for n in nodes.iter_mut() {
        if n.block_id == Some(block.block_id) && n.fm_id.is_none() {
            n.enabled = false;
        }
    }
}
