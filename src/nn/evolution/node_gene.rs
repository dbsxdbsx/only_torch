/*
 * @Author       : 老董
 * @Date         : 2026-03-25
 * @Description  : 节点级基因组数据结构
 *
 * 核心类型：
 * - NodeGene     : 单个节点级基因，对应计算图的一个节点
 * - GenomeAnalysis: 不可变分析快照，调用方式为 genome.analyze() -> GenomeAnalysis
 *                   mutation / builder / serializer 三方共同依赖此结构，不各自重复推导
 * - infer_output_shape: 覆盖全部 NodeTypeDescriptor 变体的形状推导函数
 * - GenomeKind   : 区分基因组当前使用旧层级表示还是新节点级表示
 *
 * 设计原则：
 * - 本文件只定义节点级类型，不修改现有 LayerGene / NetworkGenome / SkipEdge
 * - GenomeAnalysis 是只读快照，每次需要分析时重新计算，避免"改了基因组但忘刷新 analysis"的隐式 bug
 * - block_id 内联到 NodeGene，而不是单独维护 TemplateGroup 列表
 */

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::nn::descriptor::NodeTypeDescriptor;

use super::gene::ShapeDomain;

// ==================== GenomeKind ====================

/// 基因组当前使用的表示方式（`NetworkGenome` 内用于区分 LayerLevel / NodeLevel）
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenomeKind {
    /// 旧层级表示（LayerGene + SkipEdge，当前版本默认）
    LayerLevel,
    /// 节点级表示（NodeGene）
    NodeLevel,
}

// ==================== AnalysisError ====================

/// 基因组静态分析错误
#[derive(Debug, Clone, PartialEq)]
pub enum AnalysisError {
    /// 拓扑图中存在环（State 节点除外，State 作为叶节点处理）
    CycleDetected,
    /// 节点引用了不存在的父节点
    MissingParent { node_id: u64, parent_id: u64 },
    /// 节点的形状与父节点形状不兼容
    IncompatibleShapes { node_id: u64, message: String },
    /// 没有任何启用的节点
    Empty,
    /// 循环边引用了不存在的源节点
    RecurrentMissingSource { node_id: u64, source_id: u64 },
    /// 循环边的权重参数节点无效（不存在或非 Parameter 类型）
    RecurrentInvalidWeight { node_id: u64, weight_param_id: u64, reason: String },
    /// 循环边的形状不兼容（权重矩阵维度与源/目标不匹配）
    RecurrentShapeMismatch { node_id: u64, source_id: u64, message: String },
    /// 循环边与 cell-based 循环范式冲突
    RecurrentParadigmConflict { node_id: u64, message: String },
}

impl std::fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CycleDetected => write!(f, "图中检测到环"),
            Self::MissingParent { node_id, parent_id } => {
                write!(f, "节点 {node_id} 引用了不存在的父节点 {parent_id}")
            }
            Self::IncompatibleShapes { node_id, message } => {
                write!(f, "节点 {node_id} 形状不兼容：{message}")
            }
            Self::Empty => write!(f, "基因组中没有启用的节点"),
            Self::RecurrentMissingSource { node_id, source_id } => {
                write!(f, "节点 {node_id} 的循环边引用了不存在的源节点 {source_id}")
            }
            Self::RecurrentInvalidWeight { node_id, weight_param_id, reason } => {
                write!(f, "节点 {node_id} 的循环边权重参数 {weight_param_id} 无效：{reason}")
            }
            Self::RecurrentShapeMismatch { node_id, source_id, message } => {
                write!(f, "节点 {node_id} ← 源 {source_id} 循环边形状不兼容：{message}")
            }
            Self::RecurrentParadigmConflict { node_id, message } => {
                write!(f, "节点 {node_id} 循环范式冲突：{message}")
            }
        }
    }
}

impl std::error::Error for AnalysisError {}

// ==================== RecurrentEdge ====================

/// 单条循环边：描述从 `source_id` 到目标节点的时延连接
///
/// 运行时语义：`target_input += weight @ prev_activation[source_id]`
/// 其中 `weight` 由 `weight_param_id` 指向的 Parameter 节点持有。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecurrentEdge {
    /// 提供上一时间步输出的源节点创新号
    pub source_id: u64,
    /// 持有循环权重矩阵 `[target_dim, source_dim]` 的 Parameter 节点创新号
    pub weight_param_id: u64,
}

// ==================== NodeGene ====================

/// 节点级基因：演化系统的最小可操作单元
///
/// 每个 NodeGene 对应计算图中的一个节点，`node_type` 直接使用 `NodeTypeDescriptor`
/// 以实现 1:1 对齐，消除演化层和图 IR 层之间的抽象断层。
///
/// # 形状规则
/// - `Parameter`、`BasicInput`、`TargetInput`、`State` 节点：`output_shape` 是权威值，
///   由创建该节点的模板或外部调用方显式指定
/// - 其他计算节点：`output_shape` 是声明值，`GenomeAnalysis` 会验证其与父节点形状的一致性
///
/// # block_id
/// 模板展开的节点组共享同一个 `block_id`，细粒度单节点变异产生的节点 `block_id = None`。
/// 这既允许 Grow/Shrink/Remove 以"组"为单位操作，也为将来的 NEAT crossover 打基础
/// （交叉时以 block 为单位对齐，不会把模板组的节点拆散到不同父本）。
///
/// # recurrent_parents（循环边）
/// 存储指向本节点的时延循环连接 `(source_id, weight_param_id)`：
/// - `source_id`: 提供上一时间步输出的源节点创新号
/// - `weight_param_id`: 持有循环权重矩阵的 Parameter 节点创新号
/// 循环边不参与前向 DAG 拓扑排序，仅在时序展开时注入上一时间步的激活值。
/// 此字段仅在序列模式（`seq_len.is_some()`）的 edge-based 循环范式中有效，
/// 与 cell-based 循环（CellRnn/CellLstm/CellGru）互斥。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeGene {
    /// NEAT 风格创新号：单调递增，跨代唯一
    pub innovation_number: u64,
    /// 节点类型（直接对齐 NodeTypeDescriptor）
    pub node_type: NodeTypeDescriptor,
    /// 该节点的输出形状
    pub output_shape: Vec<usize>,
    /// 父节点创新号列表（定义数据流方向）
    pub parents: Vec<u64>,
    /// 是否启用（NEAT 风格禁用机制，禁用节点不参与构图）
    pub enabled: bool,
    /// 模板组标识：`Some(id)` 属于某个高层模板，`None` 为独立节点
    pub block_id: Option<u64>,
    /// 循环边列表：`(source_id, weight_param_id)` 对
    ///
    /// - `source_id`: 上一时间步提供激活的源节点
    /// - `weight_param_id`: 对应的 Parameter 节点，持有循环权重 `[target_dim, source_dim]`
    /// - 仅序列模式有效，与 cell-based 循环互斥
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recurrent_parents: Vec<RecurrentEdge>,
}

impl NodeGene {
    /// 创建启用状态的节点基因（无循环边）
    pub fn new(
        innovation_number: u64,
        node_type: NodeTypeDescriptor,
        output_shape: Vec<usize>,
        parents: Vec<u64>,
        block_id: Option<u64>,
    ) -> Self {
        Self {
            innovation_number,
            node_type,
            output_shape,
            parents,
            enabled: true,
            block_id,
            recurrent_parents: Vec::new(),
        }
    }

    /// 是否为参数节点（持有可训练权重）
    pub fn is_parameter(&self) -> bool {
        matches!(self.node_type, NodeTypeDescriptor::Parameter)
    }

    /// 是否为数据输入节点
    pub fn is_input(&self) -> bool {
        matches!(
            self.node_type,
            NodeTypeDescriptor::BasicInput | NodeTypeDescriptor::TargetInput
        )
    }

    /// 是否为状态节点（时序循环，作为叶节点处理）
    pub fn is_state(&self) -> bool {
        matches!(self.node_type, NodeTypeDescriptor::State)
    }

    /// 是否为叶节点（形状由自身存储，不依赖父节点推导）
    pub fn is_leaf(&self) -> bool {
        self.is_parameter() || self.is_input() || self.is_state()
    }

    /// 该节点贡献的参数量（仅 Parameter 节点有意义）
    pub fn param_count(&self) -> usize {
        if self.is_parameter() {
            self.output_shape.iter().product()
        } else {
            0
        }
    }
}

// ==================== 形状推导 ====================

/// 从节点类型和父节点输出形状推导当前节点的输出形状
///
/// # 参数
/// - `node_type`: 节点类型
/// - `parent_shapes`: 父节点输出形状列表，按 `NodeGene.parents` 顺序排列
///
/// # 注意
/// `Parameter`、`BasicInput`、`TargetInput`、`State` 节点的形状是权威值，
/// 直接使用 `NodeGene.output_shape`，不应调用此函数。
/// 此函数返回 `Err` 对这些变体以提示调用方。
pub fn infer_output_shape(
    node_type: &NodeTypeDescriptor,
    parent_shapes: &[&Vec<usize>],
) -> Result<Vec<usize>, String> {
    use NodeTypeDescriptor as NT;

    match node_type {
        // ── 叶节点：形状存储于 NodeGene.output_shape ──
        NT::BasicInput | NT::TargetInput | NT::Parameter | NT::State => {
            Err("叶节点形状由 NodeGene.output_shape 提供，不通过此函数推导".to_string())
        }

        // ── 恒等透传 ──
        NT::Identity | NT::Detach => {
            require_n(1, parent_shapes)?;
            Ok(parent_shapes[0].clone())
        }

        // ── 一元逐元素操作（形状不变）──
        NT::Sigmoid
        | NT::Softmax
        | NT::Tanh
        | NT::LeakyReLU { .. }
        | NT::Ln
        | NT::LogSoftmax
        | NT::Dropout { .. }
        | NT::Sign
        | NT::Abs
        | NT::SoftPlus
        | NT::Step
        | NT::Negate
        | NT::ReLU
        | NT::Gelu
        | NT::Swish
        | NT::Elu { .. }
        | NT::Selu
        | NT::Mish
        | NT::HardSwish
        | NT::HardSigmoid
        | NT::ReLU6
        | NT::HardTanh { .. }
        | NT::Exp
        | NT::Sqrt
        | NT::Log10
        | NT::Log2
        | NT::Pow { .. }
        | NT::Square
        | NT::Reciprocal
        | NT::Clip { .. } => {
            require_n(1, parent_shapes)?;
            Ok(parent_shapes[0].clone())
        }

        // ── 归一化操作（形状不变）──
        NT::BatchNormOp { .. } | NT::LayerNormOp { .. } | NT::RMSNormOp { .. } => {
            require_n(1, parent_shapes)?;
            Ok(parent_shapes[0].clone())
        }

        // ── ZerosLike（与输入同形）──
        NT::ZerosLike => {
            require_n(1, parent_shapes)?;
            Ok(parent_shapes[0].clone())
        }

        // ── 二元逐元素操作（遵循广播规则）──
        NT::Add | NT::Subtract | NT::Divide | NT::Multiply | NT::Maximum | NT::Minimum => {
            if parent_shapes.len() < 2 {
                return Err(format!(
                    "需要至少 2 个父节点，实际 {}",
                    parent_shapes.len()
                ));
            }
            let mut out = parent_shapes[0].clone();
            for (i, ps) in parent_shapes.iter().enumerate().skip(1) {
                out = broadcast_shape(&out, ps).map_err(|msg| {
                    format!(
                        "父节点 0..{} 无法广播到父节点 {i} 形状 {ps:?}: {msg}",
                        i - 1
                    )
                })?;
            }
            Ok(out)
        }

        // ── WhereCond（输出与第一个输入同形）──
        NT::WhereCond { .. } => {
            require_n(1, parent_shapes)?;
            Ok(parent_shapes[0].clone())
        }

        // ── MatMul: [..., m, k] × [..., k, n] → [..., m, n] ──
        NT::MatMul => {
            require_n(2, parent_shapes)?;
            let a = parent_shapes[0];
            let b = parent_shapes[1];
            if a.len() < 2 {
                return Err(format!("MatMul 左操作数需要至少 2D，得到 {a:?}"));
            }
            if b.len() < 2 {
                return Err(format!("MatMul 右操作数需要至少 2D，得到 {b:?}"));
            }
            let m = a[a.len() - 2];
            let k_a = a[a.len() - 1];
            let k_b = b[b.len() - 2];
            let n = b[b.len() - 1];
            if k_a != k_b {
                return Err(format!(
                    "MatMul 内维度不匹配：A[...,-1]={k_a} vs B[...,-2]={k_b}"
                ));
            }
            let mut out = a[..a.len() - 2].to_vec();
            out.push(m);
            out.push(n);
            Ok(out)
        }

        // ── Sum / Mean（归约）──
        NT::Sum { axis } | NT::Mean { axis } => {
            require_n(1, parent_shapes)?;
            let s = parent_shapes[0];
            match axis {
                None => Ok(vec![1]),
                Some(ax) => {
                    let ax = *ax;
                    if ax >= s.len() {
                        return Err(format!("axis {ax} 超出形状 {s:?}"));
                    }
                    let mut out = s.clone();
                    out.remove(ax);
                    if out.is_empty() {
                        out.push(1);
                    }
                    Ok(out)
                }
            }
        }

        // ── Amax / Amin ──
        NT::Amax { axis } | NT::Amin { axis } => {
            require_n(1, parent_shapes)?;
            let s = parent_shapes[0];
            let ax = *axis;
            if ax >= s.len() {
                return Err(format!("axis {ax} 超出形状 {s:?}"));
            }
            let mut out = s.clone();
            out.remove(ax);
            if out.is_empty() {
                out.push(1);
            }
            Ok(out)
        }

        // ── Reshape ──
        NT::Reshape { target_shape } => {
            require_n(1, parent_shapes)?;
            let in_size: usize = parent_shapes[0].iter().product();
            let out_size: usize = target_shape.iter().product();
            if in_size != out_size {
                return Err(format!(
                    "Reshape 元素数不匹配：输入 {in_size} vs 目标 {out_size}"
                ));
            }
            Ok(target_shape.clone())
        }

        // ── Flatten ──
        NT::Flatten { keep_first_dim } => {
            require_n(1, parent_shapes)?;
            let s = parent_shapes[0];
            if s.is_empty() {
                return Ok(vec![1]);
            }
            if *keep_first_dim {
                if s.len() < 2 {
                    return Ok(s.clone());
                }
                let flat: usize = s[1..].iter().product();
                Ok(vec![s[0], flat])
            } else {
                Ok(vec![s.iter().product()])
            }
        }

        // ── Conv2d: [N,C_in,H,W] × [C_out,C_in,kH,kW] → [N,C_out,H_out,W_out] ──
        NT::Conv2d { stride, padding } => {
            require_n(2, parent_shapes)?;
            let inp = parent_shapes[0];
            let ker = parent_shapes[1];
            if inp.len() < 4 {
                return Err(format!("Conv2d 输入需要 4D，得到 {inp:?}"));
            }
            if ker.len() < 4 {
                return Err(format!("Conv2d 权重需要 4D，得到 {ker:?}"));
            }
            let (n, c_out) = (inp[0], ker[0]);
            let h_out = (inp[2] + 2 * padding.0 - ker[2]) / stride.0 + 1;
            let w_out = (inp[3] + 2 * padding.1 - ker[3]) / stride.1 + 1;
            Ok(vec![n, c_out, h_out, w_out])
        }

        // ── MaxPool2d / AvgPool2d ──
        NT::MaxPool2d { kernel_size, stride } | NT::AvgPool2d { kernel_size, stride } => {
            require_n(1, parent_shapes)?;
            let s = parent_shapes[0];
            if s.len() < 4 {
                return Err(format!("Pool2d 输入需要 4D，得到 {s:?}"));
            }
            let h_out = if s[2] >= kernel_size.0 {
                (s[2] - kernel_size.0) / stride.0 + 1
            } else {
                1
            };
            let w_out = if s[3] >= kernel_size.1 {
                (s[3] - kernel_size.1) / stride.1 + 1
            } else {
                1
            };
            Ok(vec![s[0], s[1], h_out, w_out])
        }

        // ── Select: 沿 axis 取固定索引，移除该维度 ──
        NT::Select { axis, .. } => {
            require_n(1, parent_shapes)?;
            let s = parent_shapes[0];
            let ax = *axis;
            if ax >= s.len() {
                return Err(format!("Select axis {ax} 超出形状 {s:?}"));
            }
            let mut out = s.clone();
            out.remove(ax);
            if out.is_empty() {
                out.push(1);
            }
            Ok(out)
        }

        // ── Gather: parents[0]=data, parents[1]=indices ──
        NT::Gather { dim } => {
            require_n(2, parent_shapes)?;
            let data = parent_shapes[0];
            let indices = parent_shapes[1];
            let ax = *dim;
            if ax >= data.len() {
                return Err(format!("Gather dim {ax} 超出 data 形状 {data:?}"));
            }
            let mut out = data[..ax].to_vec();
            out.extend_from_slice(indices);
            out.extend_from_slice(&data[ax + 1..]);
            Ok(out)
        }

        // ── Stack: 沿新 axis 插入维度，值 = n_parents ──
        NT::Stack { axis } => {
            if parent_shapes.is_empty() {
                return Err("Stack 至少需要 1 个父节点".to_string());
            }
            let base = parent_shapes[0];
            for (i, ps) in parent_shapes.iter().enumerate().skip(1) {
                if *ps != base {
                    return Err(format!(
                        "Stack 所有输入形状必须相同：base={base:?}, parent{i}={ps:?}"
                    ));
                }
            }
            let ax = *axis;
            if ax > base.len() {
                return Err(format!(
                    "Stack axis {ax} 超出合法范围 0..={}",
                    base.len()
                ));
            }
            let mut out = base[..ax].to_vec();
            out.push(parent_shapes.len());
            out.extend_from_slice(&base[ax..]);
            Ok(out)
        }

        // ── Concat: 沿 axis 拼接 ──
        NT::Concat { axis } => {
            if parent_shapes.is_empty() {
                return Err("Concat 至少需要 1 个父节点".to_string());
            }
            let ax = *axis;
            let base = parent_shapes[0];
            if ax >= base.len() {
                return Err(format!("Concat axis {ax} 超出形状 {base:?}"));
            }
            let mut concat_dim = base[ax];
            for (i, ps) in parent_shapes.iter().enumerate().skip(1) {
                if ps.len() != base.len() {
                    return Err(format!(
                        "Concat 维度数不一致：base {base:?} vs parent{i} {ps:?}"
                    ));
                }
                for (j, (&b, &p)) in base.iter().zip(ps.iter()).enumerate() {
                    if j != ax && b != p {
                        return Err(format!(
                            "Concat 非拼接维度 {j} 形状不一致：{b} vs {p}"
                        ));
                    }
                }
                concat_dim += ps[ax];
            }
            let mut out = base.clone();
            out[ax] = concat_dim;
            Ok(out)
        }

        // ── 损失函数（输出标量）──
        NT::BCE { .. } | NT::Huber { .. } | NT::MAE { .. } | NT::MSE { .. }
        | NT::SoftmaxCrossEntropy => Ok(vec![1]),

        // ── Narrow: 沿 axis 取 [start, start+length) ──
        NT::Narrow { axis, length, .. } => {
            require_n(1, parent_shapes)?;
            let s = parent_shapes[0];
            let ax = *axis;
            if ax >= s.len() {
                return Err(format!("Narrow axis {ax} 超出形状 {s:?}"));
            }
            let mut out = s.clone();
            out[ax] = *length;
            Ok(out)
        }

        // ── Permute ──
        NT::Permute { dims } => {
            require_n(1, parent_shapes)?;
            let s = parent_shapes[0];
            if dims.len() != s.len() {
                return Err(format!(
                    "Permute dims {dims:?} 长度与形状 {s:?} 不一致"
                ));
            }
            Ok(dims.iter().map(|&d| s[d]).collect())
        }

        // ── Pad ──
        NT::Pad { paddings, .. } => {
            require_n(1, parent_shapes)?;
            let s = parent_shapes[0];
            if paddings.len() != s.len() {
                return Err(format!(
                    "Pad paddings {paddings:?} 长度与形状 {s:?} 不一致"
                ));
            }
            Ok(s.iter()
                .zip(paddings.iter())
                .map(|(&d, &(lo, hi))| d + lo + hi)
                .collect())
        }

        // ── Repeat ──
        NT::Repeat { repeats } => {
            require_n(1, parent_shapes)?;
            let s = parent_shapes[0];
            if repeats.len() != s.len() {
                return Err(format!(
                    "Repeat repeats {repeats:?} 长度与形状 {s:?} 不一致"
                ));
            }
            Ok(s.iter().zip(repeats.iter()).map(|(&d, &r)| d * r).collect())
        }

        // ── TopK: 沿 axis 保留前 k 个 ──
        NT::TopK { k, axis, .. } => {
            require_n(1, parent_shapes)?;
            let s = parent_shapes[0];
            let ax = *axis;
            if ax >= s.len() {
                return Err(format!("TopK axis {ax} 超出形状 {s:?}"));
            }
            let mut out = s.clone();
            out[ax] = *k;
            Ok(out)
        }

        // ── SortNode: 形状不变 ──
        NT::SortNode { .. } => {
            require_n(1, parent_shapes)?;
            Ok(parent_shapes[0].clone())
        }

        // ── Cell* 复合循环节点（RNN / LSTM / GRU）──
        //
        // parents[0] 是输入张量，其余是参数节点（叶节点，不通过此函数推导形状）。
        // 输出形状：
        //   - !return_sequences → [1, hidden_size]
        //   - return_sequences  → [1, seq_len.max(1), hidden_size]
        NT::CellRnn {
            hidden_size,
            return_sequences,
            seq_len,
            ..
        }
        | NT::CellLstm {
            hidden_size,
            return_sequences,
            seq_len,
            ..
        }
        | NT::CellGru {
            hidden_size,
            return_sequences,
            seq_len,
            ..
        } => {
            if parent_shapes.is_empty() {
                return Err("CellRnn/CellLstm/CellGru 至少需要 1 个父节点（输入张量）".to_string());
            }
            if *return_sequences {
                Ok(vec![1, (*seq_len).max(1), *hidden_size])
            } else {
                Ok(vec![1, *hidden_size])
            }
        }
    }
}

/// 辅助：检查父节点数量不少于 n
fn require_n(n: usize, parents: &[&Vec<usize>]) -> Result<(), String> {
    if parents.len() < n {
        Err(format!("需要 {n} 个父节点，实际 {}", parents.len()))
    } else {
        Ok(())
    }
}

/// 计算两个张量形状的广播结果。
fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let max_rank = a.len().max(b.len());
    let mut out_rev = Vec::with_capacity(max_rank);

    for idx in 0..max_rank {
        let da = a.iter().rev().nth(idx).copied().unwrap_or(1);
        let db = b.iter().rev().nth(idx).copied().unwrap_or(1);
        if da == db || da == 1 || db == 1 {
            out_rev.push(da.max(db));
        } else {
            return Err(format!("形状 {a:?} 与 {b:?} 在倒数第 {} 维不兼容", idx + 1));
        }
    }

    out_rev.reverse();
    Ok(out_rev)
}

// ==================== 域推导 ====================

/// 从节点类型和父节点域推导当前节点的输出域
pub fn infer_domain(
    node_type: &NodeTypeDescriptor,
    parent_domains: &[ShapeDomain],
) -> ShapeDomain {
    use NodeTypeDescriptor as NT;
    use ShapeDomain::*;

    match node_type {
        // 空间 → 空间
        NT::Conv2d { .. } | NT::MaxPool2d { .. } | NT::AvgPool2d { .. } => Spatial,
        // 空间 → 平坦
        NT::Flatten { .. } => Flat,
        // 输入节点默认平坦（调用方会根据 genome 的输入模式修正）
        NT::BasicInput | NT::TargetInput => Flat,
        // 循环单元：return_sequences → Sequence，否则 → Flat
        NT::CellRnn { return_sequences, .. }
        | NT::CellLstm { return_sequences, .. }
        | NT::CellGru { return_sequences, .. } => {
            if *return_sequences { Sequence } else { Flat }
        }
        // 其他：透传第一个父节点的域
        _ => parent_domains.first().copied().unwrap_or(Flat),
    }
}

// ==================== GenomeAnalysis ====================

/// 节点级基因组的不可变静态分析快照
///
/// 通过 `GenomeAnalysis::compute(nodes, input_id, input_shape)` 创建。
/// 每次基因组改变后应重新调用 compute，而不是修改已有的快照。
///
/// mutation / builder / serializer 全部从此结构读取分析结论，
/// 不各自重复实现形状推导、域推导或参数量统计。
#[derive(Debug)]
pub struct GenomeAnalysis {
    /// 启用节点按拓扑序排列的创新号（不含虚拟输入节点 input_id）
    pub topo_order: Vec<u64>,
    /// 每个节点（含虚拟输入）的推导输出形状
    pub output_shapes: HashMap<u64, Vec<usize>>,
    /// 每个节点的输出域
    pub domains: HashMap<u64, ShapeDomain>,
    /// 总可训练参数量
    pub param_count: usize,
    /// 启用节点数量（含 Parameter、BasicInput 等所有启用节点）
    pub enabled_node_count: usize,
    /// 参数节点数量
    pub param_node_count: usize,
    /// 是否通过合法性校验（无硬错误）
    pub is_valid: bool,
    /// 硬错误列表（任何一条存在则 is_valid = false）
    pub errors: Vec<AnalysisError>,
    /// 基因组是否包含 edge-based 循环边
    pub has_recurrent_edges: bool,
}

impl GenomeAnalysis {
    /// 对节点列表执行静态分析，返回不可变快照
    ///
    /// # 参数
    /// - `nodes`: 所有节点基因（启用和禁用的都传入，内部会过滤）
    /// - `input_id`: 虚拟输入节点的创新号（通常是 `INPUT_INNOVATION = 0`）
    /// - `input_shape`: 输入数据的形状（含 batch 维，如 `[1, input_dim]`）
    /// - `input_domain`: 输入数据的 ShapeDomain
    pub fn compute(
        nodes: &[NodeGene],
        input_id: u64,
        input_shape: Vec<usize>,
        input_domain: ShapeDomain,
    ) -> Self {
        let mut errors = Vec::new();
        let mut output_shapes: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut domains: HashMap<u64, ShapeDomain> = HashMap::new();

        // 注入虚拟输入节点
        output_shapes.insert(input_id, input_shape);
        domains.insert(input_id, input_domain);

        let enabled: Vec<&NodeGene> = nodes.iter().filter(|n| n.enabled).collect();

        if enabled.is_empty() {
            errors.push(AnalysisError::Empty);
            return Self::failed(errors, 0, 0, 0, output_shapes, domains);
        }

        // 拓扑排序
        let topo_order = match topological_sort(&enabled, input_id) {
            Ok(order) => order,
            Err(e) => {
                errors.push(e);
                return Self::failed(
                    errors,
                    enabled.len(),
                    0,
                    0,
                    output_shapes,
                    domains,
                );
            }
        };

        let node_map: HashMap<u64, &NodeGene> =
            enabled.iter().map(|n| (n.innovation_number, *n)).collect();

        let mut param_count = 0usize;
        let mut param_node_count = 0usize;

        // 按拓扑序逐节点推导形状和域
        for &id in &topo_order {
            let node = match node_map.get(&id) {
                Some(n) => *n,
                None => continue,
            };

            // 叶节点：形状和域由自身提供
            if node.is_leaf() {
                output_shapes.insert(id, node.output_shape.clone());
                domains.insert(id, ShapeDomain::Flat);
                if node.is_parameter() {
                    param_count += node.param_count();
                    param_node_count += 1;
                }
                continue;
            }

            // 收集父节点形状
            let mut parent_shapes: Vec<&Vec<usize>> = Vec::new();
            let mut parent_doms: Vec<ShapeDomain> = Vec::new();
            let mut missing = false;

            for &pid in &node.parents {
                match output_shapes.get(&pid) {
                    Some(s) => {
                        parent_shapes.push(s);
                        parent_doms.push(domains.get(&pid).copied().unwrap_or(ShapeDomain::Flat));
                    }
                    None => {
                        errors.push(AnalysisError::MissingParent {
                            node_id: id,
                            parent_id: pid,
                        });
                        missing = true;
                        break;
                    }
                }
            }

            if missing {
                // 回退：使用节点自身存储的形状
                output_shapes.insert(id, node.output_shape.clone());
                domains.insert(id, ShapeDomain::Flat);
                continue;
            }

            // 推导输出形状
            let shape = match infer_output_shape(&node.node_type, &parent_shapes) {
                Ok(s) => s,
                Err(msg) => {
                    errors.push(AnalysisError::IncompatibleShapes {
                        node_id: id,
                        message: msg,
                    });
                    node.output_shape.clone()
                }
            };
            output_shapes.insert(id, shape);

            // 推导域
            domains.insert(id, infer_domain(&node.node_type, &parent_doms));
        }

        // ── 循环边验证 ──
        let has_recurrent_edges = enabled.iter().any(|n| !n.recurrent_parents.is_empty());

        if has_recurrent_edges {
            // 检测是否存在 cell-based 循环节点（范式互斥检查）
            let has_cell_recurrent = enabled.iter().any(|n| {
                matches!(
                    n.node_type,
                    NodeTypeDescriptor::CellRnn { .. }
                        | NodeTypeDescriptor::CellLstm { .. }
                        | NodeTypeDescriptor::CellGru { .. }
                )
            });

            for node in &enabled {
                if node.recurrent_parents.is_empty() {
                    continue;
                }

                let target_id = node.innovation_number;

                // 范式互斥：edge-based 循环与 cell-based 循环不共存
                if has_cell_recurrent {
                    errors.push(AnalysisError::RecurrentParadigmConflict {
                        node_id: target_id,
                        message: "edge-based 循环边与 cell-based 循环节点不可共存".into(),
                    });
                }

                for edge in &node.recurrent_parents {
                    // 源节点必须存在
                    let source_node = enabled
                        .iter()
                        .find(|n| n.innovation_number == edge.source_id);
                    if source_node.is_none() && edge.source_id != input_id {
                        errors.push(AnalysisError::RecurrentMissingSource {
                            node_id: target_id,
                            source_id: edge.source_id,
                        });
                        continue;
                    }

                    // 权重参数节点必须存在且为 Parameter 类型
                    let weight_node = enabled
                        .iter()
                        .find(|n| n.innovation_number == edge.weight_param_id);
                    match weight_node {
                        None => {
                            errors.push(AnalysisError::RecurrentInvalidWeight {
                                node_id: target_id,
                                weight_param_id: edge.weight_param_id,
                                reason: "权重参数节点不存在".into(),
                            });
                            continue;
                        }
                        Some(w) if !w.is_parameter() => {
                            errors.push(AnalysisError::RecurrentInvalidWeight {
                                node_id: target_id,
                                weight_param_id: edge.weight_param_id,
                                reason: format!(
                                    "期望 Parameter 类型，实际为 {:?}",
                                    w.node_type
                                ),
                            });
                            continue;
                        }
                        _ => {}
                    }
                    let weight_node = weight_node.unwrap();

                    // 形状兼容性：权重应为 [target_dim, source_dim]
                    let target_shape = output_shapes.get(&target_id);
                    let source_shape = if edge.source_id == input_id {
                        output_shapes.get(&input_id)
                    } else {
                        output_shapes.get(&edge.source_id)
                    };

                    if let (Some(t_shape), Some(s_shape)) = (target_shape, source_shape) {
                        // Flat 域：target_dim = t_shape[1], source_dim = s_shape[1]
                        // Sequence 域：target_dim = t_shape[2], source_dim = s_shape[2]
                        let (t_dim, s_dim) = if t_shape.len() == 3 && s_shape.len() == 3 {
                            (t_shape[2], s_shape[2])
                        } else if t_shape.len() == 2 && s_shape.len() == 2 {
                            (t_shape[1], s_shape[1])
                        } else {
                            errors.push(AnalysisError::RecurrentShapeMismatch {
                                node_id: target_id,
                                source_id: edge.source_id,
                                message: format!(
                                    "循环边仅支持 Flat/Sequence 域，目标形状={:?}, 源形状={:?}",
                                    t_shape, s_shape
                                ),
                            });
                            continue;
                        };

                        let expected_weight = vec![t_dim, s_dim];
                        if weight_node.output_shape != expected_weight {
                            errors.push(AnalysisError::RecurrentShapeMismatch {
                                node_id: target_id,
                                source_id: edge.source_id,
                                message: format!(
                                    "权重形状应为 {:?}，实际为 {:?}",
                                    expected_weight, weight_node.output_shape
                                ),
                            });
                        }
                    }
                }
            }
        }

        let is_valid = errors.is_empty();
        Self {
            topo_order,
            output_shapes,
            domains,
            param_count,
            enabled_node_count: enabled.len(),
            param_node_count,
            is_valid,
            errors,
            has_recurrent_edges,
        }
    }

    /// 构造失败状态的分析结果
    fn failed(
        errors: Vec<AnalysisError>,
        enabled_node_count: usize,
        param_count: usize,
        param_node_count: usize,
        output_shapes: HashMap<u64, Vec<usize>>,
        domains: HashMap<u64, ShapeDomain>,
    ) -> Self {
        Self {
            topo_order: Vec::new(),
            output_shapes,
            domains,
            param_count,
            enabled_node_count,
            param_node_count,
            is_valid: false,
            errors,
            has_recurrent_edges: false,
        }
    }

    /// 获取某节点的推导输出形状（便利方法）
    pub fn shape_of(&self, id: u64) -> Option<&Vec<usize>> {
        self.output_shapes.get(&id)
    }

    /// 获取某节点的输出域（便利方法）
    pub fn domain_of(&self, id: u64) -> Option<ShapeDomain> {
        self.domains.get(&id).copied()
    }

    /// 简洁的统计摘要字符串，用于日志输出
    ///
    /// 替代原有的层级字符串摘要（如 `Linear(4) → ReLU → [Linear(1)]`），
    /// 输出节点计数作为统一复杂度指标。
    pub fn summary(&self) -> String {
        format!(
            "nodes={} active={} params={}",
            self.enabled_node_count, self.topo_order.len(), self.param_node_count
        )
    }
}

// ==================== 拓扑排序 ====================

/// Kahn 算法拓扑排序，检测环（State 节点作为叶节点处理，不产生实际回边）
fn topological_sort(
    nodes: &[&NodeGene],
    input_id: u64,
) -> Result<Vec<u64>, AnalysisError> {
    let id_set: HashSet<u64> = nodes.iter().map(|n| n.innovation_number).collect();

    // 计算入度，建立 parent → children 邻接表
    let mut in_degree: HashMap<u64, usize> = HashMap::new();
    let mut children: HashMap<u64, Vec<u64>> = HashMap::new();

    in_degree.insert(input_id, 0);
    children.insert(input_id, Vec::new());

    for node in nodes {
        let id = node.innovation_number;
        // 叶节点（Parameter、State 等）没有有效的前向依赖父节点
        let valid_parents: Vec<u64> = if node.is_leaf() {
            Vec::new()
        } else {
            node.parents
                .iter()
                .filter(|&&pid| id_set.contains(&pid) || pid == input_id)
                .copied()
                .collect()
        };
        in_degree.insert(id, valid_parents.len());
        for pid in valid_parents {
            children.entry(pid).or_default().push(id);
        }
        children.entry(id).or_default();
    }

    // 从入度为 0 的节点开始，按 innovation_number 排序确保确定性
    // （HashMap::iter() 顺序依赖内部哈希 seed，在同一进程的不同调用间不一致）
    let mut zero_indegree: Vec<u64> = in_degree
        .iter()
        .filter(|&(_, &d)| d == 0)
        .map(|(&id, _)| id)
        .collect();
    zero_indegree.sort_unstable();
    let mut queue: VecDeque<u64> = zero_indegree.into_iter().collect();

    let mut order = Vec::new();

    while let Some(id) = queue.pop_front() {
        if id != input_id {
            order.push(id);
        }
        for &child in children.get(&id).map(|v| v.as_slice()).unwrap_or_default() {
            let deg = in_degree.entry(child).or_default();
            *deg = deg.saturating_sub(1);
            if *deg == 0 {
                queue.push_back(child);
            }
        }
    }

    if order.len() < nodes.len() {
        return Err(AnalysisError::CycleDetected);
    }

    Ok(order)
}
