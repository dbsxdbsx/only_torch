/*
 * @Author       : 老董
 * @Date         : 2025-12-27
 * @Description  : 图描述符（Graph Descriptor）
 *                 统一的中间表示（IR），用于序列化、可视化和调试输出
 */

use crate::nn::nodes::raw_node::Reduction;
use serde::{Deserialize, Serialize};

/// serde 反序列化默认值：dilation = (1, 1)，兼容无 dilation 字段的旧模型
fn default_dilation() -> (usize, usize) {
    (1, 1)
}

/// serde 反序列化默认值：output_padding = (0, 0)
fn default_output_padding() -> (usize, usize) {
    (0, 0)
}

/// serde 反序列化默认值：MaxPool2d.padding = (0, 0, 0, 0)
/// 兼容无 padding 字段的旧 .otm 模型
fn default_max_pool_padding() -> (usize, usize, usize, usize) {
    (0, 0, 0, 0)
}

/// 图的可序列化描述
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDescriptor {
    /// 格式版本（用于向后兼容）
    pub version: String,
    /// 图名称
    pub name: String,
    /// 所有节点描述
    pub nodes: Vec<NodeDescriptor>,
    /// 参数文件路径（相对于 JSON 文件），仅在保存完整模型时使用
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params_file: Option<String>,
}

/// 节点描述
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDescriptor {
    /// 节点 ID
    pub id: u64,
    /// 节点名称
    pub name: String,
    /// 节点类型
    pub node_type: NodeTypeDescriptor,
    /// 输出形状（固定形状，用于参数计算等）
    pub output_shape: Vec<usize>,
    /// 动态形状（支持动态维度，None 表示动态）
    ///
    /// 例如：`[None, Some(128)]` 表示 `[?, 128]`，batch 维度动态
    /// 如果为 None，表示该节点不支持动态维度（等同于固定形状）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_shape: Option<Vec<Option<usize>>>,
    /// 父节点 ID 列表（定义拓扑）
    pub parents: Vec<u64>,
    /// 参数数量（仅 Parameter 类型有意义）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param_count: Option<usize>,
}

/// 节点类型描述（包含类型特定参数）
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum NodeTypeDescriptor {
    BasicInput,  // 普通数据输入（Data 变体）
    TargetInput, // Loss 的目标值（Target 变体，可视化橙色）
    Parameter,
    State,    // 时间状态节点（RNN 隐藏状态等）
    Identity, // 恒等映射（用于 detach 等）
    Add,
    Divide,
    Subtract,
    MatMul,
    Multiply,
    Sigmoid,
    Softmax,
    Tanh,
    LeakyReLU {
        alpha: f32,
    },
    /// 自然对数（用于计算 log 概率、KL 散度等）
    Ln,
    /// LogSoftmax（数值稳定的 log(softmax)）
    LogSoftmax,
    /// Dropout 正则化（训练时丢弃，评估时直接通过）
    Dropout {
        p: f32,
    },
    Sign,
    Abs,
    SoftPlus,
    Step,
    /// 归约求和（axis=None 全局，axis=Some(i) 按轴）
    Sum {
        axis: Option<usize>,
    },
    /// 归约求均值（axis=None 全局，axis=Some(i) 按轴）
    Mean {
        axis: Option<usize>,
    },
    Reshape {
        target_shape: Vec<usize>,
    },
    Flatten {
        keep_first_dim: bool,
    },
    Conv2d {
        stride: (usize, usize),
        padding: (usize, usize),
        #[serde(default = "default_dilation")]
        dilation: (usize, usize),
    },
    /// 2D 转置卷积（反卷积），用于上采样
    ConvTranspose2d {
        stride: (usize, usize),
        padding: (usize, usize),
        #[serde(default = "default_output_padding")]
        output_padding: (usize, usize),
    },
    MaxPool2d {
        kernel_size: (usize, usize),
        stride: (usize, usize),
        /// 填充 (top, bottom, left, right)，对称 padding 用 (p, p, p, p)
        ///
        /// MaxPool 的 padding 用 `f32::NEG_INFINITY` 填充（避免污染 max 结果），
        /// 因此不能用通用 Pad 节点替代，必须在 MaxPool 内部支持。
        ///
        /// `#[serde(default)]` 兼容旧 .otm 模型（默认 (0,0,0,0) 等价无 padding）。
        #[serde(default = "default_max_pool_padding")]
        padding: (usize, usize, usize, usize),
        /// ONNX 风格 ceil_mode：true 用 ceil 计算输出尺寸，false 用 floor
        ///
        /// 默认 false（PyTorch / 旧 .otm 行为）；YOLOv5 等真实模型部分情况会显式置 true
        #[serde(default)]
        ceil_mode: bool,
    },
    /// 逐元素取最大值（PPO/TD3 等需要可微分 max）
    Maximum,
    /// 逐元素取最小值（PPO clipping、TD3 双 Q）
    Minimum,
    /// 沿轴取最大值（DQN 选最优动作 Q 值）
    Amax {
        axis: usize,
    },
    /// 沿轴取最小值（Double DQN 选保守 Q 值）
    Amin {
        axis: usize,
    },
    AvgPool2d {
        kernel_size: (usize, usize),
        stride: (usize, usize),
    },
    /// 2D 最近邻上采样（YOLO PAN/FPN 颈部用）
    ///
    /// 仅支持整数倍 nearest 模式：输出 [N, C, H*scale_h, W*scale_w]
    Upsample2d {
        scale_h: usize,
        scale_w: usize,
    },
    /// 张量索引选择（RNN 展开式设计用，固定索引）
    Select {
        axis: usize,
        index: usize,
    },
    /// 张量按索引收集（强化学习用，动态索引）
    Gather {
        dim: usize,
    },
    /// 张量堆叠（插入新维度）
    Stack {
        axis: usize,
    },
    /// 张量拼接（沿现有维度）
    Concat {
        axis: usize,
    },
    BCE {
        reduction: Reduction,
    },
    Huber {
        delta: f32,
        reduction: Reduction,
    },
    MAE {
        reduction: Reduction,
    },
    MSE {
        reduction: Reduction,
    },
    SoftmaxCrossEntropy,
    /// 动态零张量（RNN 初始隐藏状态）
    ZerosLike,

    // ──────────────────── 循环单元（复合模板节点）────────────────────
    /// RNN 循环单元（整个 RNN 计算作为单一模板节点，不展开为原子子节点）
    ///
    /// parents 顺序：[input, w_ih, w_hh, b_h]
    /// output_shape: return_sequences=false → [1, hidden_size]；true → [1, seq_len, hidden_size]
    CellRnn {
        input_size: usize,
        hidden_size: usize,
        return_sequences: bool,
        /// 构图占位序列长度（0 = 动态，rebuild 时取 max(1, seq_len)）
        seq_len: usize,
    },
    /// LSTM 循环单元
    ///
    /// parents 顺序：[input, w_ii, w_hi, b_i, w_if, w_hf, b_f, w_ig, w_hg, b_g, w_io, w_ho, b_o]
    CellLstm {
        input_size: usize,
        hidden_size: usize,
        return_sequences: bool,
        seq_len: usize,
    },
    /// GRU 循环单元
    ///
    /// parents 顺序：[input, w_ir, w_hr, b_r, w_iz, w_hz, b_z, w_in, w_hn, b_n]
    CellGru {
        input_size: usize,
        hidden_size: usize,
        return_sequences: bool,
        seq_len: usize,
    },

    // ──────────────────── 激活函数 ────────────────────
    /// 取反 y = -x
    Negate,
    /// ReLU: max(0, x)
    ReLU,
    /// GELU (tanh 近似)
    Gelu,
    /// Swish/SiLU: x * sigmoid(x)
    Swish,
    /// ELU: x if x>0, else alpha*(exp(x)-1)
    Elu {
        alpha: f32,
    },
    /// SELU（固定常数 λ/α）
    Selu,
    /// Mish: x * tanh(softplus(x))
    Mish,
    /// HardSwish（分段线性近似 Swish）
    HardSwish,
    /// HardSigmoid（分段线性近似 Sigmoid）
    HardSigmoid,
    /// ReLU6: min(max(0,x), 6)
    ReLU6,
    /// HardTanh: min(max(min_val, x), max_val)
    HardTanh {
        min_val: f32,
        max_val: f32,
    },

    // ──────────────────── 逐元素数学运算 ────────────────────
    /// 指数 y = e^x
    Exp,
    /// 平方根 y = √x
    Sqrt,
    /// 以 10 为底对数
    Log10,
    /// 以 2 为底对数
    Log2,
    /// 幂运算 y = x^exponent
    Pow {
        exponent: f32,
    },
    /// 平方 y = x²
    Square,
    /// 倒数 y = 1/x
    Reciprocal,

    // ──────────────────── 张量变换 ────────────────────
    /// 沿轴取连续范围 narrow(axis, start, length)
    Narrow {
        axis: usize,
        start: usize,
        length: usize,
    },
    /// 维度重排 permute(dims)
    Permute {
        dims: Vec<usize>,
    },
    /// 常量填充 pad(paddings, value)
    Pad {
        paddings: Vec<(usize, usize)>,
        pad_value: f32,
    },
    /// 沿各维度重复 repeat(repeats)
    Repeat {
        repeats: Vec<usize>,
    },
    /// 沿轴取前 k 大元素
    TopK {
        k: usize,
        axis: usize,
        sorted: bool,
    },
    /// 沿轴排序
    SortNode {
        axis: usize,
        descending: bool,
    },
    /// 值域裁剪 clip(min, max)
    Clip {
        min: f32,
        max: f32,
    },
    /// 条件选择 where(condition, x, y)
    WhereCond {
        /// 条件掩码（已归一化为 0/1 的 f32 数据，展平存储）
        condition_data: Vec<f32>,
        /// 条件掩码的形状
        condition_shape: Vec<usize>,
    },
    /// 梯度屏障（前向透传，反向阻断）
    Detach,

    // ──────────────────── 归一化运算 ────────────────────
    /// 批归一化（不含 gamma/beta）
    BatchNormOp {
        eps: f32,
        momentum: f32,
        num_features: usize,
    },
    /// 层归一化（不含 gamma/beta）
    LayerNormOp {
        normalized_dims: usize,
        eps: f32,
    },
    /// RMS 归一化（不含 gamma）
    RMSNormOp {
        normalized_dims: usize,
        eps: f32,
    },
}

impl GraphDescriptor {
    /// 创建新的图描述符
    pub fn new(name: &str) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            name: name.to_string(),
            nodes: Vec::new(),
            params_file: None,
        }
    }

    /// 添加节点描述
    pub fn add_node(&mut self, node: NodeDescriptor) {
        self.nodes.push(node);
    }

    /// 获取总参数量
    pub fn total_params(&self) -> usize {
        self.nodes.iter().filter_map(|n| n.param_count).sum()
    }

    /// 转换为 JSON 字符串
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// 从 JSON 字符串解析
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl NodeDescriptor {
    /// 创建新的节点描述
    ///
    /// # 参数
    /// - `id`: 节点 ID
    /// - `name`: 节点名称
    /// - `node_type`: 节点类型描述
    /// - `output_shape`: 固定输出形状
    /// - `dynamic_shape`: 动态形状（None 表示固定形状，Some([None, Some(128)]) 表示 [?, 128]）
    /// - `parents`: 父节点 ID 列表
    pub fn new(
        id: u64,
        name: &str,
        node_type: NodeTypeDescriptor,
        output_shape: Vec<usize>,
        dynamic_shape: Option<Vec<Option<usize>>>,
        parents: Vec<u64>,
    ) -> Self {
        let param_count = if matches!(node_type, NodeTypeDescriptor::Parameter) {
            Some(output_shape.iter().product())
        } else {
            None
        };

        Self {
            id,
            name: name.to_string(),
            node_type,
            output_shape,
            dynamic_shape,
            parents,
            param_count,
        }
    }

    /// 获取用于显示的形状字符串
    ///
    /// 如果有动态形状，使用 `?` 表示动态维度
    /// 例如：`[?, 128]` 表示 batch 维度动态
    pub fn display_shape(&self) -> String {
        if let Some(dyn_shape) = &self.dynamic_shape {
            let dims: Vec<String> = dyn_shape
                .iter()
                .map(|d| match d {
                    Some(n) => n.to_string(),
                    None => "?".to_string(),
                })
                .collect();
            format!("[{}]", dims.join(", "))
        } else {
            format!("{:?}", self.output_shape)
        }
    }

    /// 检查节点是否支持动态 batch
    pub fn has_dynamic_batch(&self) -> bool {
        self.dynamic_shape
            .as_ref()
            .is_some_and(|ds| ds.first().is_some_and(std::option::Option::is_none))
    }
}
