/*
 * @Author       : 老董
 * @Date         : 2025-12-27
 * @Description  : 图描述符（Graph Descriptor）
 *                 统一的中间表示（IR），用于序列化、可视化和调试输出
 */

use crate::nn::nodes::raw_node::Reduction;
use serde::{Deserialize, Serialize};

/// 图的可序列化描述
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDescriptor {
    /// 保存时的 only_torch 版本号(用于 `from_json` 失败时的 actionable 提示)
    ///
    /// MVP 阶段不维护 `.otm` 跨版本兼容,版本号只用于报错时告诉用户
    /// "当前 vX.Y.Z,该 .otm 由 vA.B.C 保存,请用对应版本重新加载或在新版本下重新 train/save"。
    pub version: String,
    /// 图名称
    pub name: String,
    /// 所有节点描述
    pub nodes: Vec<NodeDescriptor>,
    /// 参数文件路径（相对于 JSON 文件），仅在保存完整模型时使用
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params_file: Option<String>,
    /// 显式输出节点 ID 列表(可选)
    ///
    /// 由 ONNX 导入路径填充——`graph.output` 显式声明哪些 tensor 是模型输出,
    /// 这些信息在拓扑层"无后继 = 输出"的启发式中会丢失(典型例子:YOLOv5 的
    /// 单个 `output` 节点,因常量折叠/Split 重写产出多个无后继的中间节点,
    /// 如果走拓扑推断会把它们都误当作输出)。
    ///
    /// 演化、手写 Layer 等内部路径不填(为 None),`Graph::from_descriptor`
    /// 退回到 "无后继 = 输出" 的拓扑推断,语义不变。
    ///
    /// `Option<Vec<u64>>` 缺省序列化为 None,无需 `default` 标注(serde 对
    /// `Option` 缺失字段自动设为 None)。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explicit_output_ids: Option<Vec<u64>>,
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
    /// 节点的 ONNX 来源追溯（provenance）
    ///
    /// 由 `onnx_import` 在装配时填充，记录该节点是从原 ONNX 模型的哪些节点
    /// 合并/重写而来。例如：
    /// - 普通 1:1 映射：`["Conv_5"]`
    /// - Conv+bias 拆分：拆出的两个节点 origin 都是 `["Conv_42"]`
    /// - Split→Narrow 重写：每个 Narrow 都是 `["Split_100"]`
    /// - Reshape 折叠：`["Reshape_88", "<const:shape_input>"]`
    ///
    /// 演化、单元测试等非 ONNX 来源的节点保持空 `Vec`。
    ///
    /// `default + skip_serializing_if` 是配对惯用法:无 origin 时不写 JSON 节省体积,
    /// 反序列化回来时缺字段视为空 `Vec`。语义上等价于"所有节点都有这个字段,
    /// 只是为压缩输出而省略了空值",不是版本兼容兜底。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub origin_onnx_nodes: Vec<String>,
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
        dilation: (usize, usize),
    },
    /// 2D 转置卷积（反卷积），用于上采样
    ConvTranspose2d {
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
    },
    MaxPool2d {
        kernel_size: (usize, usize),
        stride: (usize, usize),
        /// 对称填充 (pad_h, pad_w),实际等价于四角各填 (pad_h, pad_h, pad_w, pad_w)
        ///
        /// 与 [`Self::Conv2d`] 保持一致的对称语义:ONNX 规范允许任意非对称四角,
        /// 但 PyTorch 默认导出和绝大多数真实模型(YOLOv5 SPPF 等)都是对称形式。
        /// ONNX importer 遇到非对称四角会报 actionable 错误,提示用户用
        /// `nn.ZeroPad2d` 显式拆开或用 onnxsim 预处理。
        ///
        /// 注意:底层 `raw_node::MaxPool2d` 仍用 4 维 padding 表示(算法实现需要,
        /// 用 `f32::NEG_INFINITY` 虚拟填充避免污染 max 结果)。IR 层只承诺对称语义,
        /// rebuild / Layer 层会展开为 (p_h, p_h, p_w, p_w) 传给 raw_node。
        padding: (usize, usize),
        /// ONNX 风格 ceil_mode:true 用 ceil 计算输出尺寸,false 用 floor
        ///
        /// PyTorch 默认 false;YOLOv5 等真实模型部分情况会显式置 true
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
            explicit_output_ids: None,
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
    ///
    /// MVP 阶段不维护 `.otm` 跨版本兼容:节点描述符 schema 变更时旧文件直接 fail-fast。
    /// 错误信息会包含本地版本号 + 文件中记录的版本号,提示用户用对应版本重新 train/save。
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        match serde_json::from_str::<Self>(json) {
            Ok(desc) => Ok(desc),
            Err(e) => {
                // 尝试只解析 version 字段,给用户一个 actionable 提示
                let local_version = env!("CARGO_PKG_VERSION");
                let saved_version: Option<String> = serde_json::from_str::<serde_json::Value>(json)
                    .ok()
                    .and_then(|v| {
                        v.get("version")
                            .and_then(|x| x.as_str().map(str::to_string))
                    });
                let hint = match saved_version {
                    Some(v) if v != local_version => format!(
                        " (本地 only_torch v{local_version},该 .otm 由 v{v} 保存;\
                         MVP 阶段不维护 .otm 跨版本兼容,请用 v{v} 重新加载或在新版本下重新 train/save)"
                    ),
                    _ => format!(
                        " (本地 only_torch v{local_version},节点描述符 schema 可能已变更;\
                         MVP 阶段不维护 .otm 跨版本兼容,请用对应版本重新 train/save)"
                    ),
                };
                // serde_json::Error 没有公开构造器,只能透传原错误,但用 to_string 时拼上提示
                // → 这里用 Display 拼装的方式重新构造一个 IO 风格的 serde_json Error
                Err(serde::de::Error::custom(format!("{e}{hint}")))
            }
        }
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
            origin_onnx_nodes: Vec::new(),
        }
    }

    /// 链式 builder：注入 ONNX 来源追溯
    ///
    /// 由 `onnx_import` 在装配时调用,演化和测试路径无需调用(默认空 Vec)。
    pub fn with_origin_onnx_nodes(mut self, names: Vec<String>) -> Self {
        self.origin_onnx_nodes = names;
        self
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
