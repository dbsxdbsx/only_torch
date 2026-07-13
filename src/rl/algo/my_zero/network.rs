//! `MyZero` 模型定义（categorical value/reward + latent min-max 归一化）
//!
//! 三网络架构：
//! - Representation h: obs → latent，输出经 **min-max 归一化到 [0,1]**
//! - Dynamics g: (latent, `action_onehot`) → (`next_latent`, `reward_logits`, `continuation_logit`)
//!   `next_latent` 同样 min-max 归一化
//! - Prediction f: latent → (`policy_logits`, `value_logits`)
//!
//! value/reward 采用 **categorical 表示**（canonical MuZero）：head 输出 support 上的
//! logits，训练用 two-hot 目标 + 交叉熵，搜索期取 softmax 期望并 h⁻¹ 还原标量。

use super::consistency::negative_cosine_similarity;
use super::loss;
#[cfg(test)]
use super::model_error::{ModelErrorComponents, TransitionDiagnostics, score_transition};
use super::obs_transform::{maybe_symlog, maybe_symlog_in_place};
pub use super::schema::ObsSpec;
use super::schema::{ActionSchema, ObservationSchema, PolicyLayout};
use super::value_encoding::{
    SupportConfig, scalar_to_hl_gauss, scalar_to_two_hot, two_hot_to_scalar,
};
use crate::nn::{
    Conv2d, Embedding, Graph, GraphError, IntoVar, Linear, Module, Var, VarActivationOps,
    VarLossOps, VarReduceOps, VarShapeOps,
};
use crate::rl::mcts::{ActionId, ActionPayload, Dynamics, DynamicsOutput};
use crate::tensor::Tensor;

// ============================================================================
// Value Prefix LSTM（用 LSTM 预测累计 reward 前缀和）
// ============================================================================

/// LSTM cell + prefix head，用于 value prefix 消融。
///
/// 每步接收 dynamics 产生的 `next_latent`，维护 `(h, c)` 隐状态，
/// 输出 categorical prefix logits（与 SUPPORT 对齐的 two-hot 目标）。
/// 搜索期 reward = `prefix_k − prefix_{k−1}`（增量）。
pub struct ValuePrefixLstm {
    gi: Linear,          // 输入门 i = σ(gi([x, h]))
    gf: Linear,          // 遗忘门 f = σ(gf([x, h]))
    gg: Linear,          // 候选细胞 g = tanh(gg([x, h]))
    go: Linear,          // 输出门 o = σ(go([x, h]))
    prefix_head: Linear, // h → categorical logits (SUPPORT.size())
    pub hidden: usize,
}

impl ValuePrefixLstm {
    pub fn new(graph: &Graph, input_size: usize, hidden_size: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("VpLstm");
        let gate_in = input_size + hidden_size;
        Ok(Self {
            gi: Linear::new(&graph, gate_in, hidden_size, true, "gi")?,
            gf: Linear::new(&graph, gate_in, hidden_size, true, "gf")?,
            gg: Linear::new(&graph, gate_in, hidden_size, true, "gg")?,
            go: Linear::new(&graph, gate_in, hidden_size, true, "go")?,
            prefix_head: Linear::new(&graph, hidden_size, SUPPORT.size(), true, "prefix_head")?,
            hidden: hidden_size,
        })
    }

    /// LSTM cell 一步前向：标准门控
    pub fn step(&self, x: &Var, h: &Var, c: &Var) -> Result<(Var, Var), GraphError> {
        let xh = Var::concat(&[x, h], 1)?;
        let i = self.gi.forward(&xh).sigmoid();
        let f = self.gf.forward(&xh).sigmoid();
        let g = self.gg.forward(&xh).tanh();
        let o = self.go.forward(&xh).sigmoid();
        let c_new = &(&f * c) + &(&i * &g);
        let h_new = &o * &c_new.tanh();
        Ok((h_new, c_new))
    }

    /// 从 LSTM hidden state 输出 prefix categorical logits
    pub fn prefix_logits(&self, h: &Var) -> Var {
        self.prefix_head.forward(h)
    }
}

impl Module for ValuePrefixLstm {
    fn parameters(&self) -> Vec<Var> {
        [
            self.gi.parameters(),
            self.gf.parameters(),
            self.gg.parameters(),
            self.go.parameters(),
            self.prefix_head.parameters(),
        ]
        .concat()
    }
}

/// Categorical value/reward 的 support 半宽。
///
/// support = `2*20+1 = 41` 个原子，覆盖变换域 `[-20,20]` → value 域约 `±420`，
/// 足以容纳 CartPole（gamma=0.997，truncation bootstrap 后 value 趋近 `1/(1-γ)≈333`，
/// `h(333)≈17.6`）的目标范围且留有余量。
pub const SUPPORT_HALF: usize = 20;

/// 全局 support 配置（value 与 reward 共用，对齐 canonical `MuZero`）。
pub const SUPPORT: SupportConfig = SupportConfig::new(SUPPORT_HALF);

/// continuation head 的解码偏置：随机初始化时默认接近「继续」，避免早期搜索过度截断。
const CONTINUATION_LOGIT_BIAS: f32 = 5.0;

/// 搜索期 hard terminal 阈值；低于该 continuation 才停止展开。
const TERMINAL_CONTINUATION_THRESHOLD: f32 = 0.05;

fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 借用节点值并按逻辑序取出为 `Vec<f32>`（搜索期读输出专用）
///
/// 相比 `var.value()`（clone 整块 Tensor）+ `data_as_slice().to_vec()`（再拷一次）
/// 的双拷贝，本函数在借用内直接 `to_vec()`，单次拷贝；`to_vec` 按逻辑行主序
/// 迭代，对非连续布局也正确（不会像 `data_as_slice` 那样 panic）。
pub(super) fn read_value_vec(var: &Var) -> Vec<f32> {
    var.node()
        .with_value(|v| v.expect("推理输出没有值，需先执行 forward").to_vec())
}

/// 单行数值稳定 softmax（搜索期解码专用，免 Tensor 中间分配）
///
/// 与 `Tensor::softmax(1)` 在 `[1, N]` 输入上**逐 bit 一致**：
/// - max：与 `amax` 相同的 `fold(NEG_INFINITY, f32::max)`；
/// - exp(x − max)：同 `mapv`；
/// - 求和：走 ndarray 1D `sum()`（与 `sum_axis` 对 lane 的归约同一路径，
///   包括其内部的分组累加顺序）；
/// - 逐元素除以 sum。
///
/// 替代旧的 `logits_tensor.softmax(1)` 链（amax → unsqueeze → sub → exp →
/// sum → div，约 6 次小 Tensor 分配），MCTS 每次 recurrent 推理调用 3 次。
pub(super) fn softmax_row(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum = ndarray::aview1(&out).sum();
    for v in &mut out {
        *v /= sum;
    }
    out
}

/// latent min-max 归一化到 [0,1]（canonical `MuZero`，**逐样本**沿特征维）
///
/// `s_norm = (s - min(s)) / (max(s) - min(s) + eps)`，每行（样本）独立取 min/max。
/// batch 从 `latent` 的静态期望形状推断（`[B, dim]`），故同一份代码 batch=1（搜索/推理）
/// 与 batch>1（训练）通用。
///
/// 梯度经 `amin`/`amax`（梯度只流向极值位置）+ 广播反向（`sum_to_shape` 归约回 `[B,1]`）正确反传。
fn min_max_normalize(latent: &Var) -> Result<Var, GraphError> {
    let batch = latent.value_expected_shape()[0]; // [B, dim] → B
    let min_v = latent.amin(1).reshape(&[batch, 1])?; // [B,1]，逐样本最小
    let max_v = latent.amax(1).reshape(&[batch, 1])?; // [B,1]，逐样本最大
    let range = (&max_v - &min_v) + 1e-5_f32; // [B,1]，加 eps 防除零
    // Subtract/Divide 原生支持 [B,dim] ⊙ [B,1] 广播，无需 repeat 物化
    // （旧版两个 repeat 节点纯冗余；反向由 sum_to_shape 归约回 [B,1]，
    // 前向逐 bit 等价，反向归约顺序与 Repeat 反向可能有 ulp 级差异——
    // 与 2026-07 数值流重定基线同批收口）
    Ok(&(latent - &min_v) / &range)
}

// ============================================================================
// Representation 网络 h: obs → latent（min-max 归一化）
// ============================================================================

pub struct RepresentationNet {
    fc1: Linear,
    fc2: Linear,
}

impl RepresentationNet {
    pub fn new(graph: &Graph, obs_dim: usize, latent_dim: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("Repr");
        Ok(Self {
            fc1: Linear::new(&graph, obs_dim, 128, true, "fc1")?,
            fc2: Linear::new(&graph, 128, latent_dim, true, "fc2")?,
        })
    }

    /// obs → latent。隐藏层 relu，输出层线性后 **min-max 归一化**（不再 relu）。
    pub fn forward(&self, x: impl IntoVar) -> Result<Var, GraphError> {
        let h = self.fc1.forward(x).relu();
        let latent = self.fc2.forward(&h);
        min_max_normalize(&latent)
    }
}

impl Module for RepresentationNet {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}

/// CNN representation（图像 obs；EfficientZero-lite 卷积栈，Phase 1 spike 同款）。
///
/// 输入为**展平**图像 `[B, c·side²]`（与全库 flat obs 通路兼容），前向内部
/// reshape 回 `[B, c, side, side]` → 若干 stride-2 3×3 conv（空间压到 ≤7）→
/// flatten → Linear → latent（min-max 归一化，与 MLP 版同口径）。
pub struct ConvRepresentationNet {
    convs: Vec<Conv2d>,
    fc_latent: Linear,
    channels: usize,
    height: usize,
    width: usize,
}

impl ConvRepresentationNet {
    pub fn new(
        graph: &Graph,
        channels: usize,
        side: usize,
        latent_dim: usize,
    ) -> Result<Self, GraphError> {
        Self::new_rect(graph, channels, side, side, latent_dim)
    }

    pub fn new_rect(
        graph: &Graph,
        channels: usize,
        height: usize,
        width: usize,
        latent_dim: usize,
    ) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("ConvRepr");
        // stride-2 conv 堆到两个空间轴均 ≤7；矩形输入保持宽高比，不裁方图。
        let mut convs = Vec::new();
        let mut cur_h = height;
        let mut cur_w = width;
        let mut cur_c = channels;
        let mut i = 0;
        while cur_h > 7 || cur_w > 7 {
            let out_c = if i == 0 { 32 } else { 64 };
            convs.push(Conv2d::new(
                &graph,
                cur_c,
                out_c,
                (3, 3),
                (2, 2),
                (1, 1),
                (1, 1),
                true,
                &format!("conv{}", i + 1),
            )?);
            cur_c = out_c;
            cur_h = cur_h.div_ceil(2);
            cur_w = cur_w.div_ceil(2);
            i += 1;
        }
        assert!(
            !convs.is_empty(),
            "ConvRepresentationNet: 图像过小（高宽均 ≤7），请用 Flat 编码器"
        );
        let flat_dim = cur_c * cur_h * cur_w;
        let fc_latent = Linear::new(&graph, flat_dim, latent_dim, true, "fc_latent")?;
        Ok(Self {
            convs,
            fc_latent,
            channels,
            height,
            width,
        })
    }

    /// `[B, c·side²]` 展平图像 → latent（min-max 归一化）。
    pub fn forward(&self, x: impl IntoVar) -> Result<Var, GraphError> {
        let x = x.into_var(&self.fc_latent.parameters()[0].get_graph())?;
        let batch = x.value_expected_shape()[0];
        let mut h = x.reshape(&[batch, self.channels, self.height, self.width])?;
        for conv in &self.convs {
            h = conv.forward(&h).relu();
        }
        let flat = h.flatten()?;
        let latent = self.fc_latent.forward(&flat);
        min_max_normalize(&latent)
    }
}

impl Module for ConvRepresentationNet {
    fn parameters(&self) -> Vec<Var> {
        let mut p: Vec<Var> = self.convs.iter().flat_map(Module::parameters).collect();
        p.extend(self.fc_latent.parameters());
        p
    }
}

/// 棋盘 representation 编码器：conv3×3 stride-1 ×2（c→32→64，padding 保形）
/// → flatten → fc → latent（min-max 归一化，与 MLP/Conv 编码器同契约）。
pub struct BoardConvRepresentationNet {
    c1: Conv2d,
    c2: Conv2d,
    fc_latent: Linear,
    channels: usize,
    side: usize,
}

impl BoardConvRepresentationNet {
    pub fn new(
        graph: &Graph,
        channels: usize,
        side: usize,
        latent_dim: usize,
    ) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("BoardRepr");
        let c1 = Conv2d::new(
            &graph,
            channels,
            32,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            true,
            "c1",
        )?;
        let c2 = Conv2d::new(&graph, 32, 64, (3, 3), (1, 1), (1, 1), (1, 1), true, "c2")?;
        let fc_latent = Linear::new(&graph, 64 * side * side, latent_dim, true, "fc_latent")?;
        Ok(Self {
            c1,
            c2,
            fc_latent,
            channels,
            side,
        })
    }

    /// `[B, c·side²]` 展平棋盘 → latent（min-max 归一化）。
    pub fn forward(&self, x: impl IntoVar) -> Result<Var, GraphError> {
        let x = x.into_var(&self.fc_latent.parameters()[0].get_graph())?;
        let batch = x.value_expected_shape()[0];
        let h = x.reshape(&[batch, self.channels, self.side, self.side])?;
        let h = self.c1.forward(&h).relu();
        let h = self.c2.forward(&h).relu();
        let flat = h.flatten()?;
        let latent = self.fc_latent.forward(&flat);
        min_max_normalize(&latent)
    }
}

impl Module for BoardConvRepresentationNet {
    fn parameters(&self) -> Vec<Var> {
        [
            self.c1.parameters(),
            self.c2.parameters(),
            self.fc_latent.parameters(),
        ]
        .concat()
    }
}

/// 已预处理矩形图像 + 稠密辅助向量的双分支 encoder。
pub struct ImageDenseRepresentationNet {
    image: ConvRepresentationNet,
    aux: RepresentationNet,
    fusion: Linear,
    image_dim: usize,
}

impl ImageDenseRepresentationNet {
    pub fn new(
        graph: &Graph,
        channels: usize,
        height: usize,
        width: usize,
        aux_dim: usize,
        latent_dim: usize,
    ) -> Result<Self, GraphError> {
        assert!(aux_dim > 0, "ImageDenseRepresentationNet: aux_dim 必须 > 0");
        let image = ConvRepresentationNet::new_rect(graph, channels, height, width, latent_dim)?;
        let aux = RepresentationNet::new(graph, aux_dim, latent_dim)?;
        let fusion = Linear::new(
            &graph.with_model_name("ImageDenseRepr"),
            latent_dim * 2,
            latent_dim,
            true,
            "fusion",
        )?;
        Ok(Self {
            image,
            aux,
            fusion,
            image_dim: channels * height * width,
        })
    }

    pub fn forward(&self, x: impl IntoVar) -> Result<Var, GraphError> {
        let graph = self.fusion.parameters()[0].get_graph();
        let x = x.into_var(&graph)?;
        let total_dim = x.value_expected_shape()[1];
        let image_x = x.narrow(1, 0, self.image_dim)?;
        let aux_x = x.narrow(1, self.image_dim, total_dim - self.image_dim)?;
        let image_latent = self.image.forward(&image_x)?;
        let aux_latent = self.aux.forward(&aux_x)?;
        let fused = Var::concat(&[&image_latent, &aux_latent], 1)?;
        min_max_normalize(&self.fusion.forward(&fused))
    }
}

impl Module for ImageDenseRepresentationNet {
    fn parameters(&self) -> Vec<Var> {
        [
            self.image.parameters(),
            self.aux.parameters(),
            self.fusion.parameters(),
        ]
        .concat()
    }
}

/// 固定长度 token encoder：Embedding → flatten → MLP → latent。
///
/// 这是 CPU 友好的最小 sequence encoder 纵切；后续可在同一 schema 下替换为
/// RNN/Transformer，而无需改 replay / MCTS 契约。
pub struct TokenRepresentationNet {
    embedding: Embedding,
    fc1: Linear,
    fc_latent: Linear,
    length: usize,
}

impl TokenRepresentationNet {
    pub fn new(
        graph: &Graph,
        length: usize,
        vocab_size: usize,
        embed_dim: usize,
        latent_dim: usize,
    ) -> Result<Self, GraphError> {
        assert!(length > 0, "TokenRepresentationNet: length 必须 > 0");
        let graph = graph.with_model_name("TokenRepr");
        Ok(Self {
            embedding: Embedding::new(&graph, vocab_size, embed_dim, "embedding")?,
            fc1: Linear::new(&graph, length * embed_dim, 128, true, "fc1")?,
            fc_latent: Linear::new(&graph, 128, latent_dim, true, "fc_latent")?,
            length,
        })
    }

    pub fn forward(&self, x: impl IntoVar) -> Result<Var, GraphError> {
        let graph = self.fc1.parameters()[0].get_graph();
        let x = x.into_var(&graph)?;
        let batch = x.value_expected_shape()[0];
        let tokens = x.narrow(1, 0, self.length)?;
        let mask = x
            .narrow(1, self.length, self.length)?
            .reshape(&[batch, self.length, 1])?
            .repeat(&[1, 1, self.embedding.embed_dim()])?;
        let embedded = self.embedding.forward(&tokens);
        let flat = (&embedded * &mask).flatten()?;
        let hidden = self.fc1.forward(&flat).relu();
        min_max_normalize(&self.fc_latent.forward(&hidden))
    }
}

impl Module for TokenRepresentationNet {
    fn parameters(&self) -> Vec<Var> {
        [
            self.embedding.parameters(),
            self.fc1.parameters(),
            self.fc_latent.parameters(),
        ]
        .concat()
    }
}

/// representation 编码器统一封装（recipe 按 [`ObsSpec`] 注入）。
pub enum ReprNet {
    Mlp(RepresentationNet),
    Conv(ConvRepresentationNet),
    // Box：Conv2d 按值内联使该变体显著大于其余两支（clippy large_enum_variant）
    Board(Box<BoardConvRepresentationNet>),
    ImageDense(Box<ImageDenseRepresentationNet>),
    Tokens(TokenRepresentationNet),
}

impl ReprNet {
    pub fn new(graph: &Graph, spec: ObsSpec, latent_dim: usize) -> Result<Self, GraphError> {
        match spec {
            ObsSpec::Flat(obs_dim) => Ok(Self::Mlp(RepresentationNet::new(
                graph, obs_dim, latent_dim,
            )?)),
            ObsSpec::Image { channels, side } => Ok(Self::Conv(ConvRepresentationNet::new(
                graph, channels, side, latent_dim,
            )?)),
            ObsSpec::ImageRect {
                channels,
                height,
                width,
            } => Ok(Self::Conv(ConvRepresentationNet::new_rect(
                graph, channels, height, width, latent_dim,
            )?)),
            ObsSpec::Board { channels, side } => Ok(Self::Board(Box::new(
                BoardConvRepresentationNet::new(graph, channels, side, latent_dim)?,
            ))),
            ObsSpec::ImageDense {
                channels,
                height,
                width,
                aux_dim,
            } => Ok(Self::ImageDense(Box::new(
                ImageDenseRepresentationNet::new(
                    graph, channels, height, width, aux_dim, latent_dim,
                )?,
            ))),
            ObsSpec::Tokens {
                length,
                vocab_size,
                embed_dim,
                pad_id: _,
            } => Ok(Self::Tokens(TokenRepresentationNet::new(
                graph, length, vocab_size, embed_dim, latent_dim,
            )?)),
        }
    }

    pub fn forward(&self, x: impl IntoVar) -> Result<Var, GraphError> {
        match self {
            Self::Mlp(net) => net.forward(x),
            Self::Conv(net) => net.forward(x),
            Self::Board(net) => net.forward(x),
            Self::ImageDense(net) => net.forward(x),
            Self::Tokens(net) => net.forward(x),
        }
    }
}

impl Module for ReprNet {
    fn parameters(&self) -> Vec<Var> {
        match self {
            Self::Mlp(net) => net.parameters(),
            Self::Conv(net) => net.parameters(),
            Self::Board(net) => net.parameters(),
            Self::ImageDense(net) => net.parameters(),
            Self::Tokens(net) => net.parameters(),
        }
    }
}

// ============================================================================
// Dynamics 网络 g: (latent, action_onehot) → (next_latent, reward_logits)
// ============================================================================

pub struct DynamicsNet {
    fc1: Linear,
    fc_latent: Linear,
    fc_reward: Linear,
    fc_continuation: Linear,
}

impl DynamicsNet {
    pub fn new(graph: &Graph, latent_dim: usize, action_dim: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("Dyn");
        let input_dim = latent_dim + action_dim;
        Ok(Self {
            fc1: Linear::new(&graph, input_dim, 128, true, "fc1")?,
            fc_latent: Linear::new(&graph, 128, latent_dim, true, "fc_latent")?,
            fc_reward: Linear::new(&graph, 128, SUPPORT.size(), true, "fc_reward")?,
            fc_continuation: Linear::new(&graph, 128, 1, true, "fc_continuation")?,
        })
    }

    /// (latent, `action_onehot`) → (`next_latent`[min-max], `reward_logits`, `continuation_logit`)
    pub fn forward(
        &self,
        latent: &Var,
        action_onehot: &Var,
    ) -> Result<(Var, Var, Var), GraphError> {
        let input = Var::concat(&[latent, action_onehot], 1)?;
        let h = self.fc1.forward(&input).relu();
        let next_latent = min_max_normalize(&self.fc_latent.forward(&h))?;
        let reward_logits = self.fc_reward.forward(&h);
        let continuation_logit = self.fc_continuation.forward(&h);
        Ok((next_latent, reward_logits, continuation_logit))
    }
}

impl Module for DynamicsNet {
    fn parameters(&self) -> Vec<Var> {
        [
            self.fc1.parameters(),
            self.fc_latent.parameters(),
            self.fc_reward.parameters(),
            self.fc_continuation.parameters(),
        ]
        .concat()
    }
}

// ============================================================================
// Prediction 网络 f: latent → (policy_logits, value_logits)
// ============================================================================

pub struct PredictionNet {
    fc1: Linear,
    fc_policy: Linear,
    fc_value: Linear,
}

impl PredictionNet {
    pub fn new(graph: &Graph, latent_dim: usize, action_dim: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("Pred");
        Ok(Self {
            fc1: Linear::new(&graph, latent_dim, 128, true, "fc1")?,
            fc_policy: Linear::new(&graph, 128, action_dim, true, "fc_policy")?,
            fc_value: Linear::new(&graph, 128, SUPPORT.size(), true, "fc_value")?,
        })
    }

    /// latent → (`policy_logits`, `value_logits`)
    pub fn forward(&self, latent: &Var) -> (Var, Var) {
        let h = self.fc1.forward(latent).relu();
        let policy = self.fc_policy.forward(&h);
        let value_logits = self.fc_value.forward(&h);
        (policy, value_logits)
    }
}

impl Module for PredictionNet {
    fn parameters(&self) -> Vec<Var> {
        [
            self.fc1.parameters(),
            self.fc_policy.parameters(),
            self.fc_value.parameters(),
        ]
        .concat()
    }
}

// ============================================================================
// Projector / Predictor 网络（consistency 专用，SimSiam 分支）
// ============================================================================

pub struct ProjectorNet {
    fc: Linear,
}

impl ProjectorNet {
    pub fn new(graph: &Graph, latent_dim: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("Proj");
        Ok(Self {
            fc: Linear::new(&graph, latent_dim, latent_dim, true, "fc")?,
        })
    }

    pub fn forward(&self, x: &Var) -> Var {
        self.fc.forward(x)
    }
}

impl Module for ProjectorNet {
    fn parameters(&self) -> Vec<Var> {
        self.fc.parameters()
    }
}

pub struct PredictorNet {
    fc: Linear,
}

impl PredictorNet {
    pub fn new(graph: &Graph, latent_dim: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("Predictor");
        Ok(Self {
            fc: Linear::new(&graph, latent_dim, latent_dim, true, "fc")?,
        })
    }

    pub fn forward(&self, x: &Var) -> Var {
        self.fc.forward(x)
    }
}

impl Module for PredictorNet {
    fn parameters(&self) -> Vec<Var> {
        self.fc.parameters()
    }
}

// ============================================================================
// Reconstruction 网络 h⁻¹: latent → obs（reconstruction 专用，不参与 MCTS）
// ============================================================================

pub struct ReconstructionNet {
    fc1: Linear,
    fc2: Linear,
}

impl ReconstructionNet {
    pub fn new(graph: &Graph, latent_dim: usize, obs_dim: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("Recon");
        Ok(Self {
            fc1: Linear::new(&graph, latent_dim, 128, true, "fc1")?,
            fc2: Linear::new(&graph, 128, obs_dim, true, "fc2")?,
        })
    }

    /// latent → 重建观测（线性输出，与 env obs 同尺度）
    pub fn forward(&self, latent: &Var) -> Var {
        let h = self.fc1.forward(latent).relu();
        self.fc2.forward(&h)
    }
}

impl Module for ReconstructionNet {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}

// ============================================================================
// MyZero 组合模型
// ============================================================================

/// 持久化 root 推理子图（h + f）：建一次、搜索期只 `set_value(obs)` + forward + 读缓存，
/// 避免每次 root 推理重建节点。`sink` 是全部输出的 concat，用于单趟 forward 一并计算。
struct RootInfer {
    obs_in: Var,
    latent: Var,
    policy: Var,
    value_logits: Var,
    sink: Var,
}

/// 持久化 recurrent 推理子图（g + f）：建一次、搜索期只 `set_value(latent, action)` +
/// forward + 读缓存。这是 MCTS 最热路径（sims × 每步），复用节点消除每次 ~25 个节点的重建。
struct RecInfer {
    latent_in: Var,
    action_in: Var,
    next_latent: Var,
    reward_logits: Var,
    continuation_logit: Var,
    policy: Var,
    value_logits: Var,
    sink: Var,
}

/// recurrent 持久化子图的一次数值输出。
///
/// 3A0 诊断需要 categorical reward distribution；MCTS 热路径继续从同一份结果
/// 构造 [`DynamicsOutput`]，避免诊断 API 与生产解码口径漂移。
struct RecurrentValues {
    next_state: Vec<f32>,
    #[cfg(test)]
    reward_probs: Vec<f32>,
    reward: f32,
    prior: Vec<f32>,
    value: f32,
    continuation: f32,
}

pub struct MyZeroModel {
    pub repr: ReprNet,
    pub dyn_net: DynamicsNet,
    pub pred: PredictionNet,
    pub projector: ProjectorNet,
    pub predictor: PredictorNet,
    pub recon: ReconstructionNet,
    pub lstm: ValuePrefixLstm, // value prefix LSTM
    pub graph: Graph,
    /// 联合动作数（MCTS `ActionId` / replay target 的稳定宽度）。
    pub action_dim: usize,
    /// policy head 实际输出宽度；factorized schema 下等于各 factor 宽度之和。
    pub policy_dim: usize,
    pub latent_dim: usize,
    pub action_schema: ActionSchema,
    policy_layout: PolicyLayout,
    /// value/reward 训练目标编码开关（false = two-hot，true = HL-Gauss）。
    /// 只影响训练目标构造，解码端（期望 → h⁻¹）与推理路径完全无关；
    /// 由 runner 按 `Components.hl_gauss` 注入（[`Self::with_value_encoding`]）。
    value_hl_gauss: bool,
    /// obs symlog 无量纲化开关（模型 obs 入口单点：repr 输入 + recon 目标同源变换；
    /// buffer / env I/O 恒存 raw obs）。由 runner 按 `Components.obs_symlog` 注入
    /// （[`Self::with_obs_symlog`]）。**开关必须与训练时一致**，否则权重语义失配。
    obs_symlog: bool,
    // 搜索期持久化推理子图（不参与训练/序列化；训练走各网络自己的 forward 建图）
    root_infer: RootInfer,
    rec_infer: RecInfer,
}

impl MyZeroModel {
    /// 低维向量 obs 构造（MLP 编码器；历史签名，等价 `ObsSpec::Flat`）。
    pub fn new(
        graph: &Graph,
        obs_dim: usize,
        action_dim: usize,
        latent_dim: usize,
    ) -> Result<Self, GraphError> {
        Self::new_with_spec(graph, ObsSpec::Flat(obs_dim), action_dim, latent_dim)
    }

    /// 按 [`ObsSpec`] 构造（图像 obs 走 CNN 编码器）。
    pub fn new_with_spec(
        graph: &Graph,
        spec: ObsSpec,
        action_dim: usize,
        latent_dim: usize,
    ) -> Result<Self, GraphError> {
        Self::new_with_schemas(graph, spec, ActionSchema::discrete(action_dim), latent_dim)
    }

    /// 按观测与动作 schema 构造通用 learned world model。
    pub fn new_with_schemas(
        graph: &Graph,
        spec: ObservationSchema,
        action_schema: ActionSchema,
        latent_dim: usize,
    ) -> Result<Self, GraphError> {
        let obs_dim = spec.dim();
        let action_dim = action_schema.action_count();
        let policy_layout = action_schema.policy_layout();
        let policy_dim = policy_layout.output_dim();
        let lstm_hidden = latent_dim; // LSTM hidden 维度 = latent_dim
        let repr = ReprNet::new(graph, spec, latent_dim)?;
        let dyn_net = DynamicsNet::new(graph, latent_dim, action_dim)?;
        let pred = PredictionNet::new(graph, latent_dim, policy_dim)?;
        let projector = ProjectorNet::new(graph, latent_dim)?;
        let predictor = PredictorNet::new(graph, latent_dim)?;
        let recon = ReconstructionNet::new(graph, latent_dim, obs_dim)?;
        let lstm = ValuePrefixLstm::new(graph, latent_dim, lstm_hidden)?;

        // 持久化 root 推理子图：obs → latent →(policy, value)。dummy 初值，搜索期 set_value 覆盖。
        let root_infer = {
            let obs_in = graph.input(&Tensor::zeros(&[1, obs_dim]))?;
            let latent = repr.forward(&obs_in)?;
            let (policy, value_logits) = pred.forward(&latent);
            let sink = Var::concat(&[&latent, &policy, &value_logits], 1)?;
            RootInfer {
                obs_in,
                latent,
                policy,
                value_logits,
                sink,
            }
        };

        // 持久化 recurrent 推理子图：(latent, action) → (next_latent, reward, continuation, policy, value)。
        let rec_infer = {
            let latent_in = graph.input(&Tensor::zeros(&[1, latent_dim]))?;
            let action_in = graph.input(&Tensor::zeros(&[1, action_dim]))?;
            let (next_latent, reward_logits, continuation_logit) =
                dyn_net.forward(&latent_in, &action_in)?;
            let (policy, value_logits) = pred.forward(&next_latent);
            let sink = Var::concat(
                &[
                    &next_latent,
                    &reward_logits,
                    &continuation_logit,
                    &policy,
                    &value_logits,
                ],
                1,
            )?;
            RecInfer {
                latent_in,
                action_in,
                next_latent,
                reward_logits,
                continuation_logit,
                policy,
                value_logits,
                sink,
            }
        };

        Ok(Self {
            repr,
            dyn_net,
            pred,
            projector,
            predictor,
            recon,
            lstm,
            graph: graph.clone(),
            action_dim,
            policy_dim,
            latent_dim,
            action_schema,
            policy_layout,
            value_hl_gauss: false,
            obs_symlog: false,
            root_infer,
            rec_infer,
        })
    }

    /// 设置 value/reward 训练目标编码（builder 式；默认 two-hot）。
    ///
    /// 与权重/图结构无关（纯目标构造分支），故可在构造后任意时点设置；
    /// 加载旧 checkpoint 推理不受影响（解码端两种编码相同）。
    pub(crate) const fn with_value_encoding(mut self, hl_gauss: bool) -> Self {
        self.value_hl_gauss = hl_gauss;
        self
    }

    /// 设置 obs symlog 无量纲化（builder 式；默认关）。
    ///
    /// 数据空间纯函数变换（进图前作用于 f32 切片），不改图结构与持久化子图；
    /// 训练 / 搜索 / 推理共用本开关，须与权重的训练口径一致。
    pub(crate) const fn with_obs_symlog(mut self, on: bool) -> Self {
        self.obs_symlog = on;
        self
    }

    pub fn parameters(&self) -> Vec<Var> {
        [
            self.repr.parameters(),
            self.dyn_net.parameters(),
            self.pred.parameters(),
            self.projector.parameters(),
            self.predictor.parameters(),
            self.recon.parameters(),
            self.lstm.parameters(),
        ]
        .concat()
    }

    /// 用于 `.otm` 拓扑序列化的代表输出 Var（dummy obs 前向，覆盖 h/g/f 子网）。
    pub(crate) fn otm_output_vars(&self, obs_dim: usize) -> Result<Vec<Var>, GraphError> {
        let obs = vec![0.0f32; obs_dim];
        let obs_tensor = Tensor::new(&obs, &[1, obs_dim]);
        let latent = self.repr.forward(&obs_tensor)?;
        let (policy, value) = self.pred.forward(&latent);
        let oh = self.action_to_onehot(0);
        let oh_tensor = Tensor::new(&oh, &[1, self.action_dim]);
        let oh_var = self.graph.input(&oh_tensor)?;
        let (next_latent, reward_logits, continuation_logit) =
            self.dyn_net.forward(&latent, &oh_var)?;
        Ok(vec![
            policy,
            value,
            reward_logits,
            continuation_logit,
            next_latent,
        ])
    }

    fn action_to_onehot(&self, action_idx: usize) -> Vec<f32> {
        let mut oh = vec![0.0; self.action_dim];
        if action_idx < self.action_dim {
            oh[action_idx] = 1.0;
        }
        oh
    }

    fn encode_policy_target(&self, joint: &[f32]) -> Vec<f32> {
        self.action_schema.encode_policy_target(joint)
    }

    fn policy_loss(&self, logits: &Var, target: Tensor) -> Result<Var, GraphError> {
        match &self.policy_layout {
            PolicyLayout::Categorical { .. } => logits.cross_entropy(target),
            PolicyLayout::Factorized { factors } => {
                let mut offset = 0;
                let mut losses = Vec::with_capacity(factors.len());
                for &width in factors {
                    let pred = logits.narrow(1, offset, width)?;
                    let expected = target.narrow(1, offset, width);
                    losses.push(pred.cross_entropy(expected)?);
                    offset += width;
                }
                let mut total = losses.remove(0);
                for loss in losses {
                    total = &total + &loss;
                }
                Ok(total)
            }
        }
    }

    /// `Σ target·log π`；factorized policy 对每个 factor 独立 log-softmax 后求和。
    fn weighted_policy_log_prob(&self, logits: &Var, target: Tensor) -> Result<Var, GraphError> {
        match &self.policy_layout {
            PolicyLayout::Categorical { .. } => Ok((&logits.log_softmax() * target).sum()),
            PolicyLayout::Factorized { factors } => {
                let mut offset = 0;
                let mut terms = Vec::with_capacity(factors.len());
                for &width in factors {
                    let pred = logits.narrow(1, offset, width)?;
                    let expected = target.narrow(1, offset, width);
                    terms.push((&pred.log_softmax() * expected).sum());
                    offset += width;
                }
                let mut total = terms.remove(0);
                for term in terms {
                    total = &total + &term;
                }
                Ok(total)
            }
        }
    }

    fn decode_policy_logits(&self, logits: &[f32]) -> Vec<f32> {
        match &self.policy_layout {
            PolicyLayout::Categorical { .. } => softmax_row(logits),
            PolicyLayout::Factorized { factors } => {
                let mut offset = 0;
                let mut factor_probs = Vec::with_capacity(logits.len());
                for &width in factors {
                    factor_probs.extend(softmax_row(&logits[offset..offset + width]));
                    offset += width;
                }
                self.action_schema.joint_priors(&factor_probs)
            }
        }
    }

    /// 标量 value/reward → two-hot 目标张量 [1, `support_size`]
    /// 标量 → categorical 软标签向量（按 [`Self::value_hl_gauss`] 选 two-hot / HL-Gauss）。
    fn encode_scalar(&self, x: f32) -> Vec<f32> {
        if self.value_hl_gauss {
            scalar_to_hl_gauss(x, &SUPPORT)
        } else {
            scalar_to_two_hot(x, &SUPPORT)
        }
    }

    fn two_hot_target(&self, x: f32) -> Tensor {
        Tensor::new(self.encode_scalar(x), &[1, SUPPORT.size()])
    }

    /// 一批标量 value/reward → categorical 目标张量 `[G, support_size]`（逐行编码）。
    fn two_hot_batch(&self, xs: &[f32]) -> Tensor {
        let size = SUPPORT.size();
        let mut flat = Vec::with_capacity(xs.len() * size);
        for &x in xs {
            flat.extend_from_slice(&self.encode_scalar(x));
        }
        Tensor::new(flat, &[xs.len(), size])
    }

    /// value/reward logits 切片 → 标量（softmax 期望 + h⁻¹，无 Tensor 中间分配）
    pub(super) fn decode_categorical_slice(logits: &[f32]) -> f32 {
        two_hot_to_scalar(&softmax_row(logits), &SUPPORT)
    }

    /// 在冻结 world-model revision 上计算一条真实 transition 的原始诊断输出。
    ///
    /// 该方法只执行数值前向，不建训练 target、不反传，也不消耗 RNG。终止步可把
    /// `next_obs` 设为 `None`，此时 re-encoded policy/value 分量明确缺失。
    #[cfg(test)]
    pub(crate) fn transition_diagnostics(
        &self,
        obs: &[f32],
        action_id: ActionId,
        next_obs: Option<&[f32]>,
    ) -> TransitionDiagnostics {
        let (root_latent, _, _) = self.initial_state_impl(obs);
        let imagined = self.recurrent_values_impl(&root_latent, action_id);
        let (reencoded_next_policy, reencoded_next_value) = match next_obs {
            Some(next_obs) => {
                let (_, policy, value) = self.initial_state_impl(next_obs);
                (Some(policy), Some(value))
            }
            None => (None, None),
        };

        TransitionDiagnostics {
            reward_probs: imagined.reward_probs,
            continuation: imagined.continuation,
            imagined_next_policy: imagined.prior,
            imagined_next_value: imagined.value,
            reencoded_next_policy,
            reencoded_next_value,
        }
    }

    /// 用真实 reward / continuation 与可选真实 next observation 给一条 transition 打分。
    #[cfg(test)]
    pub(crate) fn transition_error_components(
        &self,
        obs: &[f32],
        action_id: ActionId,
        observed_reward: f32,
        observed_continuation: f32,
        next_obs: Option<&[f32]>,
        next_legal_mask: Option<&[bool]>,
    ) -> ModelErrorComponents {
        let diagnostics = self.transition_diagnostics(obs, action_id, next_obs);
        let reward_target = self.encode_scalar(observed_reward);
        score_transition(
            &diagnostics,
            &reward_target,
            observed_continuation,
            next_legal_mask,
        )
    }

    // ========================================================================
    // 训练用：K 步 unroll（走计算图，可反传）
    // ========================================================================

    /// K 步 unroll 训练，返回总损失 Var
    ///
    /// value/reward 用 **categorical 交叉熵**（two-hot 目标），policy 用交叉熵，
    /// 与 canonical `MuZero` 一致。
    ///
    /// # 梯度缩放（canonical MuZero，附录 G）
    /// 两处 `scale_gradient`，均只改反传、不改前向损失值：
    /// 1. **hidden state ×0.5**：每个 dynamics step 后对 latent 施加，使越深的展开步对
    ///    repr/dynamics 的梯度贡献按 `0.5^k` 衰减，防 K 步反传梯度指数增长。
    /// 2. **recurrent loss ×(1/K)**：每个 recurrent step 的 loss 梯度按 `1/K` 缩放，
    ///    初始步权重 1.0、K 个 recurrent 步合计 1.0（梯度总权重恒 2.0，与 K 无关）。
    ///
    /// # absorbing state（终止处理，canonical `MuZero`）
    /// 终止后的 unroll 位置由调用方填入 **absorbing 目标**：
    /// `reward=0 / value=0 / policy=uniform / continuation=0`。模型据此学到「终局之后
    /// 回报恒 0 且不再传播未来 value」，掐断 no-terminal 价值膨胀。
    #[allow(clippy::too_many_arguments)]
    pub fn train_unroll(
        &self,
        obs_t: &[f32],
        actions: &[usize],
        target_policies: &[Vec<f32>],
        target_values: &[f32],
        target_rewards: &[f32], // value_prefix=true 时是前缀目标，false 时是单步 reward
        target_continuations: &[f32],
        next_obs_list: Option<&[Vec<f32>]>,
        consistency_coef: f32,
        reconstruction_coef: f32,
        continuation_coef: f32,
        use_value_prefix: bool,
    ) -> Result<Var, GraphError> {
        let k = actions.len();

        // obs 入口单点变换（symlog 开关；repr 输入与 recon 目标同源）
        let obs_tf = maybe_symlog(self.obs_symlog, obs_t);
        let obs_t: &[f32] = &obs_tf;

        let obs_tensor = Tensor::new(obs_t, &[1, obs_t.len()]);
        let mut latent = self.repr.forward(&obs_tensor)?;

        let (pred_policy, pred_value_logits) = self.pred.forward(&latent);
        let target_p0_encoded = self.encode_policy_target(&target_policies[0]);
        let target_p0 = Tensor::new(&target_p0_encoded, &[1, self.policy_dim]);
        let target_v0 = self.two_hot_target(target_values[0]);
        let mut total_loss = self.policy_loss(&pred_policy, target_p0)?;
        total_loss =
            &total_loss + &(&pred_value_logits.cross_entropy(&target_v0)? * loss::VALUE_LOSS_COEF);

        // reconstruction k=0：h(obs_t) 重建 obs_t
        if reconstruction_coef > 0.0 {
            let recon0 = self.recon.forward(&latent);
            let target_obs0 = Tensor::new(obs_t, &[1, obs_t.len()]);
            let recon_loss0 = recon0.mse_loss(&target_obs0)?;
            total_loss = &total_loss + &(&recon_loss0 * reconstruction_coef);
        }

        // value prefix：LSTM hidden state 初始化为全零
        let (mut vp_h, mut vp_c) = if use_value_prefix {
            let h0 = self.graph.zeros(&[1, self.lstm.hidden])?;
            let c0 = self.graph.zeros(&[1, self.lstm.hidden])?;
            (h0, c0)
        } else {
            // 占位，不会使用（避免 Option 复杂化主循环）
            let dummy = self.graph.zeros(&[1, 1])?;
            (dummy.clone(), dummy)
        };

        let step_scale = if k > 0 { 1.0 / k as f32 } else { 1.0 };

        for i in 0..k {
            let oh = self.action_to_onehot(actions[i]);
            let oh_tensor = Tensor::new(oh, &[1, self.action_dim]);
            let oh_var = self.graph.input(&oh_tensor)?;

            let (next_latent, pred_reward_logits, pred_continuation_logit) =
                self.dyn_net.forward(&latent, &oh_var)?;
            let (pred_p, pred_v_logits) = self.pred.forward(&next_latent);

            let tp_encoded = self.encode_policy_target(&target_policies[i + 1]);
            let tp = Tensor::new(&tp_encoded, &[1, self.policy_dim]);
            let tv = self.two_hot_target(target_values[i + 1]);
            let tr = self.two_hot_target(target_rewards[i]);
            let tc = Tensor::new(&[target_continuations[i].clamp(0.0, 1.0)], &[1, 1]);

            let step_policy_loss = self.policy_loss(&pred_p, tp)?;
            let step_value_loss = pred_v_logits.cross_entropy(&tv)?;

            // reward loss：value_prefix 开启时用 LSTM prefix logits，否则走原 DynamicsNet reward head
            let step_reward_loss = if use_value_prefix {
                let (h_new, c_new) = self.lstm.step(&next_latent, &vp_h, &vp_c)?;
                let prefix_logits = self.lstm.prefix_logits(&h_new);
                vp_h = h_new;
                vp_c = c_new;
                prefix_logits.cross_entropy(&tr)?
            } else {
                pred_reward_logits.cross_entropy(&tr)?
            };
            let pred_continuation = (&pred_continuation_logit + CONTINUATION_LOGIT_BIAS).sigmoid();
            let step_continuation_loss = pred_continuation.mse_loss(&tc)?;

            let mut step_loss = &step_policy_loss
                + &(&step_value_loss * loss::VALUE_LOSS_COEF)
                + &(&step_reward_loss * loss::REWARD_LOSS_COEF)
                + &(&step_continuation_loss * continuation_coef);

            // consistency / reconstruction 的 next_obs 同经模型入口单点变换
            let next_obs_tf = next_obs_list
                .and_then(|list| list.get(i))
                .map(|no| maybe_symlog(self.obs_symlog, no));

            // consistency：dynamics 预测的 next_latent 与 repr 编码的真实 next_obs 对齐
            if consistency_coef > 0.0
                && let Some(next_obs) = next_obs_tf.as_deref()
            {
                let obs_len = obs_t.len();
                let repr_target = self.repr.forward(Tensor::new(next_obs, &[1, obs_len]))?;
                let proj_target = self.projector.forward(&repr_target);
                let proj_online = self.projector.forward(&next_latent);
                let pred_online = self.predictor.forward(&proj_online);
                let cons_loss = negative_cosine_similarity(&pred_online, &proj_target)?;
                step_loss = &step_loss + &(&cons_loss * consistency_coef);
            }

            // reconstruction k>0：dynamics latent 重建 next_obs
            if reconstruction_coef > 0.0
                && let Some(next_obs) = next_obs_tf.as_deref()
            {
                let obs_len = obs_t.len();
                let recon = self.recon.forward(&next_latent);
                let target_obs = Tensor::new(next_obs, &[1, obs_len]);
                let recon_loss = recon.mse_loss(&target_obs)?;
                step_loss = &step_loss + &(&recon_loss * reconstruction_coef);
            }

            total_loss = &total_loss + &step_loss.scale_gradient(step_scale);

            latent = next_latent.scale_gradient(loss::DYNAMICS_GRADIENT_SCALE);
        }

        Ok(total_loss)
    }

    /// batch-native K 步 unroll 训练：一次 `[G, X]` 前向 + 一次 backward，覆盖 `G` 条 position。
    ///
    /// 与逐样本 [`train_unroll`] **数学等价**（实数域），仅浮点归约顺序不同：
    /// 组内所有样本共享同一 `actual_k`（`items[*].actions.len()`）与同一 `next_obs` 步数
    /// （`items[*].next_obs.len()`），故无需 padding/mask，结构逐样本一致。
    ///
    /// CE / MSE / consistency 均按 batch 取**均值**，故返回的组 loss = `(1/G) Σ_g L_g`；
    /// 调用方须再乘 `G / batch_size` 才与逐样本累积（各 `L_g × 1/batch_size`）的梯度一致。
    ///
    /// # 前置条件
    /// `items` 非空，且所有元素 `actions.len()`、`next_obs.len()`、`obs_t.len()` 一致。
    ///
    /// # ROSMO 行为正则（`bc_coef > 0` 时）
    /// 槽位 j（j < K）追加 `−(bc_coef/G)·Σ_g w_gj·log π(a_gj | s_gj)`（优势过滤 BC，
    /// arXiv:2210.05980 Eq.11）；`w` 来自 [`UnrollItem::bc_weights`]，全 0 槽位零开销跳过。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn train_unroll_batch(
        &self,
        items: &[UnrollItem<'_>],
        consistency_coef: f32,
        reconstruction_coef: f32,
        continuation_coef: f32,
        use_value_prefix: bool,
        bc_coef: f32,
    ) -> Result<Var, GraphError> {
        let g = items.len();
        debug_assert!(g > 0, "train_unroll_batch: 空组");
        let k = items[0].actions.len();
        let n_next = items[0].next_obs.len();
        let obs_dim = items[0].obs_t.dim();

        // ---- 逐 slot 把 G 条样本堆叠成 [G, dim] 张量 ----
        let stack = |rows: &[&[f32]], dim: usize| -> Tensor {
            let mut flat = Vec::with_capacity(g * dim);
            for r in rows {
                flat.extend_from_slice(r);
            }
            Tensor::new(flat, &[g, dim])
        };
        // obs 融合堆叠：从来源（buffer 借用帧）反量化/堆叠直接写入最终 flat（物化恰好一次），
        // 入口单点 symlog（repr 输入与 recon 目标同源；policy 等目标不经过）
        let stack_obs_sources = |srcs: &mut dyn Iterator<Item = &ObsSource>| -> Tensor {
            let mut flat = Vec::with_capacity(g * obs_dim);
            for s in srcs {
                s.append_into(&mut flat);
            }
            maybe_symlog_in_place(self.obs_symlog, &mut flat);
            Tensor::new(flat, &[g, obs_dim])
        };
        // obs_t（k=0 输入 + reconstruction k=0 目标）
        let obs_tensor = stack_obs_sources(&mut items.iter().map(|it| &it.obs_t));
        // 各步 policy 目标 [G, policy_dim]（joint target 按 schema 投影）
        let policy_at = |slot: usize| -> Tensor {
            let encoded: Vec<Vec<f32>> = items
                .iter()
                .map(|it| self.encode_policy_target(&it.target_policies[slot]))
                .collect();
            let rows: Vec<&[f32]> = encoded.iter().map(Vec::as_slice).collect();
            stack(&rows, self.policy_dim)
        };
        // 各步 value 标量 → categorical 目标 [G, support]（slot 0..=k）
        let value_two_hot_at = |slot: usize| -> Tensor {
            let xs: Vec<f32> = items.iter().map(|it| it.target_values[slot]).collect();
            self.two_hot_batch(&xs)
        };
        // ROSMO 行为正则：槽位 slot 的加权 one-hot [G, action_dim]（全 0 → None，跳过）
        let bc_target_at = |slot: usize| -> Option<Tensor> {
            if bc_coef <= 0.0 {
                return None;
            }
            let mut flat = Vec::with_capacity(g * self.policy_dim);
            let mut any = false;
            for it in items {
                let w = it.bc_weights.get(slot).copied().unwrap_or(0.0);
                let mut joint = vec![0.0f32; self.action_dim];
                if w > 0.0 {
                    let a = it.actions[slot];
                    if a < self.action_dim {
                        joint[a] = w;
                        any = true;
                    }
                }
                flat.extend(self.encode_policy_target(&joint));
            }
            any.then(|| Tensor::new(flat, &[g, self.policy_dim]))
        };
        // BC 项：−(bc_coef/G)·Σ w·log π(a)。
        let bc_term = |logits: &Var, wt: Tensor| -> Result<Var, GraphError> {
            Ok(self.weighted_policy_log_prob(logits, wt)? * (-bc_coef / g as f32))
        };

        // ---- k=0：repr → pred（policy + value）+ reconstruction ----
        // recon 开启时 obs 数据在图里需要两份（repr 输入节点 + recon 目标节点），clone 一次；
        // 关闭时整条 flat 直接 move 入图，零拷贝。
        let (repr_in, recon_target0) = if reconstruction_coef > 0.0 {
            (obs_tensor.clone(), Some(obs_tensor))
        } else {
            (obs_tensor, None)
        };
        let mut latent = self.repr.forward(repr_in)?;
        let (pred_policy, pred_value_logits) = self.pred.forward(&latent);
        let tp0 = policy_at(0);
        let tv0 = value_two_hot_at(0);
        let mut total_loss = self.policy_loss(&pred_policy, tp0)?;
        total_loss =
            &total_loss + &(&pred_value_logits.cross_entropy(tv0)? * loss::VALUE_LOSS_COEF);
        if let Some(wt) = bc_target_at(0) {
            total_loss = &total_loss + &bc_term(&pred_policy, wt)?;
        }

        if let Some(target0) = recon_target0 {
            let recon0 = self.recon.forward(&latent);
            let recon_loss0 = recon0.mse_loss(target0)?;
            total_loss = &total_loss + &(&recon_loss0 * reconstruction_coef);
        }

        // value prefix：LSTM hidden 初始化为全零（[G, hidden]）
        let (mut vp_h, mut vp_c) = if use_value_prefix {
            let h0 = self.graph.zeros(&[g, self.lstm.hidden])?;
            let c0 = self.graph.zeros(&[g, self.lstm.hidden])?;
            (h0, c0)
        } else {
            let dummy = self.graph.zeros(&[1, 1])?;
            (dummy.clone(), dummy)
        };

        let step_scale = if k > 0 { 1.0 / k as f32 } else { 1.0 };

        for i in 0..k {
            // action onehot [G, action_dim]
            let mut oh_flat = vec![0.0f32; g * self.action_dim];
            for (row, it) in items.iter().enumerate() {
                let a = it.actions[i];
                if a < self.action_dim {
                    oh_flat[row * self.action_dim + a] = 1.0;
                }
            }
            let oh_var = self
                .graph
                .input(&Tensor::new(oh_flat, &[g, self.action_dim]))?;

            let (next_latent, pred_reward_logits, pred_continuation_logit) =
                self.dyn_net.forward(&latent, &oh_var)?;
            let (pred_p, pred_v_logits) = self.pred.forward(&next_latent);

            let tp = policy_at(i + 1);
            let tv = value_two_hot_at(i + 1);
            let tr = self.two_hot_batch(
                &items
                    .iter()
                    .map(|it| it.target_rewards[i])
                    .collect::<Vec<_>>(),
            );
            let tc_flat: Vec<f32> = items
                .iter()
                .map(|it| it.target_continuations[i].clamp(0.0, 1.0))
                .collect();
            let tc = Tensor::new(tc_flat, &[g, 1]);

            let step_policy_loss = self.policy_loss(&pred_p, tp)?;
            let step_value_loss = pred_v_logits.cross_entropy(tv)?;

            let step_reward_loss = if use_value_prefix {
                let (h_new, c_new) = self.lstm.step(&next_latent, &vp_h, &vp_c)?;
                let prefix_logits = self.lstm.prefix_logits(&h_new);
                vp_h = h_new;
                vp_c = c_new;
                prefix_logits.cross_entropy(tr)?
            } else {
                pred_reward_logits.cross_entropy(tr)?
            };
            let pred_continuation = (&pred_continuation_logit + CONTINUATION_LOGIT_BIAS).sigmoid();
            let step_continuation_loss = pred_continuation.mse_loss(tc)?;

            let mut step_loss = &step_policy_loss
                + &(&step_value_loss * loss::VALUE_LOSS_COEF)
                + &(&step_reward_loss * loss::REWARD_LOSS_COEF)
                + &(&step_continuation_loss * continuation_coef);

            // ROSMO 行为正则：槽位 i+1 的执行动作（槽位 K 无动作，天然跳过）
            if i + 1 < k
                && let Some(wt) = bc_target_at(i + 1)
            {
                step_loss = &step_loss + &bc_term(&pred_p, wt)?;
            }

            // consistency / reconstruction：仅在该步有真实 next_obs 时（组内 i<n_next 统一成立）
            if i < n_next && (consistency_coef > 0.0 || reconstruction_coef > 0.0) {
                let next_obs_tensor =
                    stack_obs_sources(&mut items.iter().map(|it| &it.next_obs[i]));
                // 两个分支都开时数据在图里需要两份（consistency 输入 + recon 目标），clone 一次
                let (cons_in, recon_target) =
                    match (consistency_coef > 0.0, reconstruction_coef > 0.0) {
                        (true, true) => (Some(next_obs_tensor.clone()), Some(next_obs_tensor)),
                        (true, false) => (Some(next_obs_tensor), None),
                        (false, true) => (None, Some(next_obs_tensor)),
                        (false, false) => unreachable!(),
                    };

                if let Some(cons_obs) = cons_in {
                    let repr_target = self.repr.forward(cons_obs)?;
                    let proj_target = self.projector.forward(&repr_target);
                    let proj_online = self.projector.forward(&next_latent);
                    let pred_online = self.predictor.forward(&proj_online);
                    let cons_loss = negative_cosine_similarity(&pred_online, &proj_target)?;
                    step_loss = &step_loss + &(&cons_loss * consistency_coef);
                }
                if let Some(target) = recon_target {
                    let recon = self.recon.forward(&next_latent);
                    let recon_loss = recon.mse_loss(target)?;
                    step_loss = &step_loss + &(&recon_loss * reconstruction_coef);
                }
            }

            total_loss = &total_loss + &step_loss.scale_gradient(step_scale);
            latent = next_latent.scale_gradient(loss::DYNAMICS_GRADIENT_SCALE);
        }

        Ok(total_loss)
    }
}

/// 训练 batch 中单条 obs 的**来源描述**（延迟物化，融合组装）。
///
/// 组装 `[G, obs_dim]` batch 张量时由 [`ObsSource::append_into`] 直接把数据
/// （必要时反量化 / 帧堆叠）写入最终 flat buffer——把不可避免的物化压到恰好一次，
/// 消掉旧路径「每样本先物化堆叠 `Vec<f32>`、再复制进 batch flat」的中间拷贝。
pub(crate) enum ObsSource<'a> {
    /// 已物化的 f32 向量（测试构造 / reanalyze 等已有 owned 数据的路径）
    Owned(Vec<f32>),
    /// 单帧直通（向量 obs：borrow buffer，写入时反量化/复制一次）
    Single(&'a crate::rl::StoredObs),
    /// 图像模式：从 episode steps 就地堆叠位置 `t` 的最近 `stack` 帧（老 → 新）。
    ///
    /// episode 起点（`t < stack-1`）用 `saturating_sub` 前向填充首帧，与
    /// [`assemble_stacked_obs`](super::obs_pipeline::assemble_stacked_obs) /
    /// acting 期 `ImagePipe::reset` 语义逐 bit 一致。
    Stacked {
        steps: &'a [crate::rl::SelfPlayStep],
        t: usize,
        stack: usize,
    },
}

impl ObsSource<'_> {
    /// 物化后的 f32 元素个数
    pub fn dim(&self) -> usize {
        match self {
            Self::Owned(v) => v.len(),
            Self::Single(o) => o.len(),
            Self::Stacked { steps, t, stack } => stack * steps[*t].obs.len(),
        }
    }

    /// 把 obs 数据（反量化 f32）追加到 batch flat buffer（融合组装热路径）
    pub fn append_into(&self, out: &mut Vec<f32>) {
        match self {
            Self::Owned(v) => out.extend_from_slice(v),
            Self::Single(o) => o.append_f32_into(out),
            Self::Stacked { steps, t, stack } => {
                for i in (0..*stack).rev() {
                    let idx = t.saturating_sub(i);
                    steps[idx].obs.append_f32_into(out);
                }
            }
        }
    }
}

impl From<Vec<f32>> for ObsSource<'_> {
    fn from(v: Vec<f32>) -> Self {
        Self::Owned(v)
    }
}

/// batch-native 训练的单条样本（已展开好各步目标）。
///
/// 同一组（传入 [`MyZeroModel::train_unroll_batch`]）内所有 `UnrollItem` 须满足：
/// `actions.len()`（= `actual_k）与` `next_obs.len()`（= consistency/recon 有效步数）一致，
/// 从而组内结构逐样本对齐、可直接堆叠成 batch 而无需 padding。
///
/// obs 以 [`ObsSource`]（借用 + 延迟物化）持有，batch 组装时一次性写入最终 flat buffer。
pub(crate) struct UnrollItem<'a> {
    pub obs_t: ObsSource<'a>,
    pub actions: Vec<usize>,            // len = actual_k
    pub target_policies: Vec<Vec<f32>>, // len = actual_k + 1
    pub target_values: Vec<f32>,        // len = actual_k + 1
    pub target_rewards: Vec<f32>,       // len = actual_k（value_prefix 时为前缀目标）
    pub target_continuations: Vec<f32>, // len = actual_k
    pub next_obs: Vec<ObsSource<'a>>,   // len = next_obs 有效步数（≤ actual_k）
    /// ROSMO 优势过滤行为正则权重（len = actual_k，槽位 j 对应执行动作 `actions[j]`；
    /// 空 = 非 ROSMO 路径，BC 不参与）。
    pub bc_weights: Vec<f32>,
}

// ============================================================================
// impl Dynamics —— 搜索期推理（detach，不走计算图）
// ============================================================================

impl Dynamics for &MyZeroModel {
    fn initial_state(&self, obs: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        (**self).initial_state_impl(obs)
    }

    fn recurrent(&self, state: &[f32], action: &ActionPayload) -> DynamicsOutput {
        let ActionPayload::Discrete(action_idx) = action else {
            panic!(
                "MyZeroModel::recurrent: 结构化 payload 必须通过 WorldModel / recurrent_with_id 传入稳定 ActionId"
            );
        };
        (**self).recurrent_impl(state, ActionId(*action_idx), action)
    }

    fn recurrent_with_id(
        &self,
        state: &[f32],
        action_id: ActionId,
        action: &ActionPayload,
    ) -> DynamicsOutput {
        (**self).recurrent_impl(state, action_id, action)
    }
}

impl MyZeroModel {
    fn initial_state_impl(&self, obs: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        // 复用持久化 root 子图：只写入 obs、单趟 forward、读缓存输出（不重建节点）。
        // obs 入口单点变换（与训练路径同一开关，保证搜索/推理量纲一致）。
        let obs_tf = maybe_symlog(self.obs_symlog, obs);
        let obs_len = obs_tf.len();
        let r = &self.root_infer;
        r.obs_in
            .set_value(&Tensor::new(obs_tf.into_owned(), &[1, obs_len]))
            .expect("set obs 失败");
        self.graph.forward(&r.sink).expect("root forward 失败");

        // 读取输出：借用直取（单次拷贝）+ 无 Tensor 解码（softmax/categorical 走切片路径）
        let latent_vec = read_value_vec(&r.latent);
        let policy_vec = self.decode_policy_logits(&read_value_vec(&r.policy));
        let value = Self::decode_categorical_slice(&read_value_vec(&r.value_logits));

        (latent_vec, policy_vec, value)
    }

    fn recurrent_impl(
        &self,
        state: &[f32],
        action_id: ActionId,
        _action: &ActionPayload,
    ) -> DynamicsOutput {
        let values = self.recurrent_values_impl(state, action_id);
        let terminal = values.continuation <= TERMINAL_CONTINUATION_THRESHOLD;
        DynamicsOutput {
            next_state: values.next_state,
            reward: values.reward,
            prior: values.prior,
            value: values.value,
            terminal,
            continuation: values.continuation,
        }
    }

    fn recurrent_values_impl(&self, state: &[f32], action_id: ActionId) -> RecurrentValues {
        let action_idx = action_id.index();
        assert!(
            action_idx < self.action_dim,
            "MyZeroModel::recurrent: ActionId {action_idx} 越界 {}",
            self.action_dim
        );
        let rc = &self.rec_infer;

        // setup：只写入 latent / action onehot（复用持久化输入节点，不新建；
        // Tensor 存储 Arc/CoW 化后 set_value 内部 clone 为 O(1) 浅拷贝）
        {
            crate::prof_scope!("model.rec.setup");
            rc.latent_in
                .set_value(&Tensor::new(state.to_vec(), &[1, self.latent_dim]))
                .expect("set latent 失败");
            let mut oh = vec![0.0; self.action_dim];
            if action_idx < self.action_dim {
                oh[action_idx] = 1.0;
            }
            rc.action_in
                .set_value(&Tensor::new(oh, &[1, self.action_dim]))
                .expect("set action 失败");
        }

        // 单趟 forward：sink 覆盖全部输出，一次前向算完（复用持久化子图，不重建节点）。
        {
            crate::prof_scope!("model.rec.fwd");
            self.graph
                .forward(&rc.sink)
                .expect("recurrent forward 失败");
        }

        // read + decode：借用直取（单次拷贝）+ 无 Tensor 解码（softmax/categorical 走切片路径）
        crate::prof_scope!("model.rec.decode");
        let next_state = read_value_vec(&rc.next_latent);
        let reward_logits = read_value_vec(&rc.reward_logits);
        let reward = Self::decode_categorical_slice(&reward_logits);
        #[cfg(test)]
        let reward_probs = softmax_row(&reward_logits);
        let value = Self::decode_categorical_slice(&read_value_vec(&rc.value_logits));
        let prior = self.decode_policy_logits(&read_value_vec(&rc.policy));
        let continuation_logit = rc.continuation_logit.node().with_value(|v| {
            v.expect("推理输出没有值，需先执行 forward")
                .to_vec()
                .first()
                .copied()
                .unwrap_or(0.0)
        });
        let continuation =
            sigmoid_scalar(continuation_logit + CONTINUATION_LOGIT_BIAS).clamp(0.0, 1.0);

        RecurrentValues {
            next_state,
            #[cfg(test)]
            reward_probs,
            reward,
            prior,
            value,
            continuation,
        }
    }
}
