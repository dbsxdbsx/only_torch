//! MyZero 模型定义（categorical value/reward + latent min-max 归一化）
//!
//! 三网络架构：
//! - Representation h: obs → latent，输出经 **min-max 归一化到 [0,1]**
//! - Dynamics g: (latent, action_onehot) → (next_latent, reward_logits, continuation_logit)
//!   next_latent 同样 min-max 归一化
//! - Prediction f: latent → (policy_logits, value_logits)
//!
//! value/reward 采用 **categorical 表示**（canonical MuZero）：head 输出 support 上的
//! logits，训练用 two-hot 目标 + 交叉熵，搜索期取 softmax 期望并 h⁻¹ 还原标量。

use super::consistency::negative_cosine_similarity;
use super::loss;
use super::value_encoding::{SupportConfig, scalar_to_two_hot, two_hot_to_scalar};
use crate::nn::{
    Graph, GraphError, IntoVar, Linear, Module, Var, VarActivationOps, VarLossOps, VarReduceOps,
    VarShapeOps,
};
use crate::rl::mcts::{ActionPayload, Dynamics, DynamicsOutput};
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

/// 全局 support 配置（value 与 reward 共用，对齐 canonical MuZero）。
pub const SUPPORT: SupportConfig = SupportConfig::new(SUPPORT_HALF);

/// continuation head 的解码偏置：随机初始化时默认接近「继续」，避免早期搜索过度截断。
const CONTINUATION_LOGIT_BIAS: f32 = 5.0;

/// 搜索期 hard terminal 阈值；低于该 continuation 才停止展开。
const TERMINAL_CONTINUATION_THRESHOLD: f32 = 0.05;

fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// latent min-max 归一化到 [0,1]（canonical MuZero，**逐样本**沿特征维）
///
/// `s_norm = (s - min(s)) / (max(s) - min(s) + eps)`，每行（样本）独立取 min/max。
/// batch 从 `latent` 的静态期望形状推断（`[B, dim]`），故同一份代码 batch=1（搜索/推理）
/// 与 batch>1（训练）通用；B=1 时 `reshape(&[1,1])` + `repeat(&[1,dim])` 与旧实现逐 bit 一致。
///
/// 梯度经 `amin`/`amax`（梯度只流向极值位置）+ `repeat`（梯度求和回传）正确反传。
fn min_max_normalize(latent: &Var, dim: usize) -> Result<Var, GraphError> {
    let batch = latent.value_expected_shape()[0]; // [B, dim] → B
    let min_v = latent.amin(1).reshape(&[batch, 1])?; // [B,1]，逐样本最小
    let max_v = latent.amax(1).reshape(&[batch, 1])?; // [B,1]，逐样本最大
    let range = (&max_v - &min_v) + 1e-5_f32; // [B,1]，加 eps 防除零
    let min_b = min_v.repeat(&[1, dim])?; // [B, dim]
    let range_b = range.repeat(&[1, dim])?; // [B, dim]
    Ok(&(latent - &min_b) / &range_b)
}

// ============================================================================
// Representation 网络 h: obs → latent（min-max 归一化）
// ============================================================================

pub struct RepresentationNet {
    fc1: Linear,
    fc2: Linear,
    latent_dim: usize,
}

impl RepresentationNet {
    pub fn new(graph: &Graph, obs_dim: usize, latent_dim: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("Repr");
        Ok(Self {
            fc1: Linear::new(&graph, obs_dim, 128, true, "fc1")?,
            fc2: Linear::new(&graph, 128, latent_dim, true, "fc2")?,
            latent_dim,
        })
    }

    /// obs → latent。隐藏层 relu，输出层线性后 **min-max 归一化**（不再 relu）。
    pub fn forward(&self, x: impl IntoVar) -> Result<Var, GraphError> {
        let h = self.fc1.forward(x).relu();
        let latent = self.fc2.forward(&h);
        min_max_normalize(&latent, self.latent_dim)
    }
}

impl Module for RepresentationNet {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
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
    latent_dim: usize,
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
            latent_dim,
        })
    }

    /// (latent, action_onehot) → (next_latent[min-max], reward_logits, continuation_logit)
    pub fn forward(
        &self,
        latent: &Var,
        action_onehot: &Var,
    ) -> Result<(Var, Var, Var), GraphError> {
        let input = Var::concat(&[latent, action_onehot], 1)?;
        let h = self.fc1.forward(&input).relu();
        let next_latent = min_max_normalize(&self.fc_latent.forward(&h), self.latent_dim)?;
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

    /// latent → (policy_logits, value_logits)
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

pub struct MyZeroModel {
    pub repr: RepresentationNet,
    pub dyn_net: DynamicsNet,
    pub pred: PredictionNet,
    pub projector: ProjectorNet,
    pub predictor: PredictorNet,
    pub recon: ReconstructionNet,
    pub lstm: ValuePrefixLstm, // value prefix LSTM
    pub graph: Graph,
    pub action_dim: usize,
    pub latent_dim: usize,
    // 搜索期持久化推理子图（不参与训练/序列化；训练走各网络自己的 forward 建图）
    root_infer: RootInfer,
    rec_infer: RecInfer,
}

impl MyZeroModel {
    pub fn new(
        graph: &Graph,
        obs_dim: usize,
        action_dim: usize,
        latent_dim: usize,
    ) -> Result<Self, GraphError> {
        let lstm_hidden = latent_dim; // LSTM hidden 维度 = latent_dim
        let repr = RepresentationNet::new(graph, obs_dim, latent_dim)?;
        let dyn_net = DynamicsNet::new(graph, latent_dim, action_dim)?;
        let pred = PredictionNet::new(graph, latent_dim, action_dim)?;
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
            latent_dim,
            root_infer,
            rec_infer,
        })
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

    /// 标量 value/reward → two-hot 目标张量 [1, support_size]
    fn two_hot_target(&self, x: f32) -> Tensor {
        Tensor::new(&scalar_to_two_hot(x, &SUPPORT), &[1, SUPPORT.size()])
    }

    /// 一批标量 value/reward → two-hot 目标张量 `[G, support_size]`（逐行 two-hot）。
    fn two_hot_batch(xs: &[f32]) -> Tensor {
        let size = SUPPORT.size();
        let mut flat = Vec::with_capacity(xs.len() * size);
        for &x in xs {
            flat.extend_from_slice(&scalar_to_two_hot(x, &SUPPORT));
        }
        Tensor::new(&flat, &[xs.len(), size])
    }

    /// value/reward logits Tensor → 标量（softmax 期望 + h⁻¹）
    fn decode_categorical(logits: &Tensor) -> f32 {
        let probs = logits.softmax(1);
        two_hot_to_scalar(probs.data_as_slice(), &SUPPORT)
    }

    // ========================================================================
    // 训练用：K 步 unroll（走计算图，可反传）
    // ========================================================================

    /// K 步 unroll 训练，返回总损失 Var
    ///
    /// value/reward 用 **categorical 交叉熵**（two-hot 目标），policy 用交叉熵，
    /// 与 canonical MuZero 一致。
    ///
    /// # 梯度缩放（canonical MuZero，附录 G）
    /// 两处 `scale_gradient`，均只改反传、不改前向损失值：
    /// 1. **hidden state ×0.5**：每个 dynamics step 后对 latent 施加，使越深的展开步对
    ///    repr/dynamics 的梯度贡献按 `0.5^k` 衰减，防 K 步反传梯度指数增长。
    /// 2. **recurrent loss ×(1/K)**：每个 recurrent step 的 loss 梯度按 `1/K` 缩放，
    ///    初始步权重 1.0、K 个 recurrent 步合计 1.0（梯度总权重恒 2.0，与 K 无关）。
    ///
    /// # absorbing state（终止处理，canonical MuZero）
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
        use_value_prefix: bool,
    ) -> Result<Var, GraphError> {
        let k = actions.len();

        let obs_tensor = Tensor::new(obs_t, &[1, obs_t.len()]);
        let mut latent = self.repr.forward(&obs_tensor)?;

        let (pred_policy, pred_value_logits) = self.pred.forward(&latent);
        let target_p0 = Tensor::new(&target_policies[0], &[1, self.action_dim]);
        let target_v0 = self.two_hot_target(target_values[0]);
        let mut total_loss = pred_policy.cross_entropy(&target_p0)?;
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
            let oh_tensor = Tensor::new(&oh, &[1, self.action_dim]);
            let oh_var = self.graph.input(&oh_tensor)?;

            let (next_latent, pred_reward_logits, pred_continuation_logit) =
                self.dyn_net.forward(&latent, &oh_var)?;
            let (pred_p, pred_v_logits) = self.pred.forward(&next_latent);

            let tp = Tensor::new(&target_policies[i + 1], &[1, self.action_dim]);
            let tv = self.two_hot_target(target_values[i + 1]);
            let tr = self.two_hot_target(target_rewards[i]);
            let tc = Tensor::new(&[target_continuations[i].clamp(0.0, 1.0)], &[1, 1]);

            let step_policy_loss = pred_p.cross_entropy(&tp)?;
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
                + &(&step_continuation_loss * loss::CONTINUATION_LOSS_COEF);

            // consistency：dynamics 预测的 next_latent 与 repr 编码的真实 next_obs 对齐
            if consistency_coef > 0.0
                && let Some(next_obs) = next_obs_list.and_then(|list| list.get(i))
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
                && let Some(next_obs) = next_obs_list.and_then(|list| list.get(i))
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn train_unroll_batch(
        &self,
        items: &[UnrollItem],
        consistency_coef: f32,
        reconstruction_coef: f32,
        use_value_prefix: bool,
    ) -> Result<Var, GraphError> {
        let g = items.len();
        debug_assert!(g > 0, "train_unroll_batch: 空组");
        let k = items[0].actions.len();
        let n_next = items[0].next_obs.len();
        let obs_dim = items[0].obs_t.len();

        // ---- 逐 slot 把 G 条样本堆叠成 [G, dim] 张量 ----
        let stack = |rows: &[&[f32]], dim: usize| -> Tensor {
            let mut flat = Vec::with_capacity(g * dim);
            for r in rows {
                flat.extend_from_slice(r);
            }
            Tensor::new(&flat, &[g, dim])
        };
        // obs_t（k=0 输入 + reconstruction k=0 目标）
        let obs_rows: Vec<&[f32]> = items.iter().map(|it| it.obs_t.as_slice()).collect();
        let obs_tensor = stack(&obs_rows, obs_dim);
        // 各步 policy 目标 [G, action_dim]（slot 0..=k）
        let policy_at = |slot: usize| -> Tensor {
            let rows: Vec<&[f32]> = items
                .iter()
                .map(|it| it.target_policies[slot].as_slice())
                .collect();
            stack(&rows, self.action_dim)
        };
        // 各步 value 标量 → two-hot [G, support]（slot 0..=k）
        let value_two_hot_at = |slot: usize| -> Tensor {
            let xs: Vec<f32> = items.iter().map(|it| it.target_values[slot]).collect();
            Self::two_hot_batch(&xs)
        };

        // ---- k=0：repr → pred（policy + value）+ reconstruction ----
        let mut latent = self.repr.forward(&obs_tensor)?;
        let (pred_policy, pred_value_logits) = self.pred.forward(&latent);
        let tp0 = policy_at(0);
        let tv0 = value_two_hot_at(0);
        let mut total_loss = pred_policy.cross_entropy(&tp0)?;
        total_loss =
            &total_loss + &(&pred_value_logits.cross_entropy(&tv0)? * loss::VALUE_LOSS_COEF);

        if reconstruction_coef > 0.0 {
            let recon0 = self.recon.forward(&latent);
            let recon_loss0 = recon0.mse_loss(&obs_tensor)?;
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
                .input(&Tensor::new(&oh_flat, &[g, self.action_dim]))?;

            let (next_latent, pred_reward_logits, pred_continuation_logit) =
                self.dyn_net.forward(&latent, &oh_var)?;
            let (pred_p, pred_v_logits) = self.pred.forward(&next_latent);

            let tp = policy_at(i + 1);
            let tv = value_two_hot_at(i + 1);
            let tr = Self::two_hot_batch(
                &items
                    .iter()
                    .map(|it| it.target_rewards[i])
                    .collect::<Vec<_>>(),
            );
            let tc_flat: Vec<f32> = items
                .iter()
                .map(|it| it.target_continuations[i].clamp(0.0, 1.0))
                .collect();
            let tc = Tensor::new(&tc_flat, &[g, 1]);

            let step_policy_loss = pred_p.cross_entropy(&tp)?;
            let step_value_loss = pred_v_logits.cross_entropy(&tv)?;

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
                + &(&step_continuation_loss * loss::CONTINUATION_LOSS_COEF);

            // consistency / reconstruction：仅在该步有真实 next_obs 时（组内 i<n_next 统一成立）
            if i < n_next {
                let next_rows: Vec<&[f32]> =
                    items.iter().map(|it| it.next_obs[i].as_slice()).collect();
                let next_obs_tensor = stack(&next_rows, obs_dim);

                if consistency_coef > 0.0 {
                    let repr_target = self.repr.forward(&next_obs_tensor)?;
                    let proj_target = self.projector.forward(&repr_target);
                    let proj_online = self.projector.forward(&next_latent);
                    let pred_online = self.predictor.forward(&proj_online);
                    let cons_loss = negative_cosine_similarity(&pred_online, &proj_target)?;
                    step_loss = &step_loss + &(&cons_loss * consistency_coef);
                }
                if reconstruction_coef > 0.0 {
                    let recon = self.recon.forward(&next_latent);
                    let recon_loss = recon.mse_loss(&next_obs_tensor)?;
                    step_loss = &step_loss + &(&recon_loss * reconstruction_coef);
                }
            }

            total_loss = &total_loss + &step_loss.scale_gradient(step_scale);
            latent = next_latent.scale_gradient(loss::DYNAMICS_GRADIENT_SCALE);
        }

        Ok(total_loss)
    }
}

/// batch-native 训练的单条样本（已展开好各步目标）。
///
/// 同一组（传入 [`MyZeroModel::train_unroll_batch`]）内所有 `UnrollItem` 须满足：
/// `actions.len()`（= actual_k）与 `next_obs.len()`（= consistency/recon 有效步数）一致，
/// 从而组内结构逐样本对齐、可直接堆叠成 batch 而无需 padding。
pub(crate) struct UnrollItem {
    pub obs_t: Vec<f32>,
    pub actions: Vec<usize>,            // len = actual_k
    pub target_policies: Vec<Vec<f32>>, // len = actual_k + 1
    pub target_values: Vec<f32>,        // len = actual_k + 1
    pub target_rewards: Vec<f32>,       // len = actual_k（value_prefix 时为前缀目标）
    pub target_continuations: Vec<f32>, // len = actual_k
    pub next_obs: Vec<Vec<f32>>,        // len = next_obs 有效步数（≤ actual_k）
}

// ============================================================================
// impl Dynamics —— 搜索期推理（detach，不走计算图）
// ============================================================================

impl Dynamics for &MyZeroModel {
    fn initial_state(&self, obs: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        (**self).initial_state_impl(obs)
    }

    fn recurrent(&self, state: &[f32], action: &ActionPayload) -> DynamicsOutput {
        (**self).recurrent_impl(state, action)
    }
}

impl MyZeroModel {
    fn initial_state_impl(&self, obs: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        // 复用持久化 root 子图：只写入 obs、单趟 forward、读缓存输出（不重建节点）。
        let r = &self.root_infer;
        r.obs_in
            .set_value(&Tensor::new(obs, &[1, obs.len()]))
            .expect("set obs 失败");
        self.graph.forward(&r.sink).expect("root forward 失败");

        let latent_tensor = r.latent.value().unwrap().unwrap();
        let policy_tensor = r.policy.value().unwrap().unwrap();
        let value_logits = r.value_logits.value().unwrap().unwrap();

        let policy_probs = policy_tensor.softmax(1);

        let latent_vec = latent_tensor.data_as_slice().to_vec();
        let policy_vec = policy_probs.data_as_slice().to_vec();
        let value = Self::decode_categorical(&value_logits);

        (latent_vec, policy_vec, value)
    }

    fn recurrent_impl(&self, state: &[f32], action: &ActionPayload) -> DynamicsOutput {
        let action_idx = match action {
            ActionPayload::Discrete(idx) => *idx,
            _ => 0,
        };

        let rc = &self.rec_infer;

        // setup：只写入 latent / action onehot（复用持久化输入节点，不新建）
        {
            crate::prof_scope!("model.rec.setup");
            rc.latent_in
                .set_value(&Tensor::new(state, &[1, self.latent_dim]))
                .expect("set latent 失败");
            let mut oh = vec![0.0; self.action_dim];
            if action_idx < self.action_dim {
                oh[action_idx] = 1.0;
            }
            rc.action_in
                .set_value(&Tensor::new(&oh, &[1, self.action_dim]))
                .expect("set action 失败");
        }

        // 单趟 forward：sink 覆盖全部输出，一次前向算完（复用持久化子图，不重建节点）。
        {
            crate::prof_scope!("model.rec.fwd");
            self.graph
                .forward(&rc.sink)
                .expect("recurrent forward 失败");
        }

        // read + decode：读缓存值 + categorical 解码 + to_vec 拷贝组装
        crate::prof_scope!("model.rec.decode");
        let next_latent_tensor = rc.next_latent.value().unwrap().unwrap();
        let reward_logits = rc.reward_logits.value().unwrap().unwrap();
        let continuation_logit = rc.continuation_logit.value().unwrap().unwrap();
        let policy_tensor = rc.policy.value().unwrap().unwrap();
        let value_logits = rc.value_logits.value().unwrap().unwrap();

        let policy_probs = policy_tensor.softmax(1);
        let reward = Self::decode_categorical(&reward_logits);
        let value = Self::decode_categorical(&value_logits);
        let continuation = sigmoid_scalar(
            continuation_logit
                .data_as_slice()
                .first()
                .copied()
                .unwrap_or(0.0)
                + CONTINUATION_LOGIT_BIAS,
        )
        .clamp(0.0, 1.0);

        DynamicsOutput {
            next_state: next_latent_tensor.data_as_slice().to_vec(),
            reward,
            prior: policy_probs.data_as_slice().to_vec(),
            value,
            terminal: continuation <= TERMINAL_CONTINUATION_THRESHOLD,
            continuation,
        }
    }
}
