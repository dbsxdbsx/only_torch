//! EfficientZero CartPole 模型（**base = canonical MuZero**，EZ 增量逐阶段叠加）。
//!
//! Phase 1 base 与 MuZero 完全体同构（categorical value/reward + latent min-max 归一化 +
//! absorbing state + canonical 梯度缩放），作为消融基线。后续增量在此扩展：
//! - `+consistency`：repr(next_obs) ↔ dynamics next_latent 的 SimSiam stop-grad（加 projector 头）
//! - `+value prefix`：reward head 换 LSTM value-prefix（hidden 经 State 穿过 MCTS 树）
//! - `+target network`：实例化两份本模型（online / target）
//!
//! 三网络：
//! - Representation h: obs(4) → latent，min-max 归一化到 [0,1]
//! - Dynamics g: (latent, action_onehot) → (next_latent[min-max], reward_logits)
//! - Prediction f: latent → (policy_logits, value_logits)

use only_torch::nn::{
    Graph, GraphError, IntoVar, Linear, Module, Var, VarActivationOps, VarLossOps, VarReduceOps,
    VarShapeOps,
};
use only_torch::rl::algo::efficientzero::{negative_cosine_similarity, reward_prefix_targets};
use only_torch::rl::algo::muzero::{SupportConfig, loss, scalar_to_two_hot, two_hot_to_scalar};
use only_torch::rl::mcts::{ActionPayload, Dynamics, DynamicsOutput};
use only_torch::tensor::Tensor;

/// Categorical value/reward 的 support 半宽（41 原子，覆盖 CartPole gamma=0.997 的 value 域）。
pub const SUPPORT_HALF: usize = 20;
/// 全局 support 配置（value 与 reward 共用）。
pub const SUPPORT: SupportConfig = SupportConfig::new(SUPPORT_HALF);

/// latent min-max 归一化到 [0,1]（canonical MuZero，逐样本沿特征维，batch=1）。
fn min_max_normalize(latent: &Var, dim: usize) -> Result<Var, GraphError> {
    let min_v = latent.amin(1).reshape(&[1, 1])?;
    let max_v = latent.amax(1).reshape(&[1, 1])?;
    let range = (&max_v - &min_v) + 1e-5_f32;
    let min_b = min_v.repeat(&[1, dim])?;
    let range_b = range.repeat(&[1, dim])?;
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
            latent_dim,
        })
    }

    pub fn forward(&self, latent: &Var, action_onehot: &Var) -> Result<(Var, Var), GraphError> {
        let input = Var::concat(&[latent, action_onehot], 1)?;
        let h = self.fc1.forward(&input).relu();
        let next_latent = min_max_normalize(&self.fc_latent.forward(&h), self.latent_dim)?;
        let reward_logits = self.fc_reward.forward(&h);
        Ok((next_latent, reward_logits))
    }
}

impl Module for DynamicsNet {
    fn parameters(&self) -> Vec<Var> {
        [
            self.fc1.parameters(),
            self.fc_latent.parameters(),
            self.fc_reward.parameters(),
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
// SimSiam projector / predictor（+consistency 用；base 不参与损失）
// ============================================================================

/// Projector：latent → proj 空间（online 与 target 共享）。
pub struct ProjectorNet {
    fc1: Linear,
    fc2: Linear,
}

impl ProjectorNet {
    pub fn new(graph: &Graph, latent_dim: usize, proj_dim: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("Proj");
        Ok(Self {
            fc1: Linear::new(&graph, latent_dim, proj_dim, true, "fc1")?,
            fc2: Linear::new(&graph, proj_dim, proj_dim, true, "fc2")?,
        })
    }

    pub fn forward(&self, latent: &Var) -> Var {
        let h = self.fc1.forward(latent).relu();
        self.fc2.forward(&h)
    }
}

impl Module for ProjectorNet {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}

/// Predictor：proj → proj（仅 online 分支，SimSiam 防坍缩）。
pub struct PredictorNet {
    fc1: Linear,
    fc2: Linear,
}

impl PredictorNet {
    pub fn new(graph: &Graph, proj_dim: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("PredHead");
        let hidden = (proj_dim / 2).max(1);
        Ok(Self {
            fc1: Linear::new(&graph, proj_dim, hidden, true, "fc1")?,
            fc2: Linear::new(&graph, hidden, proj_dim, true, "fc2")?,
        })
    }

    pub fn forward(&self, proj: &Var) -> Var {
        let h = self.fc1.forward(proj).relu();
        self.fc2.forward(&h)
    }
}

impl Module for PredictorNet {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}

// ============================================================================
// Value-prefix LSTM cell（+value prefix 忠实版）
// ============================================================================

/// 手搓 LSTM cell（融合门：每门一个 `Linear(concat(x,h))`，规避 Var 无 range-slice）。
///
/// EZ value prefix：用 LSTM 在 unroll / 搜索路径上累计预测 reward 前缀。hidden (h,c) 经
/// `MctsModel::State`（序列化进 latent 向量尾部）穿过搜索树；搜索期每条边 reward 取
/// value prefix 增量（见 model 的 recurrent）。`prefix_head` 输出 categorical（与 reward 同 support）。
pub struct ValuePrefixLstm {
    gi: Linear,
    gf: Linear,
    gg: Linear,
    go: Linear,
    prefix_head: Linear,
    pub hidden: usize,
}

impl ValuePrefixLstm {
    pub fn new(graph: &Graph, latent_dim: usize, hidden: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("VPrefix");
        let inp = latent_dim + hidden;
        Ok(Self {
            gi: Linear::new(&graph, inp, hidden, true, "gi")?,
            gf: Linear::new(&graph, inp, hidden, true, "gf")?,
            gg: Linear::new(&graph, inp, hidden, true, "gg")?,
            go: Linear::new(&graph, inp, hidden, true, "go")?,
            prefix_head: Linear::new(&graph, hidden, SUPPORT.size(), true, "prefix")?,
            hidden,
        })
    }

    /// 一步 LSTM cell：`(x, h, c) → (h_new, c_new)`。
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

    /// 从 hidden 预测 value prefix 的 categorical logits。
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

// ============================================================================
// EZ 组合模型（base = MuZero 三网络；+consistency 增 projector/predictor）
// ============================================================================

pub struct EzModel {
    pub repr: RepresentationNet,
    pub dyn_net: DynamicsNet,
    pub pred: PredictionNet,
    pub proj: ProjectorNet,
    pub pred_head: PredictorNet,
    pub lstm: ValuePrefixLstm,
    pub graph: Graph,
    pub action_dim: usize,
    pub latent_dim: usize,
    /// 是否启用 value prefix（忠实版）：搜索期 reward 取 LSTM prefix 增量、state 携带 (h,c,prefix)。
    pub use_value_prefix: bool,
}

impl EzModel {
    pub fn new(
        graph: &Graph,
        obs_dim: usize,
        action_dim: usize,
        latent_dim: usize,
        use_value_prefix: bool,
    ) -> Result<Self, GraphError> {
        Ok(Self {
            repr: RepresentationNet::new(graph, obs_dim, latent_dim)?,
            dyn_net: DynamicsNet::new(graph, latent_dim, action_dim)?,
            pred: PredictionNet::new(graph, latent_dim, action_dim)?,
            proj: ProjectorNet::new(graph, latent_dim, latent_dim)?,
            pred_head: PredictorNet::new(graph, latent_dim)?,
            lstm: ValuePrefixLstm::new(graph, latent_dim, 32)?,
            graph: graph.clone(),
            action_dim,
            latent_dim,
            use_value_prefix,
        })
    }

    pub fn parameters(&self) -> Vec<Var> {
        [
            self.repr.parameters(),
            self.dyn_net.parameters(),
            self.pred.parameters(),
            self.proj.parameters(),
            self.pred_head.parameters(),
            self.lstm.parameters(),
        ]
        .concat()
    }

    fn action_to_onehot(&self, action_idx: usize) -> Vec<f32> {
        let mut oh = vec![0.0; self.action_dim];
        if action_idx < self.action_dim {
            oh[action_idx] = 1.0;
        }
        oh
    }

    fn two_hot_target(&self, x: f32) -> Tensor {
        Tensor::new(&scalar_to_two_hot(x, &SUPPORT), &[1, SUPPORT.size()])
    }

    fn decode_categorical(logits: &Tensor) -> f32 {
        let probs = logits.softmax(1);
        two_hot_to_scalar(probs.data_as_slice(), &SUPPORT)
    }

    /// K 步 unroll 训练（categorical 交叉熵 + canonical 梯度缩放 + absorbing 由调用方填目标）。
    ///
    /// `next_obs[i]`：第 i 个 action 之后的真实 next obs（absorbing / 越界为 `None`），供
    /// `+consistency` 的 SimSiam target 分支；`consistency_coef == 0.0` 时关闭 consistency。
    pub fn train_unroll(
        &self,
        obs_t: &[f32],
        actions: &[usize],
        target_policies: &[Vec<f32>],
        target_values: &[f32],
        target_rewards: &[f32],
        next_obs: &[Option<Vec<f32>>],
        consistency_coef: f32,
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

        let step_scale = if k > 0 { 1.0 / k as f32 } else { 1.0 };

        // value prefix（忠实版）：训练目标用库层 helper 预计算累计前缀（与搜索期 prefix 增量同口径）
        let vp_targets = if self.use_value_prefix {
            reward_prefix_targets(target_rewards)
        } else {
            Vec::new()
        };
        // LSTM hidden 在 K 步 unroll 上累计；base 模式为 None。
        let mut vp: Option<(Var, Var)> = if self.use_value_prefix {
            let zeros = vec![0.0; self.lstm.hidden];
            let h0 = self
                .graph
                .input(&Tensor::new(&zeros, &[1, self.lstm.hidden]))?;
            let c0 = self
                .graph
                .input(&Tensor::new(&zeros, &[1, self.lstm.hidden]))?;
            Some((h0, c0))
        } else {
            None
        };

        for i in 0..k {
            let oh = self.action_to_onehot(actions[i]);
            let oh_tensor = Tensor::new(&oh, &[1, self.action_dim]);
            let oh_var = self.graph.input(&oh_tensor)?;

            let (next_latent, pred_reward_logits) = self.dyn_net.forward(&latent, &oh_var)?;
            let (pred_p, pred_v_logits) = self.pred.forward(&next_latent);

            let tp = Tensor::new(&target_policies[i + 1], &[1, self.action_dim]);
            let tv = self.two_hot_target(target_values[i + 1]);

            let step_policy_loss = pred_p.cross_entropy(&tp)?;
            let step_value_loss = pred_v_logits.cross_entropy(&tv)?;

            // reward loss：base 用 dynamics reward head；value prefix 忠实版用 LSTM 累计前缀
            let step_reward_loss = if let Some((vp_h, vp_c)) = vp.as_mut() {
                let (h_new, c_new) = self.lstm.step(&next_latent, vp_h, vp_c)?;
                let prefix_logits = self.lstm.prefix_logits(&h_new);
                let tprefix = self.two_hot_target(vp_targets[i]); // 累计 reward 前缀（库层 helper）
                *vp_h = h_new;
                *vp_c = c_new;
                prefix_logits.cross_entropy(&tprefix)?
            } else {
                let tr = self.two_hot_target(target_rewards[i]);
                pred_reward_logits.cross_entropy(&tr)?
            };

            let mut step_loss = &step_policy_loss
                + &(&step_value_loss * loss::VALUE_LOSS_COEF)
                + &(&step_reward_loss * loss::REWARD_LOSS_COEF);

            // +consistency（SimSiam 负余弦）：online = pred_head(proj(next_latent))，
            // target = proj(repr(next_obs))（stop-grad 在 helper 内）。absorbing/越界跳过。
            if consistency_coef > 0.0 {
                if let Some(obs_next) = next_obs.get(i).and_then(|o| o.as_ref()) {
                    let obs_tensor = Tensor::new(obs_next, &[1, obs_next.len()]);
                    let target_latent = self.repr.forward(&obs_tensor)?;
                    let z = self.proj.forward(&target_latent);
                    let p = self.pred_head.forward(&self.proj.forward(&next_latent));
                    let cons = negative_cosine_similarity(&p, &z)?;
                    step_loss = &step_loss + &(&cons * consistency_coef);
                }
            }

            total_loss = &total_loss + &step_loss.scale_gradient(step_scale);
            latent = next_latent.scale_gradient(loss::DYNAMICS_GRADIENT_SCALE);
        }

        Ok(total_loss)
    }

    fn initial_state_impl(&self, obs: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        let obs_tensor = Tensor::new(obs, &[1, obs.len()]);
        let latent_var = self.repr.forward(&obs_tensor).expect("repr forward 失败");
        let latent_tensor = latent_var.value().unwrap().unwrap();

        let (policy_var, value_logits_var) = self.pred.forward(&latent_var);
        let policy_tensor = policy_var.value().unwrap().unwrap();
        let value_logits = value_logits_var.value().unwrap().unwrap();

        let policy_probs = policy_tensor.softmax(1);

        // base: state = latent；value prefix 忠实版：state = [latent || h(0) || c(0) || prefix(0)]
        let mut state = latent_tensor.data_as_slice().to_vec();
        if self.use_value_prefix {
            state.extend(vec![0.0; 2 * self.lstm.hidden + 1]);
        }
        let policy_vec = policy_probs.data_as_slice().to_vec();
        let value = Self::decode_categorical(&value_logits);

        (state, policy_vec, value)
    }

    fn recurrent_impl(&self, state: &[f32], action: &ActionPayload) -> DynamicsOutput {
        let action_idx = match action {
            ActionPayload::Discrete(idx) => *idx,
            _ => 0,
        };

        // latent 取 state 头部（base：state 即 latent；value prefix：[latent || h || c || prefix]）
        let latent_tensor = Tensor::new(&state[..self.latent_dim], &[1, self.latent_dim]);
        let latent_var = self.graph.input(&latent_tensor).unwrap();

        let mut oh = vec![0.0; self.action_dim];
        if action_idx < self.action_dim {
            oh[action_idx] = 1.0;
        }
        let oh_tensor = Tensor::new(&oh, &[1, self.action_dim]);
        let oh_var = self.graph.input(&oh_tensor).unwrap();

        let (next_latent_var, reward_logits_var) =
            self.dyn_net.forward(&latent_var, &oh_var).unwrap();
        let next_latent_tensor = next_latent_var.value().unwrap().unwrap();

        let (policy_var, value_logits_var) = self.pred.forward(&next_latent_var);
        let policy_tensor = policy_var.value().unwrap().unwrap();
        let value_logits = value_logits_var.value().unwrap().unwrap();

        let policy_probs = policy_tensor.softmax(1);
        let value = Self::decode_categorical(&value_logits);
        let prior = policy_probs.data_as_slice().to_vec();

        if self.use_value_prefix {
            // value prefix 忠实版：LSTM cell 一步，reward = prefix 增量，hidden 写回 state
            let h = self.lstm.hidden;
            let (h_off, c_off, p_off) = (
                self.latent_dim,
                self.latent_dim + h,
                self.latent_dim + 2 * h,
            );
            let prev_prefix = state.get(p_off).copied().unwrap_or(0.0);
            let h_in = self
                .graph
                .input(&Tensor::new(&state[h_off..c_off], &[1, h]))
                .unwrap();
            let c_in = self
                .graph
                .input(&Tensor::new(&state[c_off..p_off], &[1, h]))
                .unwrap();
            let (h_new, c_new) = self.lstm.step(&next_latent_var, &h_in, &c_in).unwrap();
            let value_prefix = Self::decode_categorical(
                &self.lstm.prefix_logits(&h_new).value().unwrap().unwrap(),
            );
            let reward = value_prefix - prev_prefix;

            let mut next_state = next_latent_tensor.data_as_slice().to_vec();
            next_state.extend(
                h_new
                    .value()
                    .unwrap()
                    .unwrap()
                    .data_as_slice()
                    .iter()
                    .copied(),
            );
            next_state.extend(
                c_new
                    .value()
                    .unwrap()
                    .unwrap()
                    .data_as_slice()
                    .iter()
                    .copied(),
            );
            next_state.push(value_prefix);

            DynamicsOutput {
                next_state,
                reward,
                prior,
                value,
                terminal: false,
            }
        } else {
            let reward = Self::decode_categorical(&reward_logits_var.value().unwrap().unwrap());
            DynamicsOutput {
                next_state: next_latent_tensor.data_as_slice().to_vec(),
                reward,
                prior,
                value,
                terminal: false,
            }
        }
    }
}

// 搜索期推理（detach，不走计算图）
impl Dynamics for &EzModel {
    fn initial_state(&self, obs: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        (**self).initial_state_impl(obs)
    }

    fn recurrent(&self, state: &[f32], action: &ActionPayload) -> DynamicsOutput {
        (**self).recurrent_impl(state, action)
    }
}
