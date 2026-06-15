//! MuZero CartPole 模型定义（categorical value/reward + latent min-max 归一化）
//!
//! 三网络架构：
//! - Representation h: obs(4) → latent(64)，输出经 **min-max 归一化到 [0,1]**
//! - Dynamics g: (latent(64), action_onehot(2)) → (next_latent(64), reward_logits)
//!   next_latent 同样 min-max 归一化
//! - Prediction f: latent(64) → (policy_logits(2), value_logits)
//!
//! value/reward 采用 **categorical 表示**（canonical MuZero）：head 输出 support 上的
//! logits，训练用 two-hot 目标 + 交叉熵，搜索期取 softmax 期望并 h⁻¹ 还原标量。

use only_torch::nn::{
    Graph, GraphError, IntoVar, Linear, Module, Var, VarActivationOps, VarLossOps, VarReduceOps,
    VarShapeOps,
};
use only_torch::rl::algo::muzero::{loss, scalar_to_two_hot, two_hot_to_scalar, SupportConfig};
use only_torch::rl::mcts::{ActionPayload, Dynamics, DynamicsOutput};
use only_torch::tensor::Tensor;

/// Categorical value/reward 的 support 半宽。
///
/// support = `2*20+1 = 41` 个原子，覆盖变换域 `[-20,20]` → value 域约 `±420`，
/// 足以容纳 CartPole（gamma=0.997，truncation bootstrap 后 value 趋近 `1/(1-γ)≈333`，
/// `h(333)≈17.6`）的目标范围且留有余量。
pub const SUPPORT_HALF: usize = 20;

/// 全局 support 配置（value 与 reward 共用，对齐 canonical MuZero）。
pub const SUPPORT: SupportConfig = SupportConfig::new(SUPPORT_HALF);

/// latent min-max 归一化到 [0,1]（canonical MuZero，逐样本沿特征维）
///
/// `s_norm = (s - min(s)) / (max(s) - min(s) + eps)`。本示例 train/search 均 batch=1，
/// 故按 `[1, dim]` 处理；用 `repeat` 显式对齐形状（不依赖隐式广播）。
///
/// 梯度经 `amin`/`amax`（梯度只流向极值位置）+ `repeat`（梯度求和回传）正确反传。
fn min_max_normalize(latent: &Var, dim: usize) -> Result<Var, GraphError> {
    let min_v = latent.amin(1).reshape(&[1, 1])?; // [1,1]
    let max_v = latent.amax(1).reshape(&[1, 1])?; // [1,1]
    let range = (&max_v - &min_v) + 1e-5_f32; // [1,1]，加 eps 防除零
    let min_b = min_v.repeat(&[1, dim])?; // [1, dim]
    let range_b = range.repeat(&[1, dim])?; // [1, dim]
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
    latent_dim: usize,
}

impl DynamicsNet {
    pub fn new(graph: &Graph, latent_dim: usize, action_dim: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("Dyn");
        let input_dim = latent_dim + action_dim;
        Ok(Self {
            fc1: Linear::new(&graph, input_dim, 128, true, "fc1")?,
            fc_latent: Linear::new(&graph, 128, latent_dim, true, "fc_latent")?,
            // reward head 输出 categorical logits（support 大小）
            fc_reward: Linear::new(&graph, 128, SUPPORT.size(), true, "fc_reward")?,
            latent_dim,
        })
    }

    /// (latent, action_onehot) → (next_latent[min-max], reward_logits)
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
            // value head 输出 categorical logits（support 大小）
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
// MuZero 组合模型
// ============================================================================

pub struct MuZeroModel {
    pub repr: RepresentationNet,
    pub dyn_net: DynamicsNet,
    pub pred: PredictionNet,
    pub graph: Graph,
    pub action_dim: usize,
    pub latent_dim: usize,
}

impl MuZeroModel {
    pub fn new(
        graph: &Graph,
        obs_dim: usize,
        action_dim: usize,
        latent_dim: usize,
    ) -> Result<Self, GraphError> {
        Ok(Self {
            repr: RepresentationNet::new(graph, obs_dim, latent_dim)?,
            dyn_net: DynamicsNet::new(graph, latent_dim, action_dim)?,
            pred: PredictionNet::new(graph, latent_dim, action_dim)?,
            graph: graph.clone(),
            action_dim,
            latent_dim,
        })
    }

    pub fn parameters(&self) -> Vec<Var> {
        [
            self.repr.parameters(),
            self.dyn_net.parameters(),
            self.pred.parameters(),
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

    /// 标量 value/reward → two-hot 目标张量 [1, support_size]
    fn two_hot_target(&self, x: f32) -> Tensor {
        Tensor::new(&scalar_to_two_hot(x, &SUPPORT), &[1, SUPPORT.size()])
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
    /// # absorbing state（终止处理，canonical MuZero）
    /// 终止后的 unroll 位置由调用方（`train_batch`）填入 **absorbing 目标**：
    /// `reward=0 / value=0 / policy=uniform`。模型据此学到「终局之后回报恒 0」，
    /// 使搜索推演越过终局时 reward→0 自然刹车，掐断 no-terminal 价值膨胀——
    /// 无需显式 done 头（这是原版 MuZero 的做法）。
    pub fn train_unroll(
        &self,
        obs_t: &[f32],
        actions: &[usize],
        target_policies: &[Vec<f32>],
        target_values: &[f32],
        target_rewards: &[f32],
    ) -> Result<Var, GraphError> {
        let k = actions.len();

        let obs_tensor = Tensor::new(obs_t, &[1, obs_t.len()]);
        let mut latent = self.repr.forward(&obs_tensor)?;

        // 初始预测（t 位置的 policy/value 损失）
        let (pred_policy, pred_value_logits) = self.pred.forward(&latent);
        let target_p0 = Tensor::new(&target_policies[0], &[1, self.action_dim]);
        let target_v0 = self.two_hot_target(target_values[0]);
        let mut total_loss = pred_policy.cross_entropy(&target_p0)?;
        total_loss =
            &total_loss + &(&pred_value_logits.cross_entropy(&target_v0)? * loss::VALUE_LOSS_COEF);

        // K 步展开
        for i in 0..k {
            let oh = self.action_to_onehot(actions[i]);
            let oh_tensor = Tensor::new(&oh, &[1, self.action_dim]);
            let oh_var = self.graph.input(&oh_tensor)?;

            let (next_latent, pred_reward_logits) = self.dyn_net.forward(&latent, &oh_var)?;
            let (pred_p, pred_v_logits) = self.pred.forward(&next_latent);

            let tp = Tensor::new(&target_policies[i + 1], &[1, self.action_dim]);
            let tv = self.two_hot_target(target_values[i + 1]);
            let tr = self.two_hot_target(target_rewards[i]);

            let step_policy_loss = pred_p.cross_entropy(&tp)?;
            let step_value_loss = pred_v_logits.cross_entropy(&tv)?;
            let step_reward_loss = pred_reward_logits.cross_entropy(&tr)?;

            // prediction head 不缩放；dynamics 贡献通过 latent 的梯度自然衰减
            let step_loss = &step_policy_loss
                + &(&step_value_loss * loss::VALUE_LOSS_COEF)
                + &(&step_reward_loss * loss::REWARD_LOSS_COEF);

            total_loss = &total_loss + &step_loss;

            latent = next_latent;
        }

        let scale = 1.0 / (k + 1) as f32;
        Ok(total_loss * scale)
    }
}

// ============================================================================
// impl Dynamics —— 搜索期推理（detach，不走计算图）
// ============================================================================

impl Dynamics for &MuZeroModel {
    fn initial_state(&self, obs: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        (**self).initial_state_impl(obs)
    }

    fn recurrent(&self, state: &[f32], action: &ActionPayload) -> DynamicsOutput {
        (**self).recurrent_impl(state, action)
    }
}

impl MuZeroModel {
    fn initial_state_impl(&self, obs: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        let obs_tensor = Tensor::new(obs, &[1, obs.len()]);
        let latent_var = self.repr.forward(&obs_tensor).expect("repr forward 失败");
        let latent_tensor = latent_var.value().unwrap().unwrap();

        let (policy_var, value_logits_var) = self.pred.forward(&latent_var);
        let policy_tensor = policy_var.value().unwrap().unwrap();
        let value_logits = value_logits_var.value().unwrap().unwrap();

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

        let latent_tensor = Tensor::new(state, &[1, self.latent_dim]);
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
        let reward_logits = reward_logits_var.value().unwrap().unwrap();

        let (policy_var, value_logits_var) = self.pred.forward(&next_latent_var);
        let policy_tensor = policy_var.value().unwrap().unwrap();
        let value_logits = value_logits_var.value().unwrap().unwrap();

        let policy_probs = policy_tensor.softmax(1);

        let reward = Self::decode_categorical(&reward_logits);
        let value = Self::decode_categorical(&value_logits);

        // canonical absorbing state：不设硬 terminal；模型已学到「终局后 reward→0」，
        // 搜索推演越过终局时回报自然归零，从而不再无限累加 +1（无需 done 头）。
        DynamicsOutput {
            next_state: next_latent_tensor.data_as_slice().to_vec(),
            reward,
            prior: policy_probs.data_as_slice().to_vec(),
            value,
            terminal: false,
        }
    }
}
