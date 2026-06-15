//! Platform-v0 Hybrid SAC 模型定义
//!
//! ## 网络结构
//! ```text
//! Actor:
//!   obs(10) → fc1(128, ReLU) → fc2(128, ReLU) → 共享特征
//!     ├─ 离散头 → logits(3) → Categorical（平台选择）
//!     └─ 连续头 → (mean(3), log_std(3)) → TanhNormal（跳跃参数）
//!
//! Critic:
//!   [obs(10), cont_action(3)] → Concat → 128(ReLU) → 128(ReLU) → Q_values(3)
//! ```
//!
//! Platform-v0 的 3 个连续参数始终同时生效，模型结构简洁。

use only_torch::nn::distributions::{Categorical, TanhNormal};
use only_torch::nn::{Graph, GraphError, IntoVar, Linear, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

/// 离散动作数量（平台选择：0, 1, 2）
const NUM_DISCRETE: usize = 3;
/// 连续动作维度（jump_force, lateral_velocity, time_to_jump）
const CONT_DIM: usize = 3;

// ============================================================================
// Actor
// ============================================================================

pub struct SacActor {
    fc1: Linear,
    fc2: Linear,
    discrete_head: Linear,
    cont_mean: Linear,
    cont_log_std: Linear,
}

impl SacActor {
    pub fn new(graph: &Graph, obs_dim: usize, hidden_dim: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("Actor");
        Ok(Self {
            fc1: Linear::new(&graph, obs_dim, hidden_dim, true, "fc1")?,
            fc2: Linear::new(&graph, hidden_dim, hidden_dim, true, "fc2")?,
            discrete_head: Linear::new(&graph, hidden_dim, NUM_DISCRETE, true, "discrete")?,
            cont_mean: Linear::new(&graph, hidden_dim, CONT_DIM, true, "cont_mean")?,
            cont_log_std: Linear::new(&graph, hidden_dim, CONT_DIM, true, "cont_log_std")?,
        })
    }

    fn shared_features(&self, x: impl IntoVar) -> Var {
        let h = self.fc1.forward(x).relu();
        self.fc2.forward(&h).relu()
    }

    /// 推理时选择动作（不保留计算图）
    ///
    /// 返回 (discrete_action, env_action_vec)
    /// - env_action_vec: `[discrete_idx, c0, c1, c2]`
    pub fn select_action(&self, obs: &Tensor) -> Result<(usize, Vec<f32>), GraphError> {
        let features = self.shared_features(obs);

        // 离散头
        let logits = self.discrete_head.forward(&features);
        let dist = Categorical::new(logits);
        let action_tensor = dist.sample();
        let discrete_action = action_tensor[[0, 0]] as usize;

        // 连续头
        let mean = self.cont_mean.forward(&features);
        let log_std = self.cont_log_std.forward(&features).clip(-5.0, 2.0);
        let std = log_std.exp();
        let cont_dist = TanhNormal::new(mean, std);
        let (cont_action_var, _log_prob) = cont_dist.rsample_and_log_prob();
        cont_action_var.forward()?;
        let cont_action = cont_action_var.value()?.unwrap();

        let mut action_vec = vec![discrete_action as f32];
        for i in 0..CONT_DIM {
            action_vec.push(cont_action[[0, i]]);
        }

        Ok((discrete_action, action_vec))
    }

    /// 训练时前向：离散概率 + 连续采样（保留计算图）
    ///
    /// 返回 (probs_var, log_probs_var, cont_action_var, cont_log_prob_var)
    /// - cont_log_prob_var: [batch, CONT_DIM]
    pub fn forward_train(
        &self,
        obs: impl IntoVar,
    ) -> Result<(Var, Var, Var, Var), GraphError> {
        let features = self.shared_features(obs);

        let logits = self.discrete_head.forward(&features);
        let cat_dist = Categorical::new(logits);
        let probs = cat_dist.probs();
        let log_probs = cat_dist.log_probs();

        let mean = self.cont_mean.forward(&features);
        let log_std = self.cont_log_std.forward(&features).clip(-5.0, 2.0);
        let std = log_std.exp();
        let cont_dist = TanhNormal::new(mean, std);
        let (cont_action, cont_log_prob) = cont_dist.rsample_and_log_prob();

        Ok((probs, log_probs, cont_action, cont_log_prob))
    }

    /// 获取离散概率和 log 概率（纯 Tensor，用于 target V）
    pub fn get_discrete_probs(&self, obs: &Tensor) -> Result<(Tensor, Tensor), GraphError> {
        let features = self.shared_features(obs);
        let logits = self.discrete_head.forward(&features);
        let logits_t = logits.value()?.unwrap();
        Ok((logits_t.softmax(1), logits_t.log_softmax(1)))
    }

    /// 采样连续动作（纯 Tensor，用于 target V）
    pub fn sample_cont(&self, obs: &Tensor) -> Result<(Tensor, Tensor), GraphError> {
        let features = self.shared_features(obs);
        let mean = self.cont_mean.forward(&features);
        let log_std = self.cont_log_std.forward(&features).clip(-5.0, 2.0);
        let std = log_std.exp();
        let cont_dist = TanhNormal::new(mean, std);
        let (action, log_prob) = cont_dist.rsample_and_log_prob();
        action.forward()?;
        let action_t = action.value()?.unwrap();
        let log_prob_t = log_prob.value()?.unwrap();
        Ok((action_t, log_prob_t))
    }
}

impl Module for SacActor {
    fn parameters(&self) -> Vec<Var> {
        [
            self.fc1.parameters(),
            self.fc2.parameters(),
            self.discrete_head.parameters(),
            self.cont_mean.parameters(),
            self.cont_log_std.parameters(),
        ]
        .concat()
    }
}

// ============================================================================
// Critic
// ============================================================================

/// Hybrid Critic：输入 [obs, cont_action] → Q_values(3)
pub struct SacCritic {
    graph: Graph,
    fc1: Linear,
    fc2: Linear,
    out: Linear,
}

impl SacCritic {
    pub fn new(
        graph: &Graph,
        obs_dim: usize,
        hidden_dim: usize,
        name: &str,
    ) -> Result<Self, GraphError> {
        let model_graph = graph.with_model_name(name);
        let input_dim = obs_dim + CONT_DIM;
        Ok(Self {
            graph: model_graph.clone(),
            fc1: Linear::new(&model_graph, input_dim, hidden_dim, true, "fc1")?,
            fc2: Linear::new(&model_graph, hidden_dim, hidden_dim, true, "fc2")?,
            out: Linear::new(&model_graph, hidden_dim, NUM_DISCRETE, true, "out")?,
        })
    }

    /// 前向（Var 级别，保留计算图）
    pub fn forward_q(&self, obs: &Var, cont_action: &Var) -> Result<Var, GraphError> {
        let input = Var::concat(&[obs, cont_action], 1)?;
        let h = self.fc1.forward(&input).relu();
        let h = self.fc2.forward(&h).relu();
        Ok(self.out.forward(&h))
    }

    /// 获取 Q 值（纯 Tensor，不保留计算图）
    pub fn get_q_values(&self, obs: &Tensor, cont_action: &Tensor) -> Result<Tensor, GraphError> {
        let obs_var = obs.into_var(&self.graph)?;
        let act_var = cont_action.into_var(&self.graph)?;
        let q = self.forward_q(&obs_var, &act_var)?;
        Ok(q.value()?.unwrap())
    }
}

impl Module for SacCritic {
    fn parameters(&self) -> Vec<Var> {
        [
            self.fc1.parameters(),
            self.fc2.parameters(),
            self.out.parameters(),
        ]
        .concat()
    }
}

// ============================================================================
// Agent
// ============================================================================

pub struct SacAgent {
    pub actor: SacActor,
    pub critic1: SacCritic,
    pub critic2: SacCritic,
    pub target_critic1: SacCritic,
    pub target_critic2: SacCritic,
    pub log_alpha_d: f32,
    pub log_alpha_c: f32,
    pub target_entropy_d: f32,
    pub target_entropy_c: f32,
    pub alpha_lr: f32,
}

impl SacAgent {
    pub fn new(graph: &Graph, obs_dim: usize, hidden_dim: usize) -> Result<Self, GraphError> {
        Ok(Self {
            actor: SacActor::new(graph, obs_dim, hidden_dim)?,
            critic1: SacCritic::new(graph, obs_dim, hidden_dim, "Critic1")?,
            critic2: SacCritic::new(graph, obs_dim, hidden_dim, "Critic2")?,
            target_critic1: SacCritic::new(graph, obs_dim, hidden_dim, "TargetCritic1")?,
            target_critic2: SacCritic::new(graph, obs_dim, hidden_dim, "TargetCritic2")?,
            log_alpha_d: 0.0,
            log_alpha_c: 0.0,
            target_entropy_d: -(0.98 * (NUM_DISCRETE as f32).ln()),
            target_entropy_c: -(CONT_DIM as f32),
            alpha_lr: 3e-4,
        })
    }

    pub fn alpha_d(&self) -> f32 {
        self.log_alpha_d.exp()
    }

    pub fn alpha_c(&self) -> f32 {
        self.log_alpha_c.exp()
    }

    pub fn soft_update_targets(&self, tau: f32) {
        self.target_critic1.soft_update_from(&self.critic1, tau);
        self.target_critic2.soft_update_from(&self.critic2, tau);
    }
}
