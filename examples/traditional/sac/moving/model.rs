//! Moving-v0 Hybrid SAC 模型定义（方式 B — 独立连续分支）
//!
//! ## 网络结构
//! ```text
//! Actor:
//!   obs(10) → fc1(256, ReLU) → fc2(256, ReLU) → 共享特征
//!     ├─ 离散头 → logits(3) → Categorical
//!     ├─ Accelerate 头 → (mean(1), log_std(1)) → TanhNormal
//!     └─ Turn 头 → (mean(1), log_std(1)) → TanhNormal
//!     (Brake 无连续头)
//!
//! Critic:
//!   [obs(10), continuous_action(2)] → Concat → 256(ReLU) → 256(ReLU) → Q_values(3)
//! ```
//!
//! 推理时按需调用连续分支（match），训练时全部分支前向以获得完整梯度。

use only_torch::nn::distributions::{Categorical, TanhNormal};
use only_torch::nn::{Graph, GraphError, IntoVar, Linear, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

/// 离散动作数量
const NUM_DISCRETE: usize = 3;
/// 连续动作总维度（acceleration + rotation，用于 Critic 输入）
const CONT_DIM: usize = 2;

// ============================================================================
// Actor 网络（方式 B：独立连续分支）
// ============================================================================

/// Hybrid SAC Actor：离散头 + 独立连续分支
///
/// - Accelerate(0)：1 维连续（acceleration ∈ [0, 1]）
/// - Turn(1)：1 维连续（rotation ∈ [-1, 1]）
/// - Brake(2)：无连续参数
pub struct SacActor {
    // 共享隐藏层
    fc1: Linear,
    fc2: Linear,
    // 离散头
    discrete_head: Linear,
    // 独立连续分支
    acc_mean: Linear,
    acc_log_std: Linear,
    turn_mean: Linear,
    turn_log_std: Linear,
}

impl SacActor {
    pub fn new(graph: &Graph, obs_dim: usize, hidden_dim: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("Actor");
        Ok(Self {
            fc1: Linear::new(&graph, obs_dim, hidden_dim, true, "fc1")?,
            fc2: Linear::new(&graph, hidden_dim, hidden_dim, true, "fc2")?,
            discrete_head: Linear::new(&graph, hidden_dim, NUM_DISCRETE, true, "discrete")?,
            acc_mean: Linear::new(&graph, hidden_dim, 1, true, "acc_mean")?,
            acc_log_std: Linear::new(&graph, hidden_dim, 1, true, "acc_log_std")?,
            turn_mean: Linear::new(&graph, hidden_dim, 1, true, "turn_mean")?,
            turn_log_std: Linear::new(&graph, hidden_dim, 1, true, "turn_log_std")?,
        })
    }

    /// 共享隐藏层前向
    fn shared_features(&self, x: impl IntoVar) -> Var {
        let h = self.fc1.forward(x).relu();
        self.fc2.forward(&h).relu()
    }

    /// 推理时选择动作（按需调用分支，不保留计算图）
    ///
    /// 返回 (discrete_action, env_action_vec)
    /// - env_action_vec: 展平的动作向量 [discrete_float, acc_or_0, rot_or_0]
    pub fn select_action(&self, obs: &Tensor) -> Result<(usize, Vec<f32>), GraphError> {
        let features = self.shared_features(obs);

        // 离散头
        let logits = self.discrete_head.forward(&features);
        let dist = Categorical::new(logits);
        let action_tensor = dist.sample();
        let discrete_action = action_tensor[[0, 0]] as usize;

        // 根据离散动作选择对应的连续分支
        match discrete_action {
            0 => {
                // Accelerate：采样 acceleration，缩放 [-1,1] → [0,1]
                let mean = self.acc_mean.forward(&features);
                let log_std = self.acc_log_std.forward(&features).clip(-5.0, 2.0);
                let std = log_std.exp();
                let tanh_dist = TanhNormal::new(mean, std);
                let (tanh_action, _) = tanh_dist.rsample_and_log_prob();
                tanh_action.forward()?;
                let tanh_val = tanh_action.value()?.unwrap()[[0, 0]];
                let acc = (tanh_val + 1.0) / 2.0; // [-1,1] → [0,1]
                Ok((0, vec![0.0, acc, 0.0]))
            }
            1 => {
                // Turn：采样 rotation，范围 [-1,1]，TanhNormal 直接输出
                let mean = self.turn_mean.forward(&features);
                let log_std = self.turn_log_std.forward(&features).clip(-5.0, 2.0);
                let std = log_std.exp();
                let tanh_dist = TanhNormal::new(mean, std);
                let (tanh_action, _) = tanh_dist.rsample_and_log_prob();
                tanh_action.forward()?;
                let rot = tanh_action.value()?.unwrap()[[0, 0]];
                Ok((1, vec![1.0, 0.0, rot]))
            }
            _ => {
                // Brake：无连续参数
                Ok((2, vec![2.0, 0.0, 0.0]))
            }
        }
    }

    /// 训练时全部分支前向（保留计算图用于梯度反向传播）
    ///
    /// 返回 `ActorTrainOutput` 供 Actor Loss 计算使用
    pub fn forward_all_branches(&self, obs: impl IntoVar) -> Result<ActorTrainOutput, GraphError> {
        let features = self.shared_features(obs);

        // 离散头
        let logits = self.discrete_head.forward(&features);
        let cat = Categorical::new(logits);
        let probs = cat.probs();
        let log_probs = cat.log_probs();

        // Accelerate 分支
        let acc_mean = self.acc_mean.forward(&features);
        let acc_log_std = self.acc_log_std.forward(&features).clip(-5.0, 2.0);
        let acc_std = acc_log_std.exp();
        let acc_dist = TanhNormal::new(acc_mean, acc_std);
        let (acc_tanh_action, acc_log_prob) = acc_dist.rsample_and_log_prob();
        // 缩放到 [0,1]：acc_env = (tanh + 1) / 2
        let one = Tensor::new(&[1.0], &[1, 1]);
        let half = Tensor::new(&[0.5], &[1, 1]);
        let acc_env_action = (acc_tanh_action + one) * half;

        // Turn 分支
        let turn_mean = self.turn_mean.forward(&features);
        let turn_log_std = self.turn_log_std.forward(&features).clip(-5.0, 2.0);
        let turn_std = turn_log_std.exp();
        let turn_dist = TanhNormal::new(turn_mean, turn_std);
        let (turn_tanh_action, turn_log_prob) = turn_dist.rsample_and_log_prob();
        // Turn 范围 [-1,1]，TanhNormal 直接输出

        Ok(ActorTrainOutput {
            probs,
            log_probs,
            // 各分支的连续动作（用于 Critic 评估）
            acc_env_action,                    // [batch, 1]，范围 [0, 1]
            turn_env_action: turn_tanh_action, // [batch, 1]，范围 [-1, 1]
            // 各分支的 log_prob_c
            acc_log_prob,  // [batch, 1]
            turn_log_prob, // [batch, 1]
        })
    }
}

impl Module for SacActor {
    fn parameters(&self) -> Vec<Var> {
        [
            self.fc1.parameters(),
            self.fc2.parameters(),
            self.discrete_head.parameters(),
            self.acc_mean.parameters(),
            self.acc_log_std.parameters(),
            self.turn_mean.parameters(),
            self.turn_log_std.parameters(),
        ]
        .concat()
    }
}

/// Actor 训练时的全分支输出
pub struct ActorTrainOutput {
    /// 离散动作概率 [batch, 3]
    pub probs: Var,
    /// 离散动作 log 概率 [batch, 3]
    pub log_probs: Var,
    /// Accelerate 分支的环境动作 [batch, 1]（[0, 1]）
    pub acc_env_action: Var,
    /// Turn 分支的环境动作 [batch, 1]（[-1, 1]）
    pub turn_env_action: Var,
    /// Accelerate 分支的 log_prob_c [batch, 1]
    pub acc_log_prob: Var,
    /// Turn 分支的 log_prob_c [batch, 1]
    pub turn_log_prob: Var,
}

// ============================================================================
// Critic 网络
// ============================================================================

/// Hybrid SAC Critic：输入 (obs, continuous_action)，输出每个离散动作的 Q 值
///
/// 输入维度：obs_dim + CONT_DIM (= 10 + 2 = 12)
/// 输出维度：NUM_DISCRETE (= 3)
pub struct SacCritic {
    graph: Graph,
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl SacCritic {
    pub fn new(
        graph: &Graph,
        obs_dim: usize,
        hidden_dim: usize,
        name: &str,
    ) -> Result<Self, GraphError> {
        let model_graph = graph.with_model_name(name);
        Ok(Self {
            graph: model_graph.clone(),
            fc1: Linear::new(&model_graph, obs_dim + CONT_DIM, hidden_dim, true, "fc1")?,
            fc2: Linear::new(&model_graph, hidden_dim, hidden_dim, true, "fc2")?,
            fc3: Linear::new(&model_graph, hidden_dim, NUM_DISCRETE, true, "fc3")?,
        })
    }

    /// 前向传播：拼接 obs 和 continuous_action（均为 Var），输出 Q 值 [batch, 3]
    pub fn forward_q(&self, obs: &Var, cont_action: &Var) -> Result<Var, GraphError> {
        let input = Var::concat(&[obs, cont_action], 1)?;
        let h = self.fc1.forward(&input).relu();
        let h = self.fc2.forward(&h).relu();
        Ok(self.fc3.forward(&h))
    }

    /// 获取 Q 值张量（不保留计算图，用于 target Q 计算）
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
            self.fc3.parameters(),
        ]
        .concat()
    }
}

// ============================================================================
// SAC Agent
// ============================================================================

/// Hybrid SAC 智能体
///
/// 包含：Actor（方式 B）、双 Q 网络、目标网络、双温度参数
pub struct SacAgent {
    pub actor: SacActor,
    pub critic1: SacCritic,
    pub critic2: SacCritic,
    pub target_critic1: SacCritic,
    pub target_critic2: SacCritic,

    // 双温度参数
    pub log_alpha_d: f32,
    pub log_alpha_c: f32,
    pub target_entropy_d: f32,
    pub target_entropy_c: f32,

    // 超参数
    pub tau: f32,
    pub alpha_lr: f32,
}

impl SacAgent {
    pub fn new(graph: &Graph, obs_dim: usize, hidden_dim: usize) -> Result<Self, GraphError> {
        // 离散 target entropy: 0.5 * ln(K)
        let target_entropy_d = 0.5 * (NUM_DISCRETE as f32).ln();
        // 连续 target entropy: -1.0（每个分支 1 维）
        let target_entropy_c = -1.0;

        Ok(Self {
            actor: SacActor::new(graph, obs_dim, hidden_dim)?,
            critic1: SacCritic::new(graph, obs_dim, hidden_dim, "Critic1")?,
            critic2: SacCritic::new(graph, obs_dim, hidden_dim, "Critic2")?,
            target_critic1: SacCritic::new(graph, obs_dim, hidden_dim, "TargetCritic1")?,
            target_critic2: SacCritic::new(graph, obs_dim, hidden_dim, "TargetCritic2")?,

            log_alpha_d: 0.0,
            log_alpha_c: 0.0,
            target_entropy_d,
            target_entropy_c,

            tau: 0.005,
            alpha_lr: 3e-4,
        })
    }

    /// 离散温度 α_d
    pub fn alpha_d(&self) -> f32 {
        self.log_alpha_d.exp()
    }

    /// 连续温度 α_c
    pub fn alpha_c(&self) -> f32 {
        self.log_alpha_c.exp()
    }

    /// 软更新目标网络
    pub fn soft_update_targets(&self) {
        self.target_critic1
            .soft_update_from(&self.critic1, self.tau);
        self.target_critic2
            .soft_update_from(&self.critic2, self.tau);
    }

    /// 更新 α_d（离散温度）
    pub fn update_alpha_d(&mut self, avg_discrete_entropy: f32) {
        let alpha = self.alpha_d();
        let grad = alpha * (avg_discrete_entropy - self.target_entropy_d);
        self.log_alpha_d -= self.alpha_lr * grad;
        self.log_alpha_d = self.log_alpha_d.clamp(-20.0, 2.0);
    }

    /// 更新 α_c（连续温度）
    pub fn update_alpha_c(&mut self, avg_continuous_entropy: f32) {
        let alpha = self.alpha_c();
        let grad = alpha * (avg_continuous_entropy - self.target_entropy_c);
        self.log_alpha_c -= self.alpha_lr * grad;
        self.log_alpha_c = self.log_alpha_c.clamp(-20.0, 2.0);
    }
}
