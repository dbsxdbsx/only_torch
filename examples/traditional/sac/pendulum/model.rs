//! Pendulum SAC-Continuous 模型定义
//!
//! SAC（Soft Actor-Critic）的连续动作版本
//!
//! ## 网络结构
//! ```text
//! Actor:  obs(3) → 256(ReLU) → 256(ReLU) → [mean(1), log_std(1)]
//!         → TanhNormal → action ∈ [-1,1] → scale to [-2,2]
//! Critic: [obs(3), act(1)] → Concat → 256(ReLU) → 256(ReLU) → 1 (Q value)
//! ```

use only_torch::nn::distributions::TanhNormal;
use only_torch::nn::{Graph, GraphError, IntoVar, Linear, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

// ============================================================================
// Actor 网络（策略网络）
// ============================================================================

/// SAC Actor：输出连续动作的 TanhNormal 分布参数
pub struct SacActor {
    fc1: Linear,
    fc2: Linear,
    mean_head: Linear,
    log_std_head: Linear,
}

impl SacActor {
    pub fn new(
        graph: &Graph,
        obs_dim: usize,
        action_dim: usize,
        hidden_dim: usize,
    ) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("Actor");
        Ok(Self {
            fc1: Linear::new(&graph, obs_dim, hidden_dim, true, "fc1")?,
            fc2: Linear::new(&graph, hidden_dim, hidden_dim, true, "fc2")?,
            mean_head: Linear::new(&graph, hidden_dim, action_dim, true, "mean")?,
            log_std_head: Linear::new(&graph, hidden_dim, action_dim, true, "log_std")?,
        })
    }

    /// 前向传播，返回 (mean, std) 的 Var 对
    fn forward_dist_params(&self, x: impl IntoVar) -> Result<(Var, Var), GraphError> {
        let h = self.fc1.forward(x).relu();
        let h = self.fc2.forward(&h).relu();
        let mean = self.mean_head.forward(&h);
        // log_std 裁剪到 [-5, 2] 范围（CleanRL 默认，比 SpinUp 的 -20 更稳定）
        // -5 → std_min ≈ 0.0067，足够精确又避免极端确定性导致的数值问题
        let log_std = self.log_std_head.forward(&h).clip(-5.0, 2.0);
        let std = log_std.exp();
        Ok((mean, std))
    }

    /// 从观测采样动作（用于环境交互，不保留计算图）
    ///
    /// 返回 (action_tensor ∈ [-1,1], log_prob_tensor)
    pub fn sample_action(&self, x: &Tensor) -> Result<(Tensor, Tensor), GraphError> {
        let (mean, std) = self.forward_dist_params(x)?;
        let dist = TanhNormal::new(mean, std);
        let (action, log_prob) = dist.rsample_and_log_prob();
        // forward 计算值，然后提取 Tensor（脱离计算图）
        action.forward()?;
        let action_tensor = action.value()?.unwrap();
        let log_prob_tensor = log_prob.value()?.unwrap();
        Ok((action_tensor, log_prob_tensor))
    }

    /// 从观测构建分布并采样（Actor 更新时使用，保留计算图用于反向传播）
    ///
    /// 返回 (squashed_action_var ∈ [-1,1], log_prob_var)
    pub fn sample_for_update(&self, x: impl IntoVar) -> Result<(Var, Var), GraphError> {
        let (mean, std) = self.forward_dist_params(x)?;
        let dist = TanhNormal::new(mean, std);
        Ok(dist.rsample_and_log_prob())
    }
}

impl Module for SacActor {
    fn parameters(&self) -> Vec<Var> {
        [
            self.fc1.parameters(),
            self.fc2.parameters(),
            self.mean_head.parameters(),
            self.log_std_head.parameters(),
        ]
        .concat()
    }
}

// ============================================================================
// Critic 网络（Q 网络）
// ============================================================================

/// SAC Critic：输入 (obs, action)，输出标量 Q 值
///
/// 使用 `Var::concat` 拼接 obs 和 action，与 PyTorch/rustRL 标准实现一致
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
        action_dim: usize,
        hidden_dim: usize,
        name: &str,
    ) -> Result<Self, GraphError> {
        let model_graph = graph.with_model_name(name);
        Ok(Self {
            graph: model_graph.clone(),
            fc1: Linear::new(&model_graph, obs_dim + action_dim, hidden_dim, true, "fc1")?,
            fc2: Linear::new(&model_graph, hidden_dim, hidden_dim, true, "fc2")?,
            fc3: Linear::new(&model_graph, hidden_dim, 1, true, "fc3")?,
        })
    }

    /// 前向传播：拼接 obs 和 action（均为 Var），输出 Q 值
    pub fn forward_q(&self, obs: &Var, action: &Var) -> Result<Var, GraphError> {
        let input = Var::concat(&[obs, action], 1)?;
        let h = self.fc1.forward(&input).relu();
        let h = self.fc2.forward(&h).relu();
        Ok(self.fc3.forward(&h))
    }

    /// 获取 Q 值张量（不保留计算图，用于 target Q 计算）
    pub fn get_q_value(&self, obs: &Tensor, action: &Tensor) -> Result<Tensor, GraphError> {
        let obs_var = obs.into_var(&self.graph)?;
        let act_var = action.into_var(&self.graph)?;
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
// SAC Agent（完整的 SAC 智能体）
// ============================================================================

/// SAC-Continuous 智能体
///
/// 包含：Actor、双 Q 网络、目标网络、温度参数
pub struct SacAgent {
    pub actor: SacActor,
    pub critic1: SacCritic,
    pub critic2: SacCritic,
    pub target_critic1: SacCritic,
    pub target_critic2: SacCritic,

    // 温度参数
    pub log_alpha: f32,
    pub target_entropy: f32,

    // 超参数
    pub tau: f32,
    pub alpha_lr: f32,

    // 动作缩放参数
    pub action_scale: f32,
    pub action_bias: f32,
}

impl SacAgent {
    pub fn new(
        graph: &Graph,
        obs_dim: usize,
        action_dim: usize,
        hidden_dim: usize,
        action_low: f32,
        action_high: f32,
    ) -> Result<Self, GraphError> {
        // 连续动作的 target entropy: -d（SAC v2 默认）
        let target_entropy = -(action_dim as f32);

        // 动作缩放：TanhNormal 输出 [-1,1] → [low, high]
        let action_scale = (action_high - action_low) / 2.0;
        let action_bias = (action_high + action_low) / 2.0;

        Ok(Self {
            actor: SacActor::new(graph, obs_dim, action_dim, hidden_dim)?,
            critic1: SacCritic::new(graph, obs_dim, action_dim, hidden_dim, "Critic1")?,
            critic2: SacCritic::new(graph, obs_dim, action_dim, hidden_dim, "Critic2")?,
            target_critic1: SacCritic::new(
                graph,
                obs_dim,
                action_dim,
                hidden_dim,
                "TargetCritic1",
            )?,
            target_critic2: SacCritic::new(
                graph,
                obs_dim,
                action_dim,
                hidden_dim,
                "TargetCritic2",
            )?,

            log_alpha: 0.0,
            target_entropy,

            tau: 0.005,
            alpha_lr: 3e-4,

            action_scale,
            action_bias,
        })
    }

    /// 当前 alpha 值
    pub fn alpha(&self) -> f32 {
        self.log_alpha.exp()
    }

    /// 将 TanhNormal 输出 [-1,1] 缩放到环境动作范围
    pub fn scale_action(&self, tanh_action: &Tensor) -> Tensor {
        tanh_action * self.action_scale + self.action_bias
    }

    /// 将环境动作范围缩回 [-1,1]（用于 Critic 输入归一化）
    pub fn unscale_action(&self, env_action: &Tensor) -> Tensor {
        (env_action - self.action_bias) / self.action_scale
    }

    /// 软更新目标网络
    pub fn soft_update_targets(&self) {
        self.target_critic1
            .soft_update_from(&self.critic1, self.tau);
        self.target_critic2
            .soft_update_from(&self.critic2, self.tau);
    }

    /// 更新 alpha（温度参数）
    ///
    /// - `avg_entropy` — 当前策略的 batch 平均熵（从 log_prob 离线计算）
    pub fn update_alpha(&mut self, avg_entropy: f32) {
        let alpha = self.alpha();
        let alpha_grad = alpha * (avg_entropy - self.target_entropy);
        self.log_alpha -= self.alpha_lr * alpha_grad;
        // 安全 clamp
        self.log_alpha = self.log_alpha.clamp(-20.0, 2.0);
    }
}
