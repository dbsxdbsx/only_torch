//! `CartPole` SAC-Discrete 模型定义
//!
//! SAC（Soft Actor-Critic）的离散动作版本
//!
//! ## 网络结构
//! ```text
//! Actor:  Input(4) -> Linear(64, ReLU) -> Linear(2) -> Softmax → 动作概率
//! Critic: Input(4) -> Linear(64, ReLU) -> Linear(2) → 每个动作的 Q 值
//! ```

use only_torch::nn::distributions::Categorical;
use only_torch::nn::{Graph, GraphError, IntoVar, Linear, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

// ============================================================================
// Actor 网络（策略网络）
// ============================================================================

/// SAC Actor：输出离散动作的概率分布
pub struct SacActor {
    fc1: Linear,
    fc2: Linear,
}

impl SacActor {
    pub fn new(graph: &Graph, obs_dim: usize, action_dim: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("Actor");
        Ok(Self {
            fc1: Linear::new(&graph, obs_dim, 64, true, "fc1")?,
            fc2: Linear::new(&graph, 64, action_dim, true, "fc2")?,
        })
    }

    /// 前向传播，返回 logits
    ///
    /// 接受 `&Tensor`（自动创建 Input 节点）或 `&Var`（复用已有 Var）
    pub fn forward(&self, x: impl IntoVar) -> Result<Var, GraphError> {
        let h = self.fc1.forward(x).relu();
        Ok(self.fc2.forward(&h))
    }

    /// 从 logits 构建 Categorical 分布并采样动作
    ///
    /// 返回 (action_index, probs_tensor)。
    pub fn sample_action(&self, x: &Tensor) -> Result<(usize, Tensor), GraphError> {
        let logits = self.forward(x)?;
        let dist = Categorical::new(logits);
        let action_tensor = dist.sample(); // Tensor [1, 1]
        let action = action_tensor[[0, 0]] as usize;
        let probs = dist.probs().value()?.unwrap(); // Var → Tensor
        Ok((action, probs))
    }

    /// 获取动作概率和 log 概率（纯 Tensor 操作，用于 target V 计算）
    pub fn get_action_probs(&self, x: &Tensor) -> Result<(Tensor, Tensor), GraphError> {
        let logits_val = self.forward(x)?.value()?.unwrap();
        // log_softmax 比 softmax + ln 数值更稳定，无需手动加 eps
        Ok((logits_val.softmax(1), logits_val.log_softmax(1)))
    }
}

impl Module for SacActor {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}

// ============================================================================
// Critic 网络（Q 网络）
// ============================================================================

/// SAC Critic：输出每个动作的 Q 值
///
/// 离散 SAC 中，Q 网络输出 [batch, `action_dim`]，每个动作一个 Q 值
pub struct SacCritic {
    fc1: Linear,
    fc2: Linear,
}

impl SacCritic {
    pub fn new(
        graph: &Graph,
        obs_dim: usize,
        action_dim: usize,
        name: &str,
    ) -> Result<Self, GraphError> {
        let graph = graph.with_model_name(name);
        Ok(Self {
            fc1: Linear::new(&graph, obs_dim, 64, true, "fc1")?,
            fc2: Linear::new(&graph, 64, action_dim, true, "fc2")?,
        })
    }

    /// 前向传播，返回每个动作的 Q 值
    ///
    /// 接受 `&Tensor`（自动创建 Input 节点）或 `&Var`（复用已有 Var）
    pub fn forward(&self, x: impl IntoVar) -> Result<Var, GraphError> {
        let h = self.fc1.forward(x).relu();
        Ok(self.fc2.forward(&h))
    }

    /// 获取 Q 值张量（不保留计算图）
    pub fn get_q_values(&self, x: &Tensor) -> Result<Tensor, GraphError> {
        let q = self.forward(x)?;
        Ok(q.value()?.unwrap())
    }
}

impl Module for SacCritic {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}

// ============================================================================
// SAC Agent（完整的 SAC 智能体）
// ============================================================================

/// SAC-Discrete 智能体
///
/// 包含：Actor、双 Q 网络、目标网络、温度参数
pub struct SacAgent {
    pub actor: SacActor,
    pub critic1: SacCritic,
    pub critic2: SacCritic,
    // 目标网络（用于计算 target Q）
    pub target_critic1: SacCritic,
    pub target_critic2: SacCritic,

    // 温度参数（控制探索程度）
    pub log_alpha: f32,
    pub target_entropy: f32,

    // 超参数
    pub tau: f32,      // 软更新系数
    pub alpha_lr: f32, // alpha 学习率
}

impl SacAgent {
    pub fn new(graph: &Graph, obs_dim: usize, action_dim: usize) -> Result<Self, GraphError> {
        // 目标熵：离散动作的最大熵 = ln(|A|)（均匀分布时）
        // 设为最大熵的一定比例，这里取 0.5，即目标熵约为最大熵的一半
        // 也可尝试 0.98 * ln(|A|) 或其他值来调节探索-利用平衡
        let target_entropy = 0.5 * (action_dim as f32).ln();

        Ok(Self {
            actor: SacActor::new(graph, obs_dim, action_dim)?,
            critic1: SacCritic::new(graph, obs_dim, action_dim, "Critic1")?,
            critic2: SacCritic::new(graph, obs_dim, action_dim, "Critic2")?,
            target_critic1: SacCritic::new(graph, obs_dim, action_dim, "TargetCritic1")?,
            target_critic2: SacCritic::new(graph, obs_dim, action_dim, "TargetCritic2")?,

            log_alpha: 0.0, // 初始 alpha = exp(0) = 1
            target_entropy,

            tau: 0.005,
            alpha_lr: 0.001,
        })
    }

    /// 获取当前 alpha 值
    pub fn alpha(&self) -> f32 {
        self.log_alpha.exp()
    }

    /// 软更新目标网络
    /// target = tau * source + (1 - tau) * target
    pub fn soft_update_targets(&self) {
        self.target_critic1
            .soft_update_from(&self.critic1, self.tau);
        self.target_critic2
            .soft_update_from(&self.critic2, self.tau);
    }

    /// 更新 alpha（温度参数）
    ///
    /// SAC 的 alpha loss: `L_α` = α * (H(π) - `target_entropy`)
    /// 梯度: ∂`L/∂log_α` = α * (H(π) - `target_entropy`)
    ///
    /// - 若 H(π) < `target_entropy`，α 增大，鼓励更多探索
    /// - 若 H(π) > `target_entropy`，α 减小，减少探索
    ///
    /// # 参数
    /// - `avg_entropy` — 当前策略的 batch 平均熵（来自 `Categorical::entropy()`）
    pub fn update_alpha(&mut self, avg_entropy: f32) {
        let alpha = self.alpha();
        let alpha_grad = alpha * (avg_entropy - self.target_entropy);
        self.log_alpha -= self.alpha_lr * alpha_grad;
    }
}
