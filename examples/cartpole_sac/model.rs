//! CartPole SAC-Discrete 模型定义
//!
//! SAC（Soft Actor-Critic）的离散动作版本
//!
//! ## 网络结构
//! ```text
//! Actor:  Input(4) -> Linear(64, ReLU) -> Linear(2) -> Softmax → 动作概率
//! Critic: Input(4) -> Linear(64, ReLU) -> Linear(2) → 每个动作的 Q 值
//! ```

use only_torch::nn::{Graph, GraphError, Linear, ModelState, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

// ============================================================================
// Actor 网络（策略网络）
// ============================================================================

/// SAC Actor：输出离散动作的概率分布
pub struct SacActor {
    fc1: Linear,
    fc2: Linear,
    state: ModelState,
}

impl SacActor {
    pub fn new(graph: &Graph, obs_dim: usize, action_dim: usize) -> Result<Self, GraphError> {
        Ok(Self {
            fc1: Linear::new(graph, obs_dim, 64, true, "actor_fc1")?,
            fc2: Linear::new(graph, 64, action_dim, true, "actor_fc2")?,
            state: ModelState::new(graph),
        })
    }

    /// 前向传播，返回 logits
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        self.state.forward(x, |input| {
            let h = self.fc1.forward(input).relu();
            Ok(self.fc2.forward(&h))
        })
    }

    /// 获取动作概率和 log 概率（纯 Tensor 操作，不创建计算图节点）
    ///
    /// 这个方法用于动作选择，不需要梯度，所以直接用 Tensor 的 softmax。
    pub fn get_action_probs(&self, x: &Tensor) -> Result<(Tensor, Tensor), GraphError> {
        let logits = self.forward(x)?;
        logits.forward()?; // 触发前向传播
        let logits_val = logits.value()?.unwrap();

        // 使用 Tensor 的 softmax（不创建 Var 节点）
        let probs_val = logits_val.softmax(1);

        // log_probs = ln(probs + eps) 防止 log(0)
        let eps = 1e-8;
        let log_probs = (probs_val.clone() + eps).ln();

        Ok((probs_val, log_probs))
    }

    /// 采样动作（根据概率分布）
    pub fn sample_action(&self, probs: &Tensor, rng: &mut impl rand::Rng) -> usize {
        let p0 = probs[[0, 0]];
        let r: f32 = rng.gen_range(0.0..1.0);
        if r < p0 { 0 } else { 1 }
    }

    /// 清空 ModelState 缓存（配合 `graph.prune_nodes_after()` 使用）
    pub fn clear_cache(&self) {
        self.state.clear_cache();
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
/// 离散 SAC 中，Q 网络输出 [batch, action_dim]，每个动作一个 Q 值
pub struct SacCritic {
    fc1: Linear,
    fc2: Linear,
    state: ModelState,
}

impl SacCritic {
    pub fn new(graph: &Graph, obs_dim: usize, action_dim: usize, name: &str) -> Result<Self, GraphError> {
        Ok(Self {
            fc1: Linear::new(graph, obs_dim, 64, true, &format!("{}_fc1", name))?,
            fc2: Linear::new(graph, 64, action_dim, true, &format!("{}_fc2", name))?,
            state: ModelState::new(graph),
        })
    }

    /// 前向传播，返回每个动作的 Q 值
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        self.state.forward(x, |input| {
            let h = self.fc1.forward(input).relu();
            Ok(self.fc2.forward(&h))
        })
    }

    /// 获取 Q 值张量
    pub fn get_q_values(&self, x: &Tensor) -> Result<Tensor, GraphError> {
        let q = self.forward(x)?;
        Ok(q.value()?.unwrap())
    }

    /// 清空 ModelState 缓存（配合 `graph.prune_nodes_after()` 使用）
    pub fn clear_cache(&self) {
        self.state.clear_cache();
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
    pub tau: f32,        // 软更新系数
    pub alpha_lr: f32,   // alpha 学习率
}

impl SacAgent {
    pub fn new(
        graph: &Graph,
        obs_dim: usize,
        action_dim: usize,
    ) -> Result<Self, GraphError> {
        // 目标熵：离散动作的最大熵 = ln(|A|)（均匀分布时）
        // 设为最大熵的一定比例，这里取 0.5，即目标熵约为最大熵的一半
        // 也可尝试 0.98 * ln(|A|) 或其他值来调节探索-利用平衡
        let target_entropy = 0.5 * (action_dim as f32).ln();
        
        Ok(Self {
            actor: SacActor::new(graph, obs_dim, action_dim)?,
            critic1: SacCritic::new(graph, obs_dim, action_dim, "critic1")?,
            critic2: SacCritic::new(graph, obs_dim, action_dim, "critic2")?,
            target_critic1: SacCritic::new(graph, obs_dim, action_dim, "target_critic1")?,
            target_critic2: SacCritic::new(graph, obs_dim, action_dim, "target_critic2")?,
            
            log_alpha: 0.0,  // 初始 alpha = exp(0) = 1
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
        self.target_critic1.soft_update_from(&self.critic1, self.tau);
        self.target_critic2.soft_update_from(&self.critic2, self.tau);
    }

    /// 更新 alpha（温度参数）
    ///
    /// SAC 的 alpha loss: L_α = α * (H(π) - target_entropy)
    /// 梯度: ∂L/∂log_α = α * (H(π) - target_entropy)
    ///
    /// - 若 H(π) < target_entropy，梯度为负，log_α 增大，α 增大，鼓励更多探索
    /// - 若 H(π) > target_entropy，梯度为正，log_α 减小，α 减小，减少探索
    pub fn update_alpha(&mut self, log_probs: &Tensor, probs: &Tensor) {
        let entropy = self.compute_entropy(log_probs, probs);
        let alpha = self.alpha();
        // 梯度 = α * (H(π) - target_entropy)，注意这里乘以 α
        let alpha_grad = alpha * (entropy - self.target_entropy);
        self.log_alpha -= self.alpha_lr * alpha_grad;
    }

    /// 计算策略熵 H(π) = -Σ π(a|s) * log π(a|s)（向量化实现）
    fn compute_entropy(&self, log_probs: &Tensor, probs: &Tensor) -> f32 {
        // H = -Σ p * log(p)，返回 batch 平均熵
        let batch_size = probs.shape()[0] as f32;
        let neg_entropy = (probs * log_probs).sum_axis_keepdims(1); // [batch, 1]
        -neg_entropy.sum().get_data_number().unwrap() / batch_size
    }

    /// 清空所有模型的 ModelState 缓存
    ///
    /// 配合 `graph.prune_nodes_after()` 使用：prune 会删除缓存引用的节点，
    /// 必须同时清空缓存以避免访问已删除的节点。
    pub fn clear_all_caches(&self) {
        self.actor.clear_cache();
        self.critic1.clear_cache();
        self.critic2.clear_cache();
        self.target_critic1.clear_cache();
        self.target_critic2.clear_cache();
    }
}
