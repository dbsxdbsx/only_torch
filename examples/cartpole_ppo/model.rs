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

    /// 获取动作概率和 log 概率
    pub fn get_action_probs(&self, x: &Tensor) -> Result<(Tensor, Tensor), GraphError> {
        let logits = self.forward(x)?;
        let probs = logits.softmax();
        let probs_val = probs.value()?.unwrap();

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
        // 目标熵 = 0.5 * log(action_dim)，参考 SAC-Discrete 论文
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
    pub fn soft_update_targets(&mut self) {
        let tau = self.tau;
        // 更新 target_critic1
        for (target, source) in self.target_critic1.parameters().iter()
            .zip(self.critic1.parameters().iter())
        {
            if let (Ok(Some(mut t_val)), Ok(Some(s_val))) = (target.value(), source.value()) {
                t_val.soft_update(&s_val, tau);
                let _ = target.set_value(&t_val);
            }
        }
        // 更新 target_critic2
        for (target, source) in self.target_critic2.parameters().iter()
            .zip(self.critic2.parameters().iter())
        {
            if let (Ok(Some(mut t_val)), Ok(Some(s_val))) = (target.value(), source.value()) {
                t_val.soft_update(&s_val, tau);
                let _ = target.set_value(&t_val);
            }
        }
    }

    /// 更新 alpha（温度参数）
    pub fn update_alpha(&mut self, log_probs: &Tensor, probs: &Tensor) {
        // alpha_loss = -alpha * (log_pi + target_entropy)
        // 简化实现：直接用标量计算
        let entropy = self.compute_entropy(log_probs, probs);
        let alpha_grad = -(entropy - self.target_entropy);
        self.log_alpha -= self.alpha_lr * alpha_grad;
    }

    /// 计算策略熵 H(π) = -Σ π(a|s) * log π(a|s)
    fn compute_entropy(&self, log_probs: &Tensor, probs: &Tensor) -> f32 {
        // H = -Σ p * log(p)
        // 返回 batch 平均熵
        let mut entropy = 0.0;
        let batch_size = probs.shape()[0];
        let action_dim = probs.shape()[1];
        
        for b in 0..batch_size {
            for a in 0..action_dim {
                let p = probs[[b, a]];
                let log_p = log_probs[[b, a]];
                entropy -= p * log_p;
            }
        }
        entropy / batch_size as f32
    }
}
