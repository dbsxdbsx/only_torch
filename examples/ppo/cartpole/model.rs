//! CartPole PPO 模型定义
//!
//! ## 网络结构
//! ```text
//! Actor:  Input(4) -> Linear(128, ReLU) -> Linear(2) → logits → Categorical
//! Critic: Input(4) -> Linear(128, ReLU) -> Linear(1) → V(s)
//! ```

use only_torch::nn::distributions::Categorical;
use only_torch::nn::{Graph, GraphError, IntoVar, Linear, Module, Var, VarActivationOps, VarShapeOps};
use only_torch::tensor::Tensor;

pub struct PpoActor {
    fc1: Linear,
    fc2: Linear,
}

impl PpoActor {
    pub fn new(graph: &Graph, obs_dim: usize, action_dim: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("Actor");
        Ok(Self {
            fc1: Linear::new(&graph, obs_dim, 128, true, "fc1")?,
            fc2: Linear::new(&graph, 128, action_dim, true, "fc2")?,
        })
    }

    pub fn forward(&self, x: impl IntoVar) -> Result<Var, GraphError> {
        let h = self.fc1.forward(x).relu();
        Ok(self.fc2.forward(&h))
    }

    /// 采样动作并返回 (action_index, log_prob)
    pub fn sample_action(&self, x: &Tensor) -> Result<(usize, f32), GraphError> {
        let logits = self.forward(x)?;
        let dist = Categorical::new(logits);
        let action_tensor = dist.sample();
        let action = action_tensor[[0, 0]] as usize;
        let log_probs = dist.log_probs().value()?.unwrap();
        let log_prob = log_probs[[0, action]];
        Ok((action, log_prob))
    }

    /// 获取离散分布的 log_prob 和 entropy（用于 PPO 更新）
    pub fn evaluate_actions(
        &self,
        obs: impl IntoVar,
        actions: &Tensor,
    ) -> Result<(Var, Var), GraphError> {
        let logits = self.forward(obs)?;
        let dist = Categorical::new(logits);
        let log_probs = dist.log_probs();
        let selected_log_probs = log_probs.gather(1, actions)?;
        let entropy = dist.entropy();
        Ok((selected_log_probs, entropy))
    }
}

impl Module for PpoActor {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}

pub struct PpoCritic {
    fc1: Linear,
    fc2: Linear,
}

impl PpoCritic {
    pub fn new(graph: &Graph, obs_dim: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("Critic");
        Ok(Self {
            fc1: Linear::new(&graph, obs_dim, 128, true, "fc1")?,
            fc2: Linear::new(&graph, 128, 1, true, "fc2")?,
        })
    }

    pub fn forward(&self, x: impl IntoVar) -> Result<Var, GraphError> {
        let h = self.fc1.forward(x).relu();
        Ok(self.fc2.forward(&h))
    }

    pub fn get_value(&self, x: &Tensor) -> Result<f32, GraphError> {
        let v = self.forward(x)?;
        let val = v.value()?.unwrap();
        Ok(val[[0, 0]])
    }
}

impl Module for PpoCritic {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}
