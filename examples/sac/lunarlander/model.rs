//! LunarLander-v3 SAC-Discrete 模型
//!
//! 与 CartPole 同构（离散 SAC），仅维度不同：obs=8, actions=4。

use only_torch::nn::distributions::Categorical;
use only_torch::nn::{Graph, GraphError, IntoVar, Linear, Module, Var, VarActivationOps};
use only_torch::tensor::Tensor;

pub struct SacActor {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl SacActor {
    pub fn new(graph: &Graph, obs_dim: usize, action_dim: usize) -> Result<Self, GraphError> {
        let g = graph.with_model_name("Actor");
        Ok(Self {
            fc1: Linear::new(&g, obs_dim, 128, true, "fc1")?,
            fc2: Linear::new(&g, 128, 128, true, "fc2")?,
            fc3: Linear::new(&g, 128, action_dim, true, "fc3")?,
        })
    }
    pub fn forward(&self, x: impl IntoVar) -> Result<Var, GraphError> {
        let h = self.fc1.forward(x).relu();
        let h = self.fc2.forward(&h).relu();
        Ok(self.fc3.forward(&h))
    }
    pub fn sample_action(&self, x: &Tensor) -> Result<(usize, Tensor), GraphError> {
        let logits = self.forward(x)?;
        let dist = Categorical::new(logits);
        let action = dist.sample()[[0, 0]] as usize;
        Ok((action, dist.probs().value()?.unwrap()))
    }
    pub fn get_action_probs(&self, x: &Tensor) -> Result<(Tensor, Tensor), GraphError> {
        let v = self.forward(x)?.value()?.unwrap();
        Ok((v.softmax(1), v.log_softmax(1)))
    }
}

impl Module for SacActor {
    fn parameters(&self) -> Vec<Var> {
        [
            self.fc1.parameters(),
            self.fc2.parameters(),
            self.fc3.parameters(),
        ]
        .concat()
    }
}

pub struct SacCritic {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl SacCritic {
    pub fn new(
        graph: &Graph,
        obs_dim: usize,
        action_dim: usize,
        name: &str,
    ) -> Result<Self, GraphError> {
        let g = graph.with_model_name(name);
        Ok(Self {
            fc1: Linear::new(&g, obs_dim, 128, true, "fc1")?,
            fc2: Linear::new(&g, 128, 128, true, "fc2")?,
            fc3: Linear::new(&g, 128, action_dim, true, "fc3")?,
        })
    }
    pub fn forward(&self, x: impl IntoVar) -> Result<Var, GraphError> {
        let h = self.fc1.forward(x).relu();
        let h = self.fc2.forward(&h).relu();
        Ok(self.fc3.forward(&h))
    }
    pub fn get_q_values(&self, x: &Tensor) -> Result<Tensor, GraphError> {
        Ok(self.forward(x)?.value()?.unwrap())
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

pub struct SacAgent {
    pub actor: SacActor,
    pub critic1: SacCritic,
    pub critic2: SacCritic,
    pub target_critic1: SacCritic,
    pub target_critic2: SacCritic,
    pub log_alpha: f32,
    pub target_entropy: f32,
    pub alpha_lr: f32,
}

impl SacAgent {
    pub fn new(graph: &Graph, obs_dim: usize, action_dim: usize) -> Result<Self, GraphError> {
        Ok(Self {
            actor: SacActor::new(graph, obs_dim, action_dim)?,
            critic1: SacCritic::new(graph, obs_dim, action_dim, "C1")?,
            critic2: SacCritic::new(graph, obs_dim, action_dim, "C2")?,
            target_critic1: SacCritic::new(graph, obs_dim, action_dim, "TC1")?,
            target_critic2: SacCritic::new(graph, obs_dim, action_dim, "TC2")?,
            log_alpha: 0.0,
            target_entropy: 0.5 * (action_dim as f32).ln(),
            alpha_lr: 0.001,
        })
    }
    pub fn alpha(&self) -> f32 {
        self.log_alpha.exp()
    }
    pub fn soft_update_targets(&self) {
        self.target_critic1.soft_update_from(&self.critic1, 0.005);
        self.target_critic2.soft_update_from(&self.critic2, 0.005);
    }
}
