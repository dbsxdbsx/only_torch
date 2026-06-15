//! MuZero CartPole 模型定义
//!
//! 三网络架构（简化版，适合 CartPole-v0）：
//! - Representation h: obs(4) → latent(32)
//! - Dynamics g: (latent(32), action_onehot(2)) → (next_latent(32), reward(1))
//! - Prediction f: latent(32) → (policy_logits(2), value(1))

use only_torch::nn::{
    Graph, GraphError, IntoVar, Linear, Module, Var, VarActivationOps, VarLossOps,
};
use only_torch::rl::mcts::{ActionPayload, Dynamics, DynamicsOutput};
use only_torch::tensor::Tensor;

// ============================================================================
// Representation 网络 h: obs → latent
// ============================================================================

pub struct RepresentationNet {
    fc1: Linear,
    fc2: Linear,
}

impl RepresentationNet {
    pub fn new(graph: &Graph, obs_dim: usize, latent_dim: usize) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("Repr");
        Ok(Self {
            fc1: Linear::new(&graph, obs_dim, 64, true, "fc1")?,
            fc2: Linear::new(&graph, 64, latent_dim, true, "fc2")?,
        })
    }

    pub fn forward(&self, x: impl IntoVar) -> Var {
        let h = self.fc1.forward(x).relu();
        self.fc2.forward(&h).relu()
    }
}

impl Module for RepresentationNet {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}

// ============================================================================
// Dynamics 网络 g: (latent, action_onehot) → (next_latent, reward)
// ============================================================================

pub struct DynamicsNet {
    fc1: Linear,
    fc_latent: Linear,
    fc_reward: Linear,
}

impl DynamicsNet {
    pub fn new(
        graph: &Graph,
        latent_dim: usize,
        action_dim: usize,
    ) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("Dyn");
        let input_dim = latent_dim + action_dim;
        Ok(Self {
            fc1: Linear::new(&graph, input_dim, 64, true, "fc1")?,
            fc_latent: Linear::new(&graph, 64, latent_dim, true, "fc_latent")?,
            fc_reward: Linear::new(&graph, 64, 1, true, "fc_reward")?,
        })
    }

    /// 返回 (next_latent_var, reward_var)
    pub fn forward(&self, latent: &Var, action_onehot: &Var) -> Result<(Var, Var), GraphError> {
        let input = Var::concat(&[latent, action_onehot], 1)?;
        let h = self.fc1.forward(&input).relu();
        let next_latent = self.fc_latent.forward(&h).relu();
        let reward = self.fc_reward.forward(&h);
        Ok((next_latent, reward))
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
// Prediction 网络 f: latent → (policy_logits, value)
// ============================================================================

pub struct PredictionNet {
    fc1: Linear,
    fc_policy: Linear,
    fc_value: Linear,
}

impl PredictionNet {
    pub fn new(
        graph: &Graph,
        latent_dim: usize,
        action_dim: usize,
    ) -> Result<Self, GraphError> {
        let graph = graph.with_model_name("Pred");
        Ok(Self {
            fc1: Linear::new(&graph, latent_dim, 64, true, "fc1")?,
            fc_policy: Linear::new(&graph, 64, action_dim, true, "fc_policy")?,
            fc_value: Linear::new(&graph, 64, 1, true, "fc_value")?,
        })
    }

    /// 返回 (policy_logits_var, value_var)
    pub fn forward(&self, latent: &Var) -> (Var, Var) {
        let h = self.fc1.forward(latent).relu();
        let policy = self.fc_policy.forward(&h);
        let value = self.fc_value.forward(&h);
        (policy, value)
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

    /// 将离散动作编码为 one-hot 向量
    fn action_to_onehot(&self, action_idx: usize) -> Vec<f32> {
        let mut oh = vec![0.0; self.action_dim];
        if action_idx < self.action_dim {
            oh[action_idx] = 1.0;
        }
        oh
    }

    // ========================================================================
    // 训练用：K 步 unroll（走计算图，可反传）
    // ========================================================================

    /// K 步 unroll 训练，返回总损失 Var
    ///
    /// `targets`: 从 SelfPlayGame 中提取的连续 K+1 步数据
    /// - `obs_t`: 起始观测
    /// - `actions[0..K]`: 实际执行的动作索引
    /// - `target_policies[0..K+1]`: MCTS 输出的策略目标（含 t 位置）
    /// - `target_values[0..K+1]`: n-step value target（含 t 位置）
    /// - `target_rewards[0..K]`: 实际即时奖励
    pub fn train_unroll(
        &self,
        obs_t: &[f32],
        actions: &[usize],
        target_policies: &[Vec<f32>],
        target_values: &[f32],
        target_rewards: &[f32],
    ) -> Result<Var, GraphError> {
        let k = actions.len();

        // representation: obs → latent_0
        let obs_tensor = Tensor::new(obs_t, &[1, obs_t.len()]);
        let mut latent = self.repr.forward(&obs_tensor);

        // MuZero loss 系数：防止 value/reward MSE 压死 policy CE
        let value_coef = 0.25;
        let reward_coef = 0.25;

        // 初始预测（t 位置的 policy/value 损失）
        let (pred_policy, pred_value) = self.pred.forward(&latent);
        let target_p0 = Tensor::new(&target_policies[0], &[1, self.action_dim]);
        let target_v0 = Tensor::new(&[target_values[0]], &[1, 1]);
        let mut total_loss = pred_policy.cross_entropy(&target_p0)?;
        total_loss = &total_loss + &(&pred_value.mse_loss(&target_v0)? * value_coef);

        // K 步展开
        for i in 0..k {
            let oh = self.action_to_onehot(actions[i]);
            let oh_tensor = Tensor::new(&oh, &[1, self.action_dim]);
            let oh_var = self.graph.input(&oh_tensor)?;

            let (next_latent, pred_reward) = self.dyn_net.forward(&latent, &oh_var)?;

            let (pred_p, pred_v) = self.pred.forward(&next_latent);

            let tp = Tensor::new(&target_policies[i + 1], &[1, self.action_dim]);
            let tv = Tensor::new(&[target_values[i + 1]], &[1, 1]);
            let tr = Tensor::new(&[target_rewards[i]], &[1, 1]);

            total_loss = &total_loss + &pred_p.cross_entropy(&tp)?;
            total_loss = &total_loss + &(&pred_v.mse_loss(&tv)? * value_coef);
            total_loss = &total_loss + &(&pred_reward.mse_loss(&tr)? * reward_coef);

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
        let latent_var = self.repr.forward(&obs_tensor);
        let latent_tensor = latent_var.value().unwrap().unwrap();

        let (policy_var, value_var) = self.pred.forward(&latent_var);
        let policy_tensor = policy_var.value().unwrap().unwrap();
        let value_tensor = value_var.value().unwrap().unwrap();

        let policy_probs = policy_tensor.softmax(1);

        let latent_vec = latent_tensor.data_as_slice().to_vec();
        let policy_vec = policy_probs.data_as_slice().to_vec();
        let value = value_tensor.data_as_slice()[0];

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

        let (next_latent_var, reward_var) = self.dyn_net.forward(&latent_var, &oh_var).unwrap();
        let next_latent_tensor = next_latent_var.value().unwrap().unwrap();
        let reward_tensor = reward_var.value().unwrap().unwrap();

        let (policy_var, value_var) = self.pred.forward(&next_latent_var);
        let policy_tensor = policy_var.value().unwrap().unwrap();
        let value_tensor = value_var.value().unwrap().unwrap();

        let policy_probs = policy_tensor.softmax(1);

        DynamicsOutput {
            next_state: next_latent_tensor.data_as_slice().to_vec(),
            reward: reward_tensor.data_as_slice()[0],
            prior: policy_probs.data_as_slice().to_vec(),
            value: value_tensor.data_as_slice()[0],
            terminal: false, // 简化版不学终止头，靠 reward head 学习
        }
    }
}
