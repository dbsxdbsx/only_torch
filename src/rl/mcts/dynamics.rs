//! MuZero learned dynamics 模型接口
//!
//! `Dynamics` trait 贴近神经网络的 representation + dynamics + prediction 三段式，
//! `DynamicsModel<D>` 适配器将其桥接到 `MctsModel`，补齐动作空间 / 折扣 / 簿记。
//!
//! 库只服务搜索期推理（返回 detached latent）；训练期 K 步 unroll 属于 example。

use super::traits::MctsModel;
use super::types::{ActionPayload, RecurrentOut, RootOut};

/// MuZero learned dynamics 接口
///
/// - `initial_state`：representation h + prediction f
/// - `recurrent`：dynamics g + prediction f
pub trait Dynamics {
    /// obs → (latent, policy_prior, value)
    fn initial_state(&self, obs: &[f32]) -> (Vec<f32>, Vec<f32>, f32);

    /// (latent, action) → (next_latent, reward, policy_prior, value, terminal)
    ///
    /// `terminal`：是否为终止状态。简化版可始终返回 false（靠 reward head 学习终止信号）；
    /// 精确版可训练一个终止头或用 reward 阈值判定。
    fn recurrent(&self, state: &[f32], action: &ActionPayload) -> DynamicsOutput;
}

/// Dynamics::recurrent 的返回值
pub struct DynamicsOutput {
    pub next_state: Vec<f32>,
    pub reward: f32,
    pub prior: Vec<f32>,
    pub value: f32,
    pub terminal: bool,
}

/// 将 `Dynamics` 适配为 `MctsModel`
///
/// 补齐固定动作空间、折扣因子、单智能体簿记（to_play=0）。
/// terminal 由 `Dynamics::recurrent` 返回决定。
pub struct DynamicsModel<D: Dynamics> {
    inner: D,
    actions: Vec<ActionPayload>,
    discount: f32,
}

impl<D: Dynamics> DynamicsModel<D> {
    pub fn new(inner: D, actions: Vec<ActionPayload>, discount: f32) -> Self {
        Self {
            inner,
            actions,
            discount,
        }
    }
}

impl<D: Dynamics> MctsModel for DynamicsModel<D> {
    type State = Vec<f32>;

    fn root(&self, obs: &[f32]) -> RootOut<Self::State> {
        let (latent, prior, value) = self.inner.initial_state(obs);
        RootOut {
            state: latent,
            prior,
            value,
            candidate_actions: self.actions.clone(),
            to_play: 0,
        }
    }

    fn recurrent(&self, state: &Self::State, action: &ActionPayload) -> RecurrentOut<Self::State> {
        let out = self.inner.recurrent(state, action);
        RecurrentOut {
            state: out.next_state,
            reward: out.reward,
            value: out.value,
            prior: out.prior,
            candidate_actions: if out.terminal {
                vec![]
            } else {
                self.actions.clone()
            },
            terminal: out.terminal,
            to_play: 0,
            discount: self.discount,
        }
    }
}
