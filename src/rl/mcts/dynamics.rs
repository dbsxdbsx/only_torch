//! MuZero learned dynamics 模型接口
//!
//! `Dynamics` trait 贴近神经网络的 representation + dynamics + prediction 三段式，
//! `DynamicsModel<D>` 适配器将其桥接到 `MctsModel`，补齐动作空间 / 折扣 / 簿记。
//!
//! 库只服务搜索期推理（返回 detached latent）；训练期 K 步 unroll 属于 example。

use rand::SeedableRng;
use rand::rngs::StdRng;
use std::ops::Deref;

use super::traits::{ActionSampleContext, ActionSampler, MctsModel};
use super::types::{ActionId, ActionPayload, CandidateSet, RecurrentOut, RootOut};

/// 搜索树内的 learned world model 接口。
///
/// - `initial_state`：representation h + prediction f
/// - `recurrent`：dynamics g + prediction f
pub trait WorldModel {
    /// 对 MCTS 不透明的 planner state；可包含 latent、recurrent hidden 或 chance code。
    type State: Clone + 'static;

    /// obs → (planner state, policy_prior, value)
    fn initial_state(&self, obs: &[f32]) -> (Self::State, Vec<f32>, f32);

    /// (planner state, action) → 下一 planner state 与预测头。
    fn recurrent(
        &self,
        state: &Self::State,
        action_id: ActionId,
        action: &ActionPayload,
    ) -> WorldModelOutput<Self::State>;
}

/// v0.23 起的历史 learned-latent 接口。
///
/// 旧实现只需实现原有 `recurrent(state, payload)`；新结构化动作实现可覆盖
/// [`recurrent_with_id`](Self::recurrent_with_id) 使用稳定 policy 槽位。
pub trait Dynamics {
    fn initial_state(&self, obs: &[f32]) -> (Vec<f32>, Vec<f32>, f32);

    fn recurrent(&self, state: &[f32], action: &ActionPayload) -> DynamicsOutput;

    fn recurrent_with_id(
        &self,
        state: &[f32],
        _action_id: ActionId,
        action: &ActionPayload,
    ) -> DynamicsOutput {
        self.recurrent(state, action)
    }
}

impl<T: Dynamics> WorldModel for T {
    type State = LatentState;

    fn initial_state(&self, obs: &[f32]) -> (Self::State, Vec<f32>, f32) {
        let (state, prior, value) = Dynamics::initial_state(self, obs);
        (state.into(), prior, value)
    }

    fn recurrent(
        &self,
        state: &Self::State,
        action_id: ActionId,
        action: &ActionPayload,
    ) -> WorldModelOutput<Self::State> {
        let out = Dynamics::recurrent_with_id(self, state, action_id, action);
        WorldModelOutput {
            next_state: out.next_state.into(),
            reward: out.reward,
            prior: out.prior,
            value: out.value,
            terminal: out.terminal,
            continuation: out.continuation,
        }
    }
}

/// MCTS 对 learned latent 的透明包装。
#[repr(transparent)]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LatentState(Vec<f32>);

impl LatentState {
    pub fn new(values: Vec<f32>) -> Self {
        Self(values)
    }

    pub fn into_inner(self) -> Vec<f32> {
        self.0
    }
}

impl Deref for LatentState {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<[f32]> for LatentState {
    fn as_ref(&self) -> &[f32] {
        &self.0
    }
}

impl From<Vec<f32>> for LatentState {
    fn from(value: Vec<f32>) -> Self {
        Self::new(value)
    }
}

/// [`WorldModel::recurrent`] 的通用返回值。
pub struct WorldModelOutput<S> {
    pub next_state: S,
    pub reward: f32,
    pub prior: Vec<f32>,
    pub value: f32,
    /// 是否为终止状态。简化版可始终返回 false；精确版可训练终止头。
    pub terminal: bool,
    /// transition continuation `c_t`，取值建议在 `[0,1]`，语义是「这条 transition 之后是否继续」。
    ///
    /// `DynamicsModel` 把它当**二值终止 gate**用：per-edge discount = `γ·(1−done)`，`done`
    /// 即 `terminal`（由实现按 `c_t ≤ 阈值` 判定），终止边记 0、其余记 γ。**不**做 `γ·c_t`
    /// 的分数式连续衰减——那只在「随机终止」MDP 下才有 aleatoric 语义，且会和 n-step value
    /// target 的二值 continuation 口径不一致（target 内连乘的是观测到的 0/1 continuation）。
    /// 软 `c_t` 仅经 `terminal` 阈值参与硬截断，head 仍训练、为未来随机终止 env 预留。
    pub continuation: f32,
}

/// 历史 `Vec<f32>` dynamics 输出别名。
pub type DynamicsOutput = WorldModelOutput<Vec<f32>>;

/// 将 `Dynamics` 适配为 `MctsModel`
///
/// 补齐固定动作空间、折扣因子、单智能体簿记（to_play=0）。
/// terminal 由 `Dynamics::recurrent` 返回决定。
pub struct DynamicsModel<D: WorldModel> {
    inner: D,
    /// 候选动作集——经 [`ActionSampler`] 接缝统一产出（不再是外部直接塞入的裸字段）。
    actions: Vec<ActionPayload>,
    discount: f32,
}

impl<D: WorldModel> DynamicsModel<D> {
    /// 用固定动作 catalog 构造。
    pub fn new(inner: D, actions: Vec<ActionPayload>, discount: f32) -> Self {
        Self {
            inner,
            actions,
            discount,
        }
    }
}

impl<D: WorldModel<State = LatentState>> DynamicsModel<D> {
    /// 用任意 [`ActionSampler`] 作为候选来源构造（接缝统一入口）。
    ///
    /// 离散候选与 state / rng 无关，构造期经 sampler 产出一次并缓存，搜索期零开销复用；
    /// 连续 / 混合的 per-state 动态采样将在引入 `GumbelPolicy` 时扩展
    /// adapter 的动态路径（届时 `recurrent` 按 state 调 sampler），此处先打通离散统一来源。
    pub fn new_with_sampler<A: ActionSampler<Vec<f32>>>(
        inner: D,
        sampler: &A,
        discount: f32,
    ) -> Self {
        let empty = Vec::new();
        let ctx = ActionSampleContext {
            state: &empty,
            depth: 0,
            to_play: 0,
            num_candidates: 0,
        };
        // 离散采样忽略 rng；占位固定种子以满足签名且保持确定性。
        let mut rng = StdRng::seed_from_u64(0);
        let candidates = sampler.sample(ctx, &mut rng);
        Self {
            inner,
            actions: candidates.actions,
            discount,
        }
    }
}

impl<D: WorldModel> MctsModel for DynamicsModel<D> {
    type State = D::State;

    fn root(&self, obs: &[f32]) -> RootOut<Self::State> {
        let (latent, prior, value) = self.inner.initial_state(obs);
        RootOut {
            state: latent,
            value,
            candidates: CandidateSet::from_actions_and_priors_strict(self.actions.clone(), prior),
            to_play: 0,
        }
    }

    fn recurrent(
        &self,
        state: &Self::State,
        action_id: ActionId,
        action: &ActionPayload,
    ) -> RecurrentOut<Self::State> {
        let out = self.inner.recurrent(state, action_id, action);
        RecurrentOut {
            state: out.next_state,
            reward: out.reward,
            value: out.value,
            candidates: if out.terminal {
                CandidateSet::empty()
            } else {
                CandidateSet::from_actions_and_priors_strict(self.actions.clone(), out.prior)
            },
            terminal: out.terminal,
            to_play: 0,
            // per-edge discount = γ·(1−done)：终止边 0、其余 γ（canonical MuZero）。
            // 与 n-step value target 的二值 continuation 口径一致；软 c_t 只经 out.terminal
            // 阈值参与硬截断，不连续衰减健康边的 value（避免 head 未校准时系统性压低好状态）。
            discount: if out.terminal { 0.0 } else { self.discount },
        }
    }
}
