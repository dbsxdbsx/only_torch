//! 按观测 / 动作结构推断默认组件（用户不可见）。
//!
//! 只编码能写成谓词的规则；写不出谓词的旋钮保持关。
//! 量纲不在这里估：value/reward 永远 `symlog`（见 [`super::value_transform`]）；
//! 连续观测永远 `symlog`（见 [`ObservationSchema::uses_obs_symlog`]），token ID 除外。
//! CartPole 生产路径已走本模块；其余 `env_id` 仍由 [`super::recipe`] 注入。
//! 观测量纲由 runner 按 schema 覆盖。

use super::component::Components;
use super::sampled_params::sampled_would_subsample;
use super::schema::{ActionSchema, ObservationSchema};

/// 低维向量才开 reconstruction：obs 比特少、且通常任务相关。
/// 超过此维数视为「可能含大量无关坐标」，保持关。
pub(crate) const FLAT_RECON_MAX_DIM: usize = 16;

/// 结构路由能看见的环境特征。后两个不能从 Gym shape 读出，须由适配层声明。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EnvStructure {
    pub observation: ObservationSchema,
    pub action: ActionSchema,
    /// 当前观测不足以构成马尔可夫状态。
    pub partially_observable: bool,
    /// value / 回报支撑宽且噪（HL-Gauss 的适应症）。
    pub value_support_noisy: bool,
}

impl EnvStructure {
    pub(crate) fn new(observation: ObservationSchema, action: ActionSchema) -> Self {
        Self {
            observation,
            action,
            partially_observable: false,
            value_support_noisy: false,
        }
    }
}

/// CartPole-v1 的 Gym 结构（`Box(4,)` + 离散 2）。`new("CartPole-v1")` 用它走路由。
pub(crate) fn cartpole_structure() -> EnvStructure {
    EnvStructure::new(ObservationSchema::Flat(4), ActionSchema::discrete(2))
}

/// 由结构特征推出默认 [`Components`]。
///
/// `num_simulations` 只给 Sampled 用：开/关 = `K_eff < N`，读已有搜索预算，不负责选定 sims。
/// 不含 KL 自适应 lr / PER / 主动数据：那些是训练过程旋钮，不是环境结构。
pub(crate) fn route_components(env: &EnvStructure, num_simulations: u32) -> Components {
    let mut c = Components::base();
    c.consistency = obs_is_spatial(env.observation);
    c.reconstruction = obs_is_low_dim_flat(env.observation);
    c.sampled = sampled_would_subsample(env.action.action_count(), num_simulations);
    // Gumbel 与 Sampled 根语义暂不兼容；连续动作走 Sampled 离散化，不在这里开 Gumbel。
    c.gumbel = false;
    c.recurrent_posterior = env.partially_observable;
    c.obs_symlog = env.observation.uses_obs_symlog();
    c.hl_gauss = env.value_support_noisy;
    c
}

fn obs_is_spatial(obs: ObservationSchema) -> bool {
    matches!(
        obs,
        ObservationSchema::Image { .. }
            | ObservationSchema::ImageRect { .. }
            | ObservationSchema::ImageDense { .. }
            | ObservationSchema::Board { .. }
    )
}

fn obs_is_low_dim_flat(obs: ObservationSchema) -> bool {
    matches!(obs, ObservationSchema::Flat(dim) if dim > 0 && dim <= FLAT_RECON_MAX_DIM)
}
