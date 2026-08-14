//! 按观测 / 动作结构推断默认组件（用户不可见）。
//!
//! 只编码能写成谓词的规则；写不出谓词的旋钮保持关。
//! 量纲不在这里估：value/reward 永远 `symlog`（见 [`super::value_transform`]）；
//! 连续观测永远 `symlog`（见 [`ObservationSchema::uses_obs_symlog`]），token ID 除外。
//! [`super::recipe`] 仍按 `env_id` 注入已 promote 栈；观测量纲由 runner 按 schema 覆盖。

use super::component::Components;
use super::sampled_params::AUTO_SAMPLED_MIN_JOINT_ACTIONS;
use super::schema::{ActionSchema, ObservationSchema};

/// 低维向量才开 reconstruction：obs 比特少、且通常任务相关。
/// 超过此维数视为「可能含大量无关坐标」，保持关。
// 生产路径仍走 env_id recipe；本常量目前只给结构路由与其单测用。
#[cfg_attr(not(test), expect(dead_code))]
pub(crate) const FLAT_RECON_MAX_DIM: usize = 16;

/// 结构路由能看见的环境特征。后两个不能从 Gym shape 读出，须由适配层声明。
// 生产路径仍走 env_id recipe；本类型目前只给结构路由与其单测用。
#[cfg_attr(not(test), expect(dead_code))]
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
    // 生产路径仍走 env_id recipe；构造器目前只给结构路由单测用。
    #[cfg_attr(not(test), expect(dead_code))]
    pub(crate) fn new(observation: ObservationSchema, action: ActionSchema) -> Self {
        Self {
            observation,
            action,
            partially_observable: false,
            value_support_noisy: false,
        }
    }
}

/// 由结构特征推出默认 [`Components`]。
///
/// 不含 KL 自适应 lr / PER / 主动数据：那些是训练过程旋钮，不是环境结构。
// 生产路径仍走 env_id recipe；函数目前只给结构路由单测用。
#[cfg_attr(not(test), expect(dead_code))]
pub(crate) fn route_components(env: &EnvStructure) -> Components {
    let mut c = Components::base();
    c.consistency = obs_is_spatial(env.observation);
    c.reconstruction = obs_is_low_dim_flat(env.observation);
    c.sampled = action_needs_sampled(&env.action);
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

fn action_needs_sampled(action: &ActionSchema) -> bool {
    match action {
        ActionSchema::ContinuousBins { .. } | ActionSchema::HybridBins { .. } => true,
        ActionSchema::Discrete { .. } | ActionSchema::MultiDiscrete { .. } => {
            action.action_count() >= AUTO_SAMPLED_MIN_JOINT_ACTIONS
        }
    }
}
