//! 阶段 ②：随机轨迹上的算术谓词（不需要 MyZero 网）。
//!
//! 只填 [`super::route::EnvStructure`] 里 Gym shape 看不见的尺度 hint。
//! 不探测 POMDP / 随机性 / HL-Gauss / `!` 组件。

use super::route::EnvStructure;
use super::schema::ObservationSchema;

/// 视为 0 的绝对值下限，避免 `1e-20` 把比值撑爆。
pub(crate) const OBS_SCALE_ABS_FLOOR: f32 = 1e-8;

/// 任一 Flat 维 `max_abs / min_abs ≥` 此值视为跨数量级（两个数量级）。
pub(crate) const OBS_SCALE_ORDER_RATIO: f32 = 100.0;

/// `mean(|r|)` 超过此值才建议把 reward 压到 [`REWARD_SCALE_COMPRESSED`]。
pub(crate) const REWARD_SCALE_DOWN_MEAN_ABS: f32 = 5.0;

pub(crate) const REWARD_SCALE_DEFAULT: f32 = 1.0;
pub(crate) const REWARD_SCALE_COMPRESSED: f32 = 0.1;

/// 短探统计。`reward_scale` 是适配建议，不是 [`super::component::Components`] 开关。
#[cfg_attr(not(test), expect(dead_code))]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ProbeStats {
    pub obs_scale_spans_orders: bool,
    pub reward_scale: f32,
}

/// 对观测行与逐步奖励做算术。空轨迹或非有限值按「测不到 → 保守关」处理。
// 生产路径仍走 env_id recipe；本函数目前只给路由单测用。
#[cfg_attr(not(test), expect(dead_code))]
pub(crate) fn probe_trajectories(
    observation: ObservationSchema,
    observations: &[Vec<f32>],
    rewards: &[f32],
) -> ProbeStats {
    ProbeStats {
        obs_scale_spans_orders: matches!(observation, ObservationSchema::Flat(_))
            && flat_obs_spans_orders(observations),
        reward_scale: suggested_reward_scale(rewards),
    }
}

impl EnvStructure {
    /// 只写入尺度 hint；不改 POMDP / value 噪声声明。
    // 生产路径仍走 env_id recipe；本方法目前只给路由单测用。
    #[cfg_attr(not(test), expect(dead_code))]
    pub(crate) fn apply_probe(&mut self, stats: &ProbeStats) {
        self.obs_scale_spans_orders = stats.obs_scale_spans_orders;
    }
}

fn flat_obs_spans_orders(observations: &[Vec<f32>]) -> bool {
    let Some(dim) = observations.first().map(|row| row.len()) else {
        return false;
    };
    if dim == 0 {
        return false;
    }
    for d in 0..dim {
        let mut min_abs = f32::INFINITY;
        let mut max_abs: f32 = 0.0;
        for row in observations {
            let Some(&v) = row.get(d) else {
                continue;
            };
            if !v.is_finite() {
                continue;
            }
            let a = v.abs();
            if a <= OBS_SCALE_ABS_FLOOR {
                continue;
            }
            min_abs = min_abs.min(a);
            max_abs = max_abs.max(a);
        }
        if min_abs.is_finite() && max_abs / min_abs >= OBS_SCALE_ORDER_RATIO {
            return true;
        }
    }
    false
}

fn suggested_reward_scale(rewards: &[f32]) -> f32 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for &r in rewards {
        if r.is_finite() {
            sum += r.abs();
            n += 1;
        }
    }
    if n == 0 {
        return REWARD_SCALE_DEFAULT;
    }
    if sum / n as f32 > REWARD_SCALE_DOWN_MEAN_ABS {
        REWARD_SCALE_COMPRESSED
    } else {
        REWARD_SCALE_DEFAULT
    }
}
