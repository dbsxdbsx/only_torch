//! MyZero SVE：Search-based Value Estimation（Ye et al. 2021）。
//!
//! 用 MCTS 搜索本身产出的（更可靠的）root value 修正 stale buffer 的 n-step bootstrap value
//! 目标，缓解旧数据的 value 漂移。与 reanalyze 协同。

/// 把搜索 root value blend 进 n-step bootstrap 目标。
///
/// `blend = (1 - w)·n_step_target + w·search_root_value`：
/// - `w == 0` → 纯 n-step（= base 行为）；
/// - `w == 1` → 纯搜索 value；
/// - `w` 越大越信任搜索（适合 stale 数据）。`w` 自动 clamp 到 `[0,1]`。
pub fn sve_blend(n_step_target: f32, search_root_value: f32, weight: f32) -> f32 {
    let w = weight.clamp(0.0, 1.0);
    (1.0 - w) * n_step_target + w * search_root_value
}
