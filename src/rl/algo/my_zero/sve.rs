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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_and_midpoint() {
        assert!((sve_blend(1.0, 5.0, 0.0) - 1.0).abs() < 1e-6, "w=0 → n-step");
        assert!((sve_blend(1.0, 5.0, 1.0) - 5.0).abs() < 1e-6, "w=1 → 搜索 value");
        assert!((sve_blend(2.0, 4.0, 0.5) - 3.0).abs() < 1e-6, "w=0.5 → 中点");
    }

    #[test]
    fn weight_is_clamped() {
        assert!((sve_blend(1.0, 9.0, 2.0) - 9.0).abs() < 1e-6, "w>1 clamp 到 1");
        assert!((sve_blend(1.0, 9.0, -1.0) - 1.0).abs() < 1e-6, "w<0 clamp 到 0");
    }
}
