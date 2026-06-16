//! Target network 参数同步（hard / EMA + 同步间隔）。
//!
//! Target network 是 EZ 的稳定性增强（base MuZero 不需要）。本模块提供在两份参数列表
//! （online / target）间同步的纯操作；模型结构与两份实例化属环境相关，留示例。
//!
//! 配置见 [`TargetConfig`]：`sync_interval > 0` 走 hard copy（每 interval 步），
//! `== 0` 走 EMA（每步用 `tau`）。

use super::config::TargetConfig;
use crate::nn::Var;
use crate::tensor::Tensor;

/// 硬更新：`target ← online`（逐参数覆盖值）。
pub fn hard_update(online: &[Var], target: &[Var]) {
    for (o, t) in online.iter().zip(target.iter()) {
        if let Ok(Some(ov)) = o.value() {
            let _ = t.set_value(&ov);
        }
    }
}

/// EMA 软更新：`target ← (1-τ)·target + τ·online`（逐元素）。
pub fn ema_update(online: &[Var], target: &[Var], tau: f32) {
    let tau = tau.clamp(0.0, 1.0);
    for (o, t) in online.iter().zip(target.iter()) {
        let (ov, tv) = match (o.value(), t.value()) {
            (Ok(Some(ov)), Ok(Some(tv))) => (ov, tv),
            _ => continue,
        };
        let os = ov.data_as_slice();
        let ts = tv.data_as_slice();
        if os.len() != ts.len() {
            continue;
        }
        let blended: Vec<f32> = os
            .iter()
            .zip(ts.iter())
            .map(|(a, b)| (1.0 - tau) * b + tau * a)
            .collect();
        let _ = t.set_value(&Tensor::new(&blended, ov.shape()));
    }
}

/// 是否在第 `step` 步做 hard 同步（`sync_interval > 0` 且 step 为其正倍数）。
pub fn is_hard_sync_step(step: u32, sync_interval: u32) -> bool {
    sync_interval > 0 && step > 0 && step.is_multiple_of(sync_interval)
}

/// 按 [`TargetConfig`] 在第 `step` 步同步 target：
/// - `enabled == false`：不动；
/// - `sync_interval > 0`：到间隔 hard copy，否则不动；
/// - `sync_interval == 0`：每步 EMA（用 `tau`）。
pub fn sync_target(online: &[Var], target: &[Var], cfg: &TargetConfig, step: u32) {
    if !cfg.enabled {
        return;
    }
    if cfg.sync_interval > 0 {
        if is_hard_sync_step(step, cfg.sync_interval) {
            hard_update(online, target);
        }
    } else {
        ema_update(online, target, cfg.tau);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Graph;

    fn vars(graph: &Graph, data: &[f32]) -> Vec<Var> {
        vec![graph.input(&Tensor::new(data, &[1, data.len()])).unwrap()]
    }

    #[test]
    fn ema_blends_halfway() {
        let graph = Graph::new_with_seed(0);
        let online = vars(&graph, &[1.0, 1.0]);
        let target = vars(&graph, &[0.0, 0.0]);
        ema_update(&online, &target, 0.5);
        let s = target[0].value().unwrap().unwrap();
        let s = s.data_as_slice();
        assert!((s[0] - 0.5).abs() < 1e-6 && (s[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn hard_copies_online() {
        let graph = Graph::new_with_seed(0);
        let online = vars(&graph, &[2.0, 3.0]);
        let target = vars(&graph, &[0.0, 0.0]);
        hard_update(&online, &target);
        let s = target[0].value().unwrap().unwrap();
        let s = s.data_as_slice();
        assert!((s[0] - 2.0).abs() < 1e-6 && (s[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn hard_sync_step_schedule() {
        assert!(!is_hard_sync_step(0, 5));
        assert!(!is_hard_sync_step(3, 5));
        assert!(is_hard_sync_step(5, 5));
        assert!(is_hard_sync_step(10, 5));
        assert!(!is_hard_sync_step(7, 0), "interval=0 永不 hard（走 EMA）");
    }
}
