//! MyZero target network 参数同步（hard / EMA + 同步间隔）。
//!
//! Target network 是稳定性增强（base 不需要）。本模块提供在两份参数列表（online / target）
//! 间同步的纯操作；模型结构与两份实例化属网络结构，留 [`super::network`]。
//!
//! 配置见 [`TargetConfig`]：`sync_interval > 0` 走 hard copy（每 interval 步），
//! `== 0` 走 EMA（每步用 `tau`）。
//!
//! 注：目前作为可用组件入库（消融开关 `Components::target_net`），尚未接入训练循环——
//! 接线属后续消融工作。

use crate::nn::Var;
use crate::tensor::Tensor;

/// Target network 配置。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetConfig {
    /// 是否启用 target net（消融开关）。
    pub enabled: bool,
    /// EMA 软更新系数 τ（`sync_interval == 0` 时生效）。
    pub tau: f32,
    /// hard update 间隔（步）：`> 0` 走 hard copy；`== 0` 走 EMA（用 `tau`）。
    pub sync_interval: u32,
}

impl Default for TargetConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tau: 0.01,
            sync_interval: 0,
        }
    }
}

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
        // to_vec 按逻辑行主序展开、布局无关：即便参数被 set_value 塞入非连续视图也不会 panic
        let os = ov.to_vec();
        let ts = tv.to_vec();
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

/// 按 [`TargetConfig`] 在第 `step` 步同步 target。
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

    /// **回归测试**：online 参数被塞入**非连续**视图（`permute` 产物）时，
    /// `ema_update` 必须按逻辑行主序混合、不得 panic（此前 `data_as_slice()` 会 panic）。
    #[test]
    fn ema_update_noncontiguous_online_no_panic() {
        let graph = Graph::new_with_seed(0);
        // base [2,2]=[1,2,3,4] → permute[1,0] → 非连续，逻辑行主序为 [1,3,2,4]
        let base = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let nc = base.permute(&[1, 0]);
        assert!(!nc.is_contiguous(), "permute 结果应为非连续");

        let online = vec![graph.input(&Tensor::zeros(&[2, 2])).unwrap()];
        online[0].set_value(&nc).unwrap();
        let target = vec![graph.input(&Tensor::zeros(&[2, 2])).unwrap()];

        ema_update(&online, &target, 0.5);

        // target = 0.5·online_logical + 0.5·0 = [0.5, 1.5, 1.0, 2.0]（逻辑行主序）
        let out = target[0].value().unwrap().unwrap().to_vec();
        let expected = [0.5, 1.5, 1.0, 2.0];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "EMA 应按逻辑序混合：{out:?} vs {expected:?}"
            );
        }
    }
}
