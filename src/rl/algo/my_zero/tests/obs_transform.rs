//! `obs_transform.rs` symlog 无量纲化测试：数学性质 + 开关行为 + 模型入口接线。

use super::super::network::MyZeroModel;
use super::super::obs_transform::{maybe_symlog, symexp, symlog};
use crate::nn::Graph;
use crate::rl::mcts::Dynamics;
use std::borrow::Cow;

#[test]
fn symlog_basic_properties() {
    // 原点不动
    assert_eq!(symlog(0.0), 0.0);
    // 奇函数
    for &x in &[0.5f32, 1.0, 4.8, 100.0] {
        assert!((symlog(x) + symlog(-x)).abs() < 1e-6);
    }
    // 单调递增
    let xs: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.5).collect();
    for w in xs.windows(2) {
        assert!(symlog(w[1]) > symlog(w[0]));
    }
    // 小值近恒等（|x|<0.5 时偏差 < 20%），大值对数压缩
    assert!((symlog(0.1) - 0.1).abs() < 0.01);
    assert!(symlog(100.0) < 5.0);
}

#[test]
fn symlog_symexp_roundtrip() {
    for &x in &[0.0f32, 0.1, -0.1, 1.0, -4.8, 12.0, -100.0, 333.0] {
        let back = symexp(symlog(x));
        assert!(
            (back - x).abs() < 1e-3 * x.abs().max(1.0),
            "round-trip 失败：x={x} back={back}"
        );
    }
}

#[test]
fn maybe_symlog_off_is_zero_copy_identity() {
    let obs = [0.3f32, -1.2, 4.8, -0.01];
    let out = maybe_symlog(false, &obs);
    assert!(matches!(out, Cow::Borrowed(_)), "关闭时应为零拷贝借用");
    assert_eq!(out.as_ref(), &obs);
}

#[test]
fn maybe_symlog_on_transforms_elementwise() {
    let obs = [0.3f32, -1.2, 4.8];
    let out = maybe_symlog(true, &obs);
    for (o, r) in obs.iter().zip(out.iter()) {
        assert!((symlog(*o) - r).abs() < 1e-7);
    }
}

/// 模型入口接线：同权重下，开关只改变 obs 进入 repr 的数值（root 推理 latent 应不同）；
/// 关闭时与默认构造完全一致。
#[test]
fn model_root_inference_respects_obs_symlog_switch() {
    let obs = vec![0.5f32, -2.0, 3.0, -0.4];

    let graph_a = Graph::new_with_seed(7);
    let model_a = MyZeroModel::new(&graph_a, 4, 2, 32).unwrap();
    let (latent_off, _, _) = (&model_a).initial_state(&obs);

    let graph_b = Graph::new_with_seed(7);
    let model_b = MyZeroModel::new(&graph_b, 4, 2, 32)
        .unwrap()
        .with_obs_symlog(true);
    let (latent_on, _, _) = (&model_b).initial_state(&obs);

    // 同 seed 同权重：off 与默认构造逐 bit 一致由现有测试隐式覆盖；
    // on 与 off 的 latent 必须不同（symlog 确实生效）
    assert_ne!(latent_off, latent_on, "obs_symlog=true 应改变 root latent");

    // on 路径等价于「先手动 symlog 再喂关闭开关的模型」
    let obs_manual: Vec<f32> = obs.iter().map(|&x| symlog(x)).collect();
    let graph_c = Graph::new_with_seed(7);
    let model_c = MyZeroModel::new(&graph_c, 4, 2, 32).unwrap();
    let (latent_manual, _, _) = (&model_c).initial_state(&obs_manual);
    assert_eq!(
        latent_on, latent_manual,
        "开关路径应与手动预变换逐 bit 一致（单点变换、无其他副作用）"
    );
}
