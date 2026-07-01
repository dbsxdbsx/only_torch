//! batch-native 训练路径的等价性守门测试。
//!
//! 目的：证明 [`MyZeroModel::train_unroll_batch`]（batch 化）在 **G=1** 时是逐样本
//! [`MyZeroModel::train_unroll`] 的正确推广，把「良性浮点漂移」与「实现 bug」区分开：
//! - [`bitexact_g1_without_consistency`]：无 consistency 时两路径 **forward 逐 bit 一致**
//!   （CE/MSE 在 [1,X] 上算子、形状完全相同）——任何差异都是 bug。
//! - [`grad_equivalence_g1_full_stack`]：含 consistency+reconstruction 的完整 CartPole 栈，
//!   **forward 与逐参数梯度**在极紧容差内一致（consistency 的 `sum` vs `sum_axis` 归约顺序
//!   可能带 ULP 级良性漂移，故用容差而非逐 bit）。梯度一致 = 训练动力学等价。

use crate::nn::{Graph, Var};
use crate::rl::algo::my_zero::network::{MyZeroModel, UnrollItem};

const OBS_DIM: usize = 4;
const ACTION_DIM: usize = 2;
const LATENT_DIM: usize = 64;

/// 构造一条 K=3 步的合成样本（obs/action/各步目标/next_obs 全部给足）。
#[allow(clippy::type_complexity)]
fn sample() -> (
    Vec<f32>,      // obs_t
    Vec<usize>,    // actions (K)
    Vec<Vec<f32>>, // target_policies (K+1)
    Vec<f32>,      // target_values (K+1)
    Vec<f32>,      // target_rewards (K)
    Vec<f32>,      // target_continuations (K)
    Vec<Vec<f32>>, // next_obs (K)
) {
    let obs_t = vec![0.10, -0.20, 0.05, 0.30];
    let actions = vec![0usize, 1, 0];
    let target_policies = vec![
        vec![0.70, 0.30],
        vec![0.40, 0.60],
        vec![0.55, 0.45],
        vec![0.50, 0.50],
    ];
    let target_values = vec![1.0, 2.0, 3.0, 4.0];
    let target_rewards = vec![1.0, 1.0, 1.0];
    let target_continuations = vec![1.0, 1.0, 0.0];
    let next_obs = vec![
        vec![0.11, -0.19, 0.06, 0.31],
        vec![0.12, -0.18, 0.07, 0.32],
        vec![0.13, -0.17, 0.08, 0.33],
    ];
    (
        obs_t,
        actions,
        target_policies,
        target_values,
        target_rewards,
        target_continuations,
        next_obs,
    )
}

fn to_item(
    obs_t: &[f32],
    actions: &[usize],
    tps: &[Vec<f32>],
    tvs: &[f32],
    trs: &[f32],
    tcs: &[f32],
    next_obs: &[Vec<f32>],
) -> UnrollItem {
    UnrollItem {
        obs_t: obs_t.to_vec(),
        actions: actions.to_vec(),
        target_policies: tps.to_vec(),
        target_values: tvs.to_vec(),
        target_rewards: trs.to_vec(),
        target_continuations: tcs.to_vec(),
        next_obs: next_obs.to_vec(),
    }
}

/// 逐参数快照当前梯度（None → 空 vec，表示该参数不在本次 loss 子图内）。
fn grad_snapshot(params: &[Var]) -> Vec<Vec<f32>> {
    params
        .iter()
        .map(|p| {
            p.grad()
                .unwrap()
                .map(|t| t.data_as_slice().to_vec())
                .unwrap_or_default()
        })
        .collect()
}

#[test]
fn bitexact_g1_without_consistency() {
    let graph = Graph::new_with_seed(0);
    let model = MyZeroModel::new(&graph, OBS_DIM, ACTION_DIM, LATENT_DIM).unwrap();
    let (obs, actions, tps, tvs, trs, tcs, next_obs) = sample();

    // consistency=0，reconstruction=1.0，value_prefix=false
    let (cons, recon, vp) = (0.0f32, 1.0f32, false);

    let ref_loss = model
        .train_unroll(
            &obs,
            &actions,
            &tps,
            &tvs,
            &trs,
            &tcs,
            Some(&next_obs),
            cons,
            recon,
            vp,
        )
        .unwrap();
    let ref_val = ref_loss.value().unwrap().unwrap().data_as_slice()[0];

    let item = to_item(&obs, &actions, &tps, &tvs, &trs, &tcs, &next_obs);
    let batch_loss = model.train_unroll_batch(&[item], cons, recon, vp).unwrap();
    let batch_val = batch_loss.value().unwrap().unwrap().data_as_slice()[0];

    assert_eq!(
        ref_val, batch_val,
        "G=1 无 consistency 时 batched forward 应与逐样本逐 bit 一致：\
         ref={ref_val} vs batch={batch_val}"
    );
}

#[test]
fn grad_equivalence_g1_full_stack() {
    let graph = Graph::new_with_seed(0);
    let model = MyZeroModel::new(&graph, OBS_DIM, ACTION_DIM, LATENT_DIM).unwrap();
    let params = model.parameters();
    let (obs, actions, tps, tvs, trs, tcs, next_obs) = sample();

    // 完整 CartPole 栈：consistency=2.0，reconstruction=1.0，value_prefix=false
    let (cons, recon, vp) = (2.0f32, 1.0f32, false);

    // 逐样本参考（forward + backward，快照梯度）
    graph.zero_grad().unwrap();
    let ref_loss = model
        .train_unroll(
            &obs,
            &actions,
            &tps,
            &tvs,
            &trs,
            &tcs,
            Some(&next_obs),
            cons,
            recon,
            vp,
        )
        .unwrap();
    let ref_val = ref_loss.value().unwrap().unwrap().data_as_slice()[0];
    ref_loss.backward().unwrap();
    let ref_grads = grad_snapshot(&params);

    // batch 路径（G=1，forward + backward）
    graph.zero_grad().unwrap();
    let item = to_item(&obs, &actions, &tps, &tvs, &trs, &tcs, &next_obs);
    let batch_loss = model.train_unroll_batch(&[item], cons, recon, vp).unwrap();
    let batch_val = batch_loss.value().unwrap().unwrap().data_as_slice()[0];
    batch_loss.backward().unwrap();
    let batch_grads = grad_snapshot(&params);

    // forward 一致（容差：consistency 归约顺序带 ULP 级良性漂移）
    assert!(
        (ref_val - batch_val).abs() <= 1e-5 * ref_val.abs().max(1.0),
        "G=1 完整栈 forward 应等价：ref={ref_val} vs batch={batch_val}"
    );

    // 逐参数梯度一致（训练动力学等价的决定性证据）
    assert_eq!(ref_grads.len(), batch_grads.len(), "参数数量应一致");
    let mut max_abs_diff = 0.0f32;
    for (pi, (rg, bg)) in ref_grads.iter().zip(&batch_grads).enumerate() {
        assert_eq!(rg.len(), bg.len(), "参数 {pi} 梯度长度应一致");
        for (a, b) in rg.iter().zip(bg) {
            max_abs_diff = max_abs_diff.max((a - b).abs());
        }
    }
    assert!(
        max_abs_diff <= 1e-4,
        "G=1 完整栈逐参数梯度应等价（max_abs_diff={max_abs_diff}）→ 若超差说明 batched 实现有逻辑 bug"
    );
    println!(
        "[batch 等价] G=1 完整栈：forward ref={ref_val:.6} batch={batch_val:.6}，\
         梯度 max_abs_diff={max_abs_diff:.2e}"
    );
}

/// 第二条样本（与 [`sample`] 结构相同、数值不同），用于 G=2 等价性验证。
#[allow(clippy::type_complexity)]
fn sample2() -> (
    Vec<f32>,
    Vec<usize>,
    Vec<Vec<f32>>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<Vec<f32>>,
) {
    let obs_t = vec![-0.30, 0.25, -0.10, -0.05];
    let actions = vec![1usize, 0, 1];
    let target_policies = vec![
        vec![0.20, 0.80],
        vec![0.65, 0.35],
        vec![0.45, 0.55],
        vec![0.50, 0.50],
    ];
    let target_values = vec![-2.0, 0.5, 1.5, 3.0];
    let target_rewards = vec![1.0, 0.0, 1.0];
    let target_continuations = vec![1.0, 0.0, 0.0];
    let next_obs = vec![
        vec![-0.29, 0.26, -0.11, -0.06],
        vec![-0.28, 0.27, -0.12, -0.07],
        vec![-0.27, 0.28, -0.13, -0.08],
    ];
    (
        obs_t,
        actions,
        target_policies,
        target_values,
        target_rewards,
        target_continuations,
        next_obs,
    )
}

/// G=2 梯度等价性核心：返回「逐样本累积」与「batched」的逐参数 max_abs_diff + worst idx。
fn g2_grad_max_diff(cons: f32, recon: f32) -> (f32, usize) {
    let graph = Graph::new_with_seed(0);
    let model = MyZeroModel::new(&graph, OBS_DIM, ACTION_DIM, LATENT_DIM).unwrap();
    let params = model.parameters();

    let s1 = sample();
    let s2 = sample2();
    let batch_size = 2.0f32;
    let vp = false;

    graph.zero_grad().unwrap();
    for s in [&s1, &s2] {
        let (obs, actions, tps, tvs, trs, tcs, next_obs) = s;
        let loss = model
            .train_unroll(
                obs,
                actions,
                tps,
                tvs,
                trs,
                tcs,
                Some(next_obs),
                cons,
                recon,
                vp,
            )
            .unwrap()
            * (1.0 / batch_size);
        loss.backward().unwrap();
    }
    let ref_grads = grad_snapshot(&params);

    graph.zero_grad().unwrap();
    let items = vec![
        to_item(&s1.0, &s1.1, &s1.2, &s1.3, &s1.4, &s1.5, &s1.6),
        to_item(&s2.0, &s2.1, &s2.2, &s2.3, &s2.4, &s2.5, &s2.6),
    ];
    let g = items.len() as f32;
    let loss = model.train_unroll_batch(&items, cons, recon, vp).unwrap() * (g / batch_size);
    loss.backward().unwrap();
    let batch_grads = grad_snapshot(&params);

    let mut worst = (0usize, 0.0f32);
    for (i, (rg, bg)) in ref_grads.iter().zip(&batch_grads).enumerate() {
        for (a, b) in rg.iter().zip(bg) {
            let d = (a - b).abs();
            if d > worst.1 {
                worst = (i, d);
            }
        }
    }
    (worst.1, worst.0)
}

/// **决定性测试**：G=2 完整栈的 batched backward 与逐样本累积等价（G=1 补不上的缩放验证）。
#[test]
fn grad_equivalence_g2_matches_per_sample_accumulation() {
    let (diff, idx) = g2_grad_max_diff(2.0, 1.0);
    println!("[batch 等价] G=2 完整栈：max_abs_diff={diff:.3e} worst idx={idx}");
    assert!(
        diff <= 1e-4,
        "G=2 完整栈 batched 梯度与逐样本累积不一致（diff={diff} idx={idx}）→ G>1 缩放/归约有 bug"
    );
}
