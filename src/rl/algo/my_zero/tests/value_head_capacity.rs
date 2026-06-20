//! value-head 容量诊断（Pendulum value 坍缩 issue 的决定性隔离实验）
//!
//! 背景：`pendulum_failure_diagnosis.md` 锁定根因为「value head 坍缩成常数」
//! （预测 std 26 vs 真实 MC return std 175）。开放分叉：
//! - (a) **value target 本身坍缩**（td_steps/gamma 使目标近常数）；
//! - (b) **head 学不动**（网络/loss/lr 表达力不足）。
//!
//! 本测试隔离 (b)：**喂已知、高方差、obs 可分的 value 目标**，只训练 repr+pred 的
//! value head（k=0 unroll，不经 dynamics），看它能否把「高价值 obs」与「低价值 obs」
//! 的预测**拉开**。
//! - 若能拉开 → head 有容量，坍缩来自上游（target/搜索）→ 排除 (b)。
//! - 若拉不开 → head/loss/repr 本身表达力不足 → 坐实 (b)。

use crate::nn::{Adam, Graph, Optimizer};
use crate::rl::algo::my_zero::network::MyZeroModel;
use crate::rl::mcts::Dynamics;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// 训练 value head 拟合「obs[0] 符号 → 高/低 value」的可分回归任务，断言不坍缩。
#[test]
fn value_head_can_separate_high_low_targets() {
    let obs_dim = 3;
    let action_dim = 9; // Pendulum 同款离散档数
    let latent_dim = 64;

    let graph = Graph::new_with_seed(0);
    let model = MyZeroModel::new(&graph, obs_dim, action_dim, latent_dim).unwrap();
    let mut opt = Adam::new(&graph, &model.parameters(), 0.02);

    // 合成数据：obs[0] > 0 → 高价值目标；obs[0] < 0 → 低价值目标（scaled 空间，落在 support 内）。
    // 其余特征随机，保证 obs 非常量（min-max 归一化不退化）。
    const N: usize = 16;
    const TARGET_HIGH: f32 = -2.0;
    const TARGET_LOW: f32 = -16.0;
    let uniform = vec![1.0 / action_dim as f32; action_dim];

    let mut rng = StdRng::seed_from_u64(7);
    let mut obses: Vec<Vec<f32>> = Vec::with_capacity(N);
    let mut targets: Vec<f32> = Vec::with_capacity(N);
    for i in 0..N {
        let high = i % 2 == 0;
        let x0 = if high {
            rng.gen_range(0.3..1.0)
        } else {
            rng.gen_range(-1.0..-0.3)
        };
        let obs = vec![x0, rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
        obses.push(obs);
        targets.push(if high { TARGET_HIGH } else { TARGET_LOW });
    }

    let predict = |m: &MyZeroModel, obs: &[f32]| -> f32 {
        let (_latent, _prior, value) = (&m).initial_state(obs);
        value
    };

    // 训练前后的可分度：高价值组均值 − 低价值组均值
    let gap = |m: &MyZeroModel| -> f32 {
        let mut hi = 0.0;
        let mut lo = 0.0;
        for (obs, &t) in obses.iter().zip(&targets) {
            let v = predict(m, obs);
            if t == TARGET_HIGH {
                hi += v;
            } else {
                lo += v;
            }
        }
        (hi - lo) / (N as f32 / 2.0)
    };

    let gap_before = gap(&model);

    let n_iters = 250;
    let mut last_loss = f32::INFINITY;
    for _ in 0..n_iters {
        opt.zero_grad().unwrap();
        let mut loss_sum = 0.0;
        for (obs, &t) in obses.iter().zip(&targets) {
            let loss = model
                .train_unroll(
                    obs,
                    &[],
                    &[uniform.clone()],
                    &[t],
                    &[],
                    None,
                    0.0,
                    0.0,
                    false,
                )
                .unwrap()
                * (1.0 / N as f32);
            loss_sum += loss.backward().unwrap();
        }
        opt.step().unwrap();
        last_loss = loss_sum;
    }

    let gap_after = gap(&model);
    let true_gap = TARGET_HIGH - TARGET_LOW; // = 14.0

    println!(
        "[value-head 容量] gap 训练前={gap_before:.2} → 训练后={gap_after:.2}（真实间隔={true_gap:.1}，末 loss={last_loss:.4}）"
    );

    // 决定性断言：训练后高/低价值组预测必须显著拉开（坍缩成常数则 gap≈0）。
    assert!(
        gap_after > 4.0,
        "value head 未能区分高/低价值 obs（gap={gap_after:.2}）→ 坐实分叉 (b)：head 学不动，\
         病根在 head/loss/repr 表达力，而非上游 target"
    );
    // 至少把真实间隔学到一半以上，确认是「真拟合」而非边际抖动。
    assert!(
        gap_after > true_gap * 0.5,
        "value head 仅学到 gap={gap_after:.2} < 真实间隔一半（{:.1}）",
        true_gap * 0.5
    );
}
