//! CartPole 同 seed 训后表现可复现性（含真实 backward 阶段）。
//!
//! 40 局：buffer ≥10 后触发训练，ep25 有一次 periodic greedy eval，训末再 eval。

use crate::rl::algo::my_zero::MyZero;
use crate::rl::algo::my_zero::report::{EvalReport, TrainReport};
use serial_test::serial;

fn run_cartpole_seed42_40ep() -> (TrainReport, EvalReport) {
    let mz = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(40)
        .seed(42)
        .train()
        .expect("CartPole 训练应成功");
    let train = mz.train_report().expect("train 应附带报告").clone();
    let mz = mz.eval(10).expect("eval 应成功");
    let eval = mz.eval_report().expect("eval 应附带报告").clone();
    (train, eval)
}

#[test]
#[serial]
fn cartpole_same_seed_two_runs_match() {
    let (t1, e1) = run_cartpole_seed42_40ep();
    let (t2, e2) = run_cartpole_seed42_40ep();
    assert_eq!(
        t1.final_greedy, t2.final_greedy,
        "训末 greedy 应一致：{} vs {}",
        t1.final_greedy, t2.final_greedy
    );
    assert_eq!(
        t1.best_greedy, t2.best_greedy,
        "训练内 best greedy 应一致：{} vs {}",
        t1.best_greedy, t2.best_greedy
    );
    assert_eq!(e1, e2, "独立 eval×10 应完全一致");
}

/// 供 shell 并行跑多次、抓 stdout 对比（`--exact --nocapture`）。
#[test]
#[serial]
fn cartpole_seed42_40ep_metrics_line() {
    let (train, eval) = run_cartpole_seed42_40ep();
    println!(
        "SEED_METRICS final_greedy={:.6} best_greedy={:.6} best_at_ep={:?} eval_mean={:.6} eval_returns={:?}",
        train.final_greedy,
        train.best_greedy,
        train.best_at_episode,
        eval.mean_return,
        eval.episode_returns,
    );
}
