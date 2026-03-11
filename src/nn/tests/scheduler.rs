/*
 * LR Scheduler 单元测试
 *
 * 参考值由 PyTorch 脚本 scripts/_lr_scheduler_ref.py 生成，
 * 确保与 PyTorch torch.optim.lr_scheduler 行为一致。
 */

use crate::nn::optimizer::scheduler::*;

const TOL: f32 = 1e-5;

fn assert_lr_close(actual: f32, expected: f32, label: &str) {
    assert!(
        (actual - expected).abs() < TOL,
        "{label}: expected {expected:.6}, got {actual:.6}"
    );
}

// ═══════════════════════════════════════════════════════════════
// CosineAnnealingLR
// ═══════════════════════════════════════════════════════════════

#[test]
fn cosine_annealing_default_eta_min() {
    // PyTorch: CosineAnnealingLR(optimizer, T_max=10), lr=0.1
    let mut s = CosineAnnealingLR::new(0.1, 10, 0.0);

    assert_lr_close(s.get_lr(), 0.1, "epoch 0");

    let lr1 = s.step();
    assert_lr_close(lr1, 0.097553, "epoch 1");
    assert_eq!(s.get_last_epoch(), 1);

    // Advance to epoch 5
    for _ in 2..=4 { s.step(); }
    let lr5 = s.step();
    assert_lr_close(lr5, 0.050000, "epoch 5");

    // Advance to epoch 10
    for _ in 6..=9 { s.step(); }
    let lr10 = s.step();
    assert_lr_close(lr10, 0.0, "epoch 10");
    assert_eq!(s.get_last_epoch(), 10);
}

#[test]
fn cosine_annealing_with_eta_min() {
    // PyTorch: CosineAnnealingLR(optimizer, T_max=10, eta_min=0.01), lr=0.1
    let mut s = CosineAnnealingLR::new(0.1, 10, 0.01);

    let lr1 = s.step();
    assert_lr_close(lr1, 0.097798, "epoch 1");

    for _ in 2..=4 { s.step(); }
    let lr5 = s.step();
    assert_lr_close(lr5, 0.055000, "epoch 5");

    for _ in 6..=9 { s.step(); }
    let lr10 = s.step();
    assert_lr_close(lr10, 0.010000, "epoch 10");
}

#[test]
fn cosine_annealing_full_cycle() {
    // 验证整个衰减曲线单调递减（eta_min=0, T_max=20）
    let mut s = CosineAnnealingLR::new(0.01, 20, 0.0);
    let mut prev = s.get_lr();
    for _ in 0..20 {
        let lr = s.step();
        assert!(lr <= prev + TOL, "lr should be non-increasing");
        prev = lr;
    }
    assert_lr_close(prev, 0.0, "final lr");
}

// ═══════════════════════════════════════════════════════════════
// StepLR
// ═══════════════════════════════════════════════════════════════

#[test]
fn step_lr_basic() {
    // PyTorch: StepLR(optimizer, step_size=3, gamma=0.5), lr=0.1
    let mut s = StepLR::new(0.1, 3, 0.5);

    // epoch 1-2: lr = 0.1 * 0.5^0 = 0.1
    let lr1 = s.step();
    assert_lr_close(lr1, 0.100000, "epoch 1");
    let lr2 = s.step();
    assert_lr_close(lr2, 0.100000, "epoch 2");

    // epoch 3: lr = 0.1 * 0.5^1 = 0.05
    let lr3 = s.step();
    assert_lr_close(lr3, 0.050000, "epoch 3");

    // epoch 4-5: still 0.05
    let lr4 = s.step();
    assert_lr_close(lr4, 0.050000, "epoch 4");
    s.step(); // epoch 5

    // epoch 6: lr = 0.1 * 0.5^2 = 0.025
    let lr6 = s.step();
    assert_lr_close(lr6, 0.025000, "epoch 6");

    // epoch 9: lr = 0.1 * 0.5^3 = 0.0125
    s.step(); // 7
    s.step(); // 8
    let lr9 = s.step();
    assert_lr_close(lr9, 0.012500, "epoch 9");
    assert_eq!(s.get_last_epoch(), 9);
}

#[test]
fn step_lr_gamma_01() {
    // gamma=0.1, step_size=5
    let mut s = StepLR::new(1.0, 5, 0.1);
    for _ in 0..4 { s.step(); }
    assert_lr_close(s.get_lr(), 1.0, "epoch 4");
    let lr5 = s.step();
    assert_lr_close(lr5, 0.1, "epoch 5");
    for _ in 6..=9 { s.step(); }
    let lr10 = s.step();
    assert_lr_close(lr10, 0.01, "epoch 10");
}

// ═══════════════════════════════════════════════════════════════
// LambdaLR
// ═══════════════════════════════════════════════════════════════

#[test]
fn lambda_lr_exponential_decay() {
    // PyTorch: LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95**epoch), lr=0.1
    let mut s = LambdaLR::new(0.1, |epoch| 0.95_f32.powi(epoch as i32));

    let lr1 = s.step();
    assert_lr_close(lr1, 0.095000, "epoch 1");

    for _ in 2..=4 { s.step(); }
    let lr5 = s.step();
    assert_lr_close(lr5, 0.077378, "epoch 5");

    for _ in 6..=9 { s.step(); }
    let lr10 = s.step();
    assert_lr_close(lr10, 0.059874, "epoch 10");
    assert_eq!(s.get_last_epoch(), 10);
}

#[test]
fn lambda_lr_linear_warmup() {
    // 线性 warmup: lr = lr_init * min(1, epoch / warmup_steps)
    let warmup = 5;
    let mut s = LambdaLR::new(0.01, move |epoch| {
        (epoch as f32 / warmup as f32).min(1.0)
    });

    let lr1 = s.step(); // epoch 1: 0.01 * (1/5) = 0.002
    assert_lr_close(lr1, 0.002, "warmup epoch 1");

    s.step(); // 2
    let lr3 = s.step(); // 3: 0.01 * (3/5) = 0.006
    assert_lr_close(lr3, 0.006, "warmup epoch 3");

    s.step(); // 4
    let lr5 = s.step(); // 5: 0.01 * 1.0 = 0.01
    assert_lr_close(lr5, 0.01, "warmup epoch 5");

    let lr6 = s.step(); // 6: still clamped at 1.0
    assert_lr_close(lr6, 0.01, "post-warmup epoch 6");
}

// ═══════════════════════════════════════════════════════════════
// Trait 接口
// ═══════════════════════════════════════════════════════════════

#[test]
fn trait_get_lr_before_step() {
    let s = CosineAnnealingLR::new(0.05, 10, 0.0);
    assert_lr_close(s.get_lr(), 0.05, "initial lr");
    assert_eq!(s.get_last_epoch(), 0);
}

#[test]
fn trait_step_consistency() {
    // step() 返回值应该与 get_lr() 一致
    let mut s = StepLR::new(0.1, 2, 0.5);
    for _ in 0..5 {
        let lr = s.step();
        assert_lr_close(lr, s.get_lr(), "step() == get_lr()");
    }
}
