use crate::nn::evolution::convergence::*;

// ==================== Default 配置 ====================

#[test]
fn test_default_config() {
    let config = ConvergenceConfig::default();
    assert!(matches!(config.budget, TrainingBudget::UntilConverged));
    assert_eq!(config.patience, 5);
    assert!((config.loss_tolerance - 1e-4).abs() < 1e-10);
    assert!((config.grad_tolerance - 1e-5).abs() < 1e-10);
    assert_eq!(config.max_epochs, 100);
}

// ==================== UntilConverged 模式 ====================

#[test]
fn test_stable_loss_converges() {
    let config = ConvergenceConfig {
        patience: 3,
        loss_tolerance: 1e-4,
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    assert_eq!(detector.should_stop(0, 0.5, 1.0), None);
    assert_eq!(detector.should_stop(1, 0.5, 1.0), None);
    assert_eq!(
        detector.should_stop(2, 0.5, 1.0),
        Some(StopReason::LossConverged)
    );
}

#[test]
fn test_continuous_descent_not_converged() {
    let config = ConvergenceConfig {
        patience: 3,
        loss_tolerance: 1e-4,
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    assert_eq!(detector.should_stop(0, 1.0, 1.0), None);
    assert_eq!(detector.should_stop(1, 0.8, 1.0), None);
    assert_eq!(detector.should_stop(2, 0.6, 1.0), None);
    assert_eq!(detector.should_stop(3, 0.4, 1.0), None);
    assert_eq!(detector.should_stop(4, 0.2, 1.0), None);
}

#[test]
fn test_gradient_vanishing_converges() {
    let config = ConvergenceConfig {
        patience: 3,
        grad_tolerance: 1e-5,
        loss_tolerance: 1e-4,
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    // loss 持续下降（不触发 loss 收敛），但梯度消失
    assert_eq!(detector.should_stop(0, 1.0, 1e-6), None);   // stall=1
    assert_eq!(detector.should_stop(1, 0.9, 1e-6), None);   // stall=2
    assert_eq!(
        detector.should_stop(2, 0.8, 1e-6),
        Some(StopReason::GradientVanished) // stall=3 = patience
    );
}

#[test]
fn test_gradient_stall_resets_on_recovery() {
    let config = ConvergenceConfig {
        patience: 3,
        grad_tolerance: 1e-5,
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    // 梯度消失 2 次后恢复，计数器重置
    assert_eq!(detector.should_stop(0, 1.0, 1e-6), None);   // stall=1
    assert_eq!(detector.should_stop(1, 0.9, 1e-6), None);   // stall=2
    assert_eq!(detector.should_stop(2, 0.8, 0.1), None);    // stall 重置=0
    assert_eq!(detector.should_stop(3, 0.7, 1e-6), None);   // stall=1
    assert_eq!(detector.should_stop(4, 0.6, 1e-6), None);   // stall=2
    assert_eq!(
        detector.should_stop(5, 0.5, 1e-6),
        Some(StopReason::GradientVanished) // stall=3
    );
}

// ==================== FixedEpochs 模式 ====================

#[test]
fn test_fixed_epochs() {
    let config = ConvergenceConfig {
        budget: TrainingBudget::FixedEpochs(10),
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    for epoch in 0..9 {
        assert_eq!(
            detector.should_stop(epoch, 0.5, 1.0),
            None,
            "epoch {epoch} should not stop"
        );
    }
    assert_eq!(
        detector.should_stop(9, 0.5, 1.0),
        Some(StopReason::BudgetExhausted)
    );
}

#[test]
fn test_fixed_epochs_one() {
    let config = ConvergenceConfig {
        budget: TrainingBudget::FixedEpochs(1),
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    assert_eq!(
        detector.should_stop(0, 0.5, 1.0),
        Some(StopReason::BudgetExhausted)
    );
}

// ==================== patience 精确语义 ====================

#[test]
fn test_patience_exact_boundary() {
    let config = ConvergenceConfig {
        patience: 5,
        loss_tolerance: 1e-4,
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    // patience-1 = 4 次稳定 loss 不触发
    for epoch in 0..4 {
        assert_eq!(
            detector.should_stop(epoch, 0.5, 1.0),
            None,
            "epoch {epoch}: patience 窗口未满不应触发"
        );
    }
    assert_eq!(
        detector.should_stop(4, 0.5, 1.0),
        Some(StopReason::LossConverged)
    );
}

// ==================== 异常输入 ====================

#[test]
fn test_nan_loss_immediate_stop() {
    let config = ConvergenceConfig::default();
    let mut detector = ConvergenceDetector::new(config);

    assert_eq!(
        detector.should_stop(0, f32::NAN, 1.0),
        Some(StopReason::AbnormalLoss)
    );
}

#[test]
fn test_infinity_loss_immediate_stop() {
    let config = ConvergenceConfig::default();
    let mut detector = ConvergenceDetector::new(config);

    assert_eq!(
        detector.should_stop(0, f32::INFINITY, 1.0),
        Some(StopReason::AbnormalLoss)
    );
}

#[test]
fn test_neg_infinity_loss_immediate_stop() {
    let config = ConvergenceConfig::default();
    let mut detector = ConvergenceDetector::new(config);

    assert_eq!(
        detector.should_stop(0, f32::NEG_INFINITY, 1.0),
        Some(StopReason::AbnormalLoss)
    );
}

#[test]
fn test_nan_after_normal_training() {
    let config = ConvergenceConfig::default();
    let mut detector = ConvergenceDetector::new(config);

    assert_eq!(detector.should_stop(0, 0.5, 1.0), None);
    assert_eq!(detector.should_stop(1, 0.4, 1.0), None);
    assert_eq!(
        detector.should_stop(2, f32::NAN, 1.0),
        Some(StopReason::AbnormalLoss)
    );
}

// ==================== 震荡 loss ====================

#[test]
fn test_oscillating_loss_converges_when_range_small() {
    let config = ConvergenceConfig {
        patience: 4,
        loss_tolerance: 0.01, // 1%
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    // 在 0.5 附近微小震荡：range / min < 1%
    assert_eq!(detector.should_stop(0, 0.500, 1.0), None);
    assert_eq!(detector.should_stop(1, 0.501, 1.0), None);
    assert_eq!(detector.should_stop(2, 0.499, 1.0), None);
    // 窗口填满：range=0.002, min=0.499, relative=0.002/0.499≈0.004 < 0.01 → 收敛
    assert_eq!(
        detector.should_stop(3, 0.500, 1.0),
        Some(StopReason::LossConverged)
    );
}

#[test]
fn test_oscillating_loss_not_converged_when_range_large() {
    let config = ConvergenceConfig {
        patience: 3,
        loss_tolerance: 1e-4,
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    assert_eq!(detector.should_stop(0, 0.5, 1.0), None);
    assert_eq!(detector.should_stop(1, 0.8, 1.0), None);
    assert_eq!(detector.should_stop(2, 0.3, 1.0), None);
    assert_eq!(detector.should_stop(3, 0.7, 1.0), None);
    assert_eq!(detector.should_stop(4, 0.4, 1.0), None);
}

// ==================== max_epochs 安全上限 ====================

#[test]
fn test_max_epochs_forces_stop() {
    let config = ConvergenceConfig {
        budget: TrainingBudget::UntilConverged,
        max_epochs: 10,
        patience: 5,
        loss_tolerance: 1e-4,
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    // loss 持续下降（未收敛），但到达 max_epochs 强制停止
    for epoch in 0..9 {
        let loss = 1.0 - epoch as f32 * 0.1;
        assert_eq!(
            detector.should_stop(epoch, loss, 1.0),
            None,
            "epoch {epoch}: 未到 max_epochs 不应停止"
        );
    }
    assert_eq!(
        detector.should_stop(9, 0.05, 1.0),
        Some(StopReason::MaxEpochsReached)
    );
}

#[test]
fn test_fixed_epochs_vs_max_epochs_smaller_budget() {
    // FixedEpochs(5) + max_epochs=10：以 FixedEpochs 为准
    let config = ConvergenceConfig {
        budget: TrainingBudget::FixedEpochs(5),
        max_epochs: 10,
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    for epoch in 0..4 {
        assert_eq!(detector.should_stop(epoch, 0.5, 1.0), None);
    }
    assert_eq!(
        detector.should_stop(4, 0.5, 1.0),
        Some(StopReason::BudgetExhausted)
    );
}

#[test]
fn test_fixed_epochs_vs_max_epochs_larger_budget() {
    // FixedEpochs(15) + max_epochs=10：max_epochs 不影响 FixedEpochs
    // （FixedEpochs 自带预算，不受 max_epochs 约束）
    let config = ConvergenceConfig {
        budget: TrainingBudget::FixedEpochs(15),
        max_epochs: 10,
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    for epoch in 0..14 {
        assert_eq!(
            detector.should_stop(epoch, 0.5, 1.0),
            None,
            "epoch {epoch}: FixedEpochs(15) 未完成不应停止"
        );
    }
    assert_eq!(
        detector.should_stop(14, 0.5, 1.0),
        Some(StopReason::BudgetExhausted)
    );
}

// ==================== 滑动窗口行为 ====================

#[test]
fn test_window_slides_correctly() {
    let config = ConvergenceConfig {
        patience: 3,
        loss_tolerance: 0.01,
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    // 先喂入差异大的 loss → 不收敛
    assert_eq!(detector.should_stop(0, 1.0, 1.0), None);
    assert_eq!(detector.should_stop(1, 0.5, 1.0), None);
    assert_eq!(detector.should_stop(2, 0.1, 1.0), None); // 窗口 [1.0, 0.5, 0.1]

    // 继续喂入稳定的 loss → 旧值滑出窗口
    assert_eq!(detector.should_stop(3, 0.1, 1.0), None);  // 窗口 [0.5, 0.1, 0.1]
    assert_eq!(
        detector.should_stop(4, 0.1, 1.0), // 窗口 [0.1, 0.1, 0.1] → range=0
        Some(StopReason::LossConverged)
    );
}

// ==================== 优先级验证 ====================

#[test]
fn test_nan_takes_priority_over_fixed_epochs() {
    let config = ConvergenceConfig {
        budget: TrainingBudget::FixedEpochs(100),
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    assert_eq!(
        detector.should_stop(0, f32::NAN, 1.0),
        Some(StopReason::AbnormalLoss)
    );
}

#[test]
fn test_loss_convergence_before_max_epochs() {
    let config = ConvergenceConfig {
        budget: TrainingBudget::UntilConverged,
        patience: 3,
        loss_tolerance: 1e-4,
        max_epochs: 100,
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    // 第 3 个 epoch 就 loss 稳定 → 不等到 max_epochs=100
    assert_eq!(detector.should_stop(0, 0.1, 1.0), None);
    assert_eq!(detector.should_stop(1, 0.1, 1.0), None);
    assert_eq!(
        detector.should_stop(2, 0.1, 1.0),
        Some(StopReason::LossConverged)
    );
}

#[test]
fn test_loss_convergence_takes_priority_over_gradient_vanishing() {
    let config = ConvergenceConfig {
        patience: 3,
        loss_tolerance: 1e-4,
        grad_tolerance: 1e-5,
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    // loss 稳定 + 梯度消失同 epoch 满足 → LossConverged 优先级更高
    assert_eq!(detector.should_stop(0, 0.5, 1e-6), None); // stall=1
    assert_eq!(detector.should_stop(1, 0.5, 1e-6), None); // stall=2
    // epoch 2: 窗口 [0.5, 0.5, 0.5] → LossConverged; 同时 stall=3 → GradientVanished
    assert_eq!(
        detector.should_stop(2, 0.5, 1e-6),
        Some(StopReason::LossConverged)
    );
}

// ==================== 边界条件 ====================

#[test]
fn test_near_zero_loss() {
    let config = ConvergenceConfig {
        patience: 3,
        loss_tolerance: 1e-4,
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    // loss 接近 0，分母的 1e-8 防止除零
    assert_eq!(detector.should_stop(0, 1e-10, 1.0), None);
    assert_eq!(detector.should_stop(1, 1e-10, 1.0), None);
    assert_eq!(
        detector.should_stop(2, 1e-10, 1.0),
        Some(StopReason::LossConverged)
    );
}

#[test]
fn test_patience_one() {
    let config = ConvergenceConfig {
        patience: 1,
        loss_tolerance: 1e-4,
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    // patience=1：窗口大小 1，第一次就填满；单个值的 range=0 → 收敛
    assert_eq!(
        detector.should_stop(0, 0.5, 1.0),
        Some(StopReason::LossConverged)
    );
}

#[test]
fn test_grad_stall_independent_of_loss() {
    let config = ConvergenceConfig {
        patience: 3,
        loss_tolerance: 1e-4,
        grad_tolerance: 1e-5,
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    // loss 持续下降（不触发 loss 收敛），但梯度一直很小
    assert_eq!(detector.should_stop(0, 1.0, 1e-6), None);
    assert_eq!(detector.should_stop(1, 0.5, 1e-6), None);
    assert_eq!(
        detector.should_stop(2, 0.1, 1e-6),
        Some(StopReason::GradientVanished)
    );
}

#[test]
#[should_panic(expected = "patience 必须 >= 1")]
fn test_patience_zero_panics() {
    let config = ConvergenceConfig {
        patience: 0,
        ..Default::default()
    };
    ConvergenceDetector::new(config);
}

// ==================== 负数 loss ====================

#[test]
fn test_negative_loss_convergence() {
    let config = ConvergenceConfig {
        patience: 3,
        loss_tolerance: 0.01, // 1%
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    // 负 loss（如 ELBO 变体）在 patience 窗口内稳定 → 收敛
    // 窗口 [-2.001, -2.000, -1.999]: range=0.002, min.abs()=2.001
    // relative_change = 0.002 / 2.001 ≈ 0.001 < 0.01 → 收敛
    assert_eq!(detector.should_stop(0, -2.001, 1.0), None);
    assert_eq!(detector.should_stop(1, -2.000, 1.0), None);
    assert_eq!(
        detector.should_stop(2, -1.999, 1.0),
        Some(StopReason::LossConverged)
    );
}

#[test]
fn test_negative_loss_descent_not_converged() {
    let config = ConvergenceConfig {
        patience: 3,
        loss_tolerance: 1e-4,
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    // 负 loss 持续下降 → 不收敛
    assert_eq!(detector.should_stop(0, -1.0, 1.0), None);
    assert_eq!(detector.should_stop(1, -2.0, 1.0), None);
    assert_eq!(detector.should_stop(2, -3.0, 1.0), None);
    assert_eq!(detector.should_stop(3, -4.0, 1.0), None);
}

// ==================== NaN/Inf grad_norm ====================

#[test]
fn test_nan_grad_norm_does_not_trigger_vanishing() {
    let config = ConvergenceConfig {
        patience: 3,
        grad_tolerance: 1e-5,
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    // NaN < threshold 为 false → 不计入梯度停滞，且重置计数器
    assert_eq!(detector.should_stop(0, 1.0, f32::NAN), None);
    assert_eq!(detector.should_stop(1, 0.9, f32::NAN), None);
    assert_eq!(detector.should_stop(2, 0.8, f32::NAN), None);
}

#[test]
fn test_inf_grad_norm_does_not_trigger_vanishing() {
    let config = ConvergenceConfig {
        patience: 3,
        grad_tolerance: 1e-5,
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    // Inf > threshold → 不计入梯度停滞
    assert_eq!(detector.should_stop(0, 1.0, f32::INFINITY), None);
    assert_eq!(detector.should_stop(1, 0.9, f32::INFINITY), None);
    assert_eq!(detector.should_stop(2, 0.8, f32::INFINITY), None);
}

#[test]
fn test_nan_grad_norm_resets_stall_counter() {
    let config = ConvergenceConfig {
        patience: 3,
        grad_tolerance: 1e-5,
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    // 2 次梯度消失 → NaN 打断（重置计数器）→ 需要再连续 3 次才触发
    assert_eq!(detector.should_stop(0, 1.0, 1e-6), None);   // stall=1
    assert_eq!(detector.should_stop(1, 0.9, 1e-6), None);   // stall=2
    assert_eq!(detector.should_stop(2, 0.8, f32::NAN), None); // NaN → stall 重置=0
    assert_eq!(detector.should_stop(3, 0.7, 1e-6), None);   // stall=1
    assert_eq!(detector.should_stop(4, 0.6, 1e-6), None);   // stall=2
    assert_eq!(
        detector.should_stop(5, 0.5, 1e-6),
        Some(StopReason::GradientVanished) // stall=3
    );
}

// ==================== 中途好转后重新停滞 ====================

#[test]
fn test_recovery_interrupts_then_restall() {
    let config = ConvergenceConfig {
        patience: 3,
        loss_tolerance: 0.01, // 1%
        ..Default::default()
    };
    let mut detector = ConvergenceDetector::new(config);

    // loss 停滞中突然下降一次（打断窗口），随后重新稳定
    // epoch 0: 窗口 [0.5]
    assert_eq!(detector.should_stop(0, 0.5, 1.0), None);
    // epoch 1: 窗口 [0.5, 0.5]
    assert_eq!(detector.should_stop(1, 0.5, 1.0), None);
    // epoch 2: loss 突然下降，窗口 [0.5, 0.5, 0.3]
    //   range=0.2, min=0.3, relative≈0.667 → 不收敛
    assert_eq!(detector.should_stop(2, 0.3, 1.0), None);
    // epoch 3: 新水平稳定，窗口 [0.5, 0.3, 0.3]
    //   range=0.2, min=0.3, relative≈0.667 → 不收敛（旧 0.5 还在窗口中）
    assert_eq!(detector.should_stop(3, 0.3, 1.0), None);
    // epoch 4: 窗口 [0.3, 0.3, 0.3] → range=0, relative=0 → 收敛
    assert_eq!(
        detector.should_stop(4, 0.3, 1.0),
        Some(StopReason::LossConverged)
    );
}
