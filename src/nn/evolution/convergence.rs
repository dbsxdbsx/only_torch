/*
 * @Author       : 老董
 * @Date         : 2026-03-06
 * @Description  : 训练收敛检测
 *
 * ConvergenceDetector 在训练循环中按优先级判断是否应停止训练：
 * NaN/Inf → FixedEpochs → max_epochs → loss 稳定 → 梯度消失。
 *
 * 调研依据：BOHB/Successive Halving 论文证实 budget-based 训练
 * 可以加速搜索 2-5 倍。当前默认 UntilConverged，后续可切换。
 */

use std::collections::VecDeque;

// ==================== TrainingBudget ====================

/// 训练预算模式
#[derive(Clone, Debug)]
pub enum TrainingBudget {
    /// 训练到收敛（默认）
    UntilConverged,
    /// 固定 epoch 数（用于快速筛选候选架构）
    FixedEpochs(usize),
}

// ==================== StopReason ====================

/// 训练停止原因（`should_stop` 的返回值语义）
///
/// DefaultCallback 可据此输出可读的停止原因日志。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// loss 为 NaN 或 Infinity（训练已爆炸）
    AbnormalLoss,
    /// FixedEpochs 模式到达指定 epoch 数
    BudgetExhausted,
    /// 到达 max_epochs 硬性上限（仅 UntilConverged 模式）
    MaxEpochsReached,
    /// loss 在 patience 窗口内相对变化低于阈值
    LossConverged,
    /// 梯度范数连续 patience 次低于阈值
    GradientVanished,
}

// ==================== ConvergenceConfig ====================

/// 收敛检测配置
///
/// 1 epoch = 对完整训练集遍历一次。
/// 全量训练（如 XOR）：1 epoch = 1 次 forward+backward。
/// Mini-batch：1 epoch = ceil(n_samples / batch_size) 次 forward+backward。
#[derive(Clone, Debug)]
pub struct ConvergenceConfig {
    pub budget: TrainingBudget,
    /// 判定收敛所需的连续 epoch 数（loss 窗口大小 / 梯度停滞计数阈值），最小为 1
    pub patience: usize,
    /// loss 相对变化阈值：(max - min) / (min.abs() + 1e-8) < tolerance 视为收敛
    pub loss_tolerance: f32,
    /// 梯度范数阈值：grad_norm < grad_tolerance 视为梯度消失
    pub grad_tolerance: f32,
    /// UntilConverged 模式下的硬性安全上限（防止 loss/grad 均不收敛时的无限循环）
    ///
    /// 仅对 UntilConverged 模式生效；FixedEpochs 模式自带预算，不受此限制。
    pub max_epochs: usize,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            budget: TrainingBudget::UntilConverged,
            patience: 5,
            loss_tolerance: 1e-4,
            grad_tolerance: 1e-5,
            max_epochs: 100,
        }
    }
}

// ==================== ConvergenceDetector ====================

/// 训练收敛检测器
///
/// 在训练循环中逐 epoch 调用 `should_stop()`，
/// 按优先级判断是否应终止当前网络的训练。
#[derive(Clone, Debug)]
pub struct ConvergenceDetector {
    /// 最近 patience 个 epoch 的 loss 值（滑动窗口）
    loss_history: VecDeque<f32>,
    /// 连续 grad_norm < grad_tolerance 的次数
    grad_stall_count: usize,
    config: ConvergenceConfig,
}

impl ConvergenceDetector {
    pub fn new(config: ConvergenceConfig) -> Self {
        assert!(
            config.patience >= 1,
            "patience 必须 >= 1，当前值: {}",
            config.patience
        );
        let capacity = config.patience;
        Self {
            loss_history: VecDeque::with_capacity(capacity),
            grad_stall_count: 0,
            config,
        }
    }

    /// 判断是否应停止训练
    ///
    /// `epoch` 为 0-indexed，在该 epoch 的训练步完成后调用。
    /// 返回 `Some(StopReason)` 表示应停止，`None` 表示继续训练。
    ///
    /// 判定优先级：
    /// 1. NaN / Infinity loss → `AbnormalLoss`
    /// 2. FixedEpochs 模式 → `BudgetExhausted`
    /// 3. max_epochs 硬性安全上限 → `MaxEpochsReached`
    /// 4. Loss 稳定 → `LossConverged`
    /// 5. 梯度消失 → `GradientVanished`
    pub fn should_stop(&mut self, epoch: usize, loss: f32, grad_norm: f32) -> Option<StopReason> {
        // 1. NaN / Infinity → 立即停止
        if loss.is_nan() || loss.is_infinite() {
            return Some(StopReason::AbnormalLoss);
        }

        // 2. FixedEpochs → 完成指定数量即停（不受 max_epochs 约束）
        if let TrainingBudget::FixedEpochs(n) = self.config.budget {
            return (epoch + 1 >= n).then_some(StopReason::BudgetExhausted);
        }

        // 3. max_epochs 硬性上限（仅 UntilConverged 模式）
        if epoch + 1 >= self.config.max_epochs {
            return Some(StopReason::MaxEpochsReached);
        }

        // 更新 loss 滑动窗口
        if self.loss_history.len() == self.config.patience {
            self.loss_history.pop_front();
        }
        self.loss_history.push_back(loss);

        // 4. Loss 稳定（窗口填满后才检查）
        if self.loss_history.len() == self.config.patience {
            let min = self
                .loss_history
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min);
            let max = self
                .loss_history
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let relative_change = (max - min) / (min.abs() + 1e-8);
            if relative_change < self.config.loss_tolerance {
                return Some(StopReason::LossConverged);
            }
        }

        // 5. 梯度消失（NaN/Inf grad_norm 不计入停滞：NaN < threshold 始终为 false）
        if grad_norm < self.config.grad_tolerance {
            self.grad_stall_count += 1;
        } else {
            self.grad_stall_count = 0;
        }
        if self.grad_stall_count >= self.config.patience {
            return Some(StopReason::GradientVanished);
        }

        None
    }
}
