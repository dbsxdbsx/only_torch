/*
 * @Author       : 老董
 * @Date         : 2026-03-11
 * @Description  : 学习率调度器 — PyTorch 风格
 *
 * 提供常用的学习率衰减策略，与 PyTorch `torch.optim.lr_scheduler` 行为对齐。
 *
 * 设计要点：
 * - LrScheduler trait 只负责计算 lr，不持有 Optimizer（避免生命周期耦合）
 * - step() 返回新 lr，用户可手动调用 optimizer.set_learning_rate()
 * - step_with() 是便捷方法，自动设置优化器学习率
 * - epoch 从 0 开始计数，第一次 step() 后 epoch 变为 1（与 PyTorch 一致）
 */

use super::Optimizer;

// ═══════════════════════════════════════════════════════════════
// LrScheduler Trait
// ═══════════════════════════════════════════════════════════════

/// 学习率调度器 trait
///
/// # 使用示例
/// ```ignore
/// let mut scheduler = CosineAnnealingLR::new(0.001, 50, 0.0);
///
/// for epoch in 0..50 {
///     // 训练 ...
///     scheduler.step_with(&mut optimizer);
/// }
/// ```
pub trait LrScheduler {
    /// 推进一步，返回新的学习率
    ///
    /// 每个 epoch 结束后调用一次。内部 epoch 计数器自动递增。
    fn step(&mut self) -> f32;

    /// 便捷方法：推进一步并自动设置优化器学习率
    fn step_with(&mut self, optimizer: &mut dyn Optimizer) -> f32 {
        let lr = self.step();
        optimizer.set_learning_rate(lr);
        lr
    }

    /// 获取当前学习率（不推进 epoch）
    fn get_lr(&self) -> f32;

    /// 获取当前 epoch 计数
    fn get_last_epoch(&self) -> usize;
}

// ═══════════════════════════════════════════════════════════════
// CosineAnnealingLR
// ═══════════════════════════════════════════════════════════════

/// 余弦退火学习率调度器
///
/// 学习率按余弦曲线从 `lr_init` 衰减到 `eta_min`：
///
/// ```text
/// lr = eta_min + 0.5 * (lr_init - eta_min) * (1 + cos(π * epoch / T_max))
/// ```
///
/// # 参数
/// - `lr_init`: 初始学习率
/// - `t_max`: 半周期长度（通常等于总 epoch 数）
/// - `eta_min`: 最小学习率（默认 0.0）
///
/// # 使用示例
/// ```ignore
/// let mut scheduler = CosineAnnealingLR::new(0.001, 50, 0.0);
/// for epoch in 0..50 {
///     // ... 训练 ...
///     scheduler.step_with(&mut optimizer);
/// }
/// ```
pub struct CosineAnnealingLR {
    lr_init: f32,
    t_max: usize,
    eta_min: f32,
    epoch: usize,
    current_lr: f32,
}

impl CosineAnnealingLR {
    /// 创建余弦退火调度器
    ///
    /// # 参数
    /// - `lr_init`: 初始学习率
    /// - `t_max`: 半周期长度（通常等于总 epoch 数）
    /// - `eta_min`: 最小学习率
    pub fn new(lr_init: f32, t_max: usize, eta_min: f32) -> Self {
        assert!(t_max > 0, "CosineAnnealingLR: t_max 必须 > 0");
        Self {
            lr_init,
            t_max,
            eta_min,
            epoch: 0,
            current_lr: lr_init,
        }
    }
}

impl LrScheduler for CosineAnnealingLR {
    fn step(&mut self) -> f32 {
        self.epoch += 1;
        self.current_lr = self.eta_min
            + 0.5
                * (self.lr_init - self.eta_min)
                * (1.0 + (std::f32::consts::PI * self.epoch as f32 / self.t_max as f32).cos());
        self.current_lr
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn get_last_epoch(&self) -> usize {
        self.epoch
    }
}

// ═══════════════════════════════════════════════════════════════
// StepLR
// ═══════════════════════════════════════════════════════════════

/// 阶梯式学习率调度器
///
/// 每隔 `step_size` 个 epoch，学习率乘以 `gamma`：
///
/// ```text
/// lr = lr_init * gamma^(epoch / step_size)
/// ```
///
/// # 参数
/// - `lr_init`: 初始学习率
/// - `step_size`: 每隔多少个 epoch 衰减一次
/// - `gamma`: 衰减系数（如 0.1 表示每次衰减为原来的 10%）
///
/// # 使用示例
/// ```ignore
/// // 每 30 个 epoch 学习率衰减为原来的 10%
/// let mut scheduler = StepLR::new(0.1, 30, 0.1);
/// ```
pub struct StepLR {
    lr_init: f32,
    step_size: usize,
    gamma: f32,
    epoch: usize,
    current_lr: f32,
}

impl StepLR {
    /// 创建阶梯式调度器
    pub fn new(lr_init: f32, step_size: usize, gamma: f32) -> Self {
        assert!(step_size > 0, "StepLR: step_size 必须 > 0");
        Self {
            lr_init,
            step_size,
            gamma,
            epoch: 0,
            current_lr: lr_init,
        }
    }
}

impl LrScheduler for StepLR {
    fn step(&mut self) -> f32 {
        self.epoch += 1;
        // PyTorch StepLR: lr = lr_init * gamma^(epoch // step_size)
        let power = self.epoch / self.step_size;
        self.current_lr = self.lr_init * self.gamma.powi(power as i32);
        self.current_lr
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn get_last_epoch(&self) -> usize {
        self.epoch
    }
}

// ═══════════════════════════════════════════════════════════════
// LambdaLR
// ═══════════════════════════════════════════════════════════════

/// 自定义函数学习率调度器
///
/// 学习率由用户提供的 lambda 函数决定：
///
/// ```text
/// lr = lr_init * lambda_fn(epoch)
/// ```
///
/// # 参数
/// - `lr_init`: 初始学习率
/// - `lambda_fn`: 接收 epoch（从 1 开始）返回乘数的函数
///
/// # 使用示例
/// ```ignore
/// // 指数衰减：lr = lr_init * 0.95^epoch
/// let mut scheduler = LambdaLR::new(0.1, |epoch| 0.95_f32.powi(epoch as i32));
///
/// // 线性衰减到 0：lr = lr_init * (1 - epoch/100)
/// let mut scheduler = LambdaLR::new(0.1, |epoch| 1.0 - epoch as f32 / 100.0);
/// ```
pub struct LambdaLR {
    lr_init: f32,
    lambda_fn: Box<dyn Fn(usize) -> f32>,
    epoch: usize,
    current_lr: f32,
}

impl LambdaLR {
    /// 创建自定义函数调度器
    pub fn new(lr_init: f32, lambda_fn: impl Fn(usize) -> f32 + 'static) -> Self {
        Self {
            lr_init,
            lambda_fn: Box::new(lambda_fn),
            epoch: 0,
            current_lr: lr_init,
        }
    }
}

impl LrScheduler for LambdaLR {
    fn step(&mut self) -> f32 {
        self.epoch += 1;
        self.current_lr = self.lr_init * (self.lambda_fn)(self.epoch);
        self.current_lr
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn get_last_epoch(&self) -> usize {
        self.epoch
    }
}
