//! # 回归评估指标
//!
//! 用于评估回归模型性能的指标函数。
//!
//! ## 统一接口
//!
//! 所有回归指标函数返回 [`RegressionMetric`](super::RegressionMetric)，
//! 实现了 [`Metric`](super::Metric) trait，提供统一的访问方式：
//!
//! - `.value()` - 获取指标值
//! - `.n_samples()` - 获取样本数
//! - `.percent()` - 百分比形式
//! - `.weighted()` - 加权值（批处理累加用）
//!
//! ## 多态输入
//!
//! 所有函数都支持多种输入类型：
//! - `&[f32]` / `Vec<f32>` / `[f32; N]` - 浮点数数组
//! - `&Tensor` - 自动处理：
//!   - `[batch]` 形状 → 直接取值
//!   - `[batch, 1]` 形状 → 展平取值

use super::RegressionMetric;
use super::traits::IntoFloatValues;

/// 计算 R²（决定系数 / Coefficient of Determination）
///
/// R² 表示模型对目标变量方差的解释程度：
/// - R² = 1.0：完美拟合
/// - R² = 0.0：模型等同于简单预测均值
/// - R² < 0.0：模型比均值预测更差
///
/// ## 公式
///
/// ```text
/// R² = 1 - SS_res / SS_tot
///
/// 其中：
/// - SS_res = Σ(actual - pred)²  （残差平方和）
/// - SS_tot = Σ(actual - mean)²  （总平方和）
/// ```
///
/// ## 参数
///
/// - `predictions`: 模型预测值（支持多种类型，见模块文档）
/// - `actuals`: 真实值（支持多种类型，见模块文档）
///
/// ## 返回值
///
/// 返回 [`RegressionMetric`](super::RegressionMetric)，通过统一接口访问：
/// - `.value()` - R² 分数（通常在 `(-∞, 1.0]`）
/// - `.n_samples()` - 样本数
///
/// ## 示例
///
/// ```rust
/// use only_torch::metrics::r2_score;
/// use only_torch::tensor::Tensor;
///
/// // 方式 1：直接传 slice
/// let result = r2_score(&[2.5, 0.0, 2.0, 8.0], &[3.0, -0.5, 2.0, 7.0]);
/// assert!((result.value() - 0.9486).abs() < 0.001);
/// assert_eq!(result.n_samples(), 4);
///
/// // 方式 2：传 Tensor
/// let preds = Tensor::new(&[2.5f32, 0.0, 2.0, 8.0], &[4]);
/// let actuals = Tensor::new(&[3.0f32, -0.5, 2.0, 7.0], &[4]);
/// let result = r2_score(&preds, &actuals);
/// assert!((result.value() - 0.9486).abs() < 0.001);
/// ```
///
/// ## 边界情况
///
/// - `如果所有真实值相同（SS_tot` = 0）：
///   - `预测完美（SS_res` = 0）→ 返回 value=1.0
///   - 预测有误差 → 返回 value=0.0
/// - 空输入 → 返回 value=0.0, `n_samples=0`
pub fn r2_score(
    predictions: &(impl IntoFloatValues + ?Sized),
    actuals: &(impl IntoFloatValues + ?Sized),
) -> RegressionMetric {
    let pred_vals = predictions.to_float_values();
    let true_vals = actuals.to_float_values();

    if pred_vals.is_empty() || true_vals.is_empty() {
        return RegressionMetric::new(0.0, 0);
    }

    let n = pred_vals.len().min(true_vals.len());

    // 计算真实值的均值
    let mean_actual: f32 = true_vals[..n].iter().sum::<f32>() / n as f32;

    // 计算残差平方和（SS_res）
    let ss_res: f32 = pred_vals[..n]
        .iter()
        .zip(true_vals[..n].iter())
        .map(|(pred, actual)| (actual - pred).powi(2))
        .sum();

    // 计算总平方和（SS_tot）
    let ss_tot: f32 = true_vals[..n]
        .iter()
        .map(|actual| (actual - mean_actual).powi(2))
        .sum();

    // 处理边界情况：所有真实值相同
    let value = if ss_tot == 0.0 {
        if ss_res == 0.0 { 1.0 } else { 0.0 }
    } else {
        1.0 - (ss_res / ss_tot)
    };

    RegressionMetric::new(value, n)
}
