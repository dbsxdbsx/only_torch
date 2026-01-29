//! # 评估指标模块
//!
//! 提供机器学习模型评估的常用指标函数。
//!
//! ## 模块结构
//!
//! - [`classification`] - 分类指标（Accuracy, Precision, Recall 等）
//! - [`regression`] - 回归指标（R², MSE, MAE 等）
//! - [`traits`] - 输入类型转换 Trait
//!
//! ## 设计理念
//!
//! 所有指标函数返回实现了 [`Metric`] trait 的结构体，提供**完全统一**的访问方式：
//!
//! | 方法 | 说明 |
//! |------|------|
//! | `.value()` | 获取主指标值（比例或数值） |
//! | `.n_samples()` | 获取样本数 |
//! | `.percent()` | 百分比形式（value × 100） |
//! | `.weighted()` | 加权值（value × `n_samples，批处理累加用`） |
//!
//! ## 多态输入
//!
//! 所有指标函数都支持多种输入类型，无需手动转换：
//!
//! - **分类指标**：支持 `&[usize]`、`Vec<i32>`、`Tensor`（自动 argmax）等
//! - **回归指标**：支持 `&[f32]`、`Vec<f32>`、`Tensor` 等
//!
//! ## 使用示例
//!
//! ```rust
//! use only_torch::metrics::{accuracy, precision, r2_score};
//!
//! // ========== 统一的访问方式（无需导入 Metric trait）==========
//! let acc = accuracy(&[0, 1, 1, 0, 1], &[0, 1, 0, 0, 1]);
//! let prec = precision(&[1, 1, 0, 1, 0], &[1, 0, 0, 1, 1]);
//! let r2 = r2_score(&[2.5, 0.0, 2.0, 8.0], &[3.0, -0.5, 2.0, 7.0]);
//!
//! // 所有指标都用 .value() 获取主值
//! println!("Accuracy = {:.4}", acc.value());   // 0.8
//! println!("Precision = {:.4}", prec.value()); // ~0.58
//! println!("R² = {:.4}", r2.value());          // ~0.95
//!
//! // 所有指标都用 .n_samples() 获取样本数
//! println!("n = {}", acc.n_samples());  // 5
//! println!("n = {}", r2.n_samples());   // 4
//!
//! // ========== 批处理累加 ==========
//! let mut weighted_sum = 0.0f32;
//! let mut total_n = 0usize;
//!
//! // 模拟多个 batch
//! let batches = vec![
//!     (&[0, 1, 1][..], &[0, 1, 0][..]),
//!     (&[0, 1][..], &[0, 1][..]),
//! ];
//!
//! for (preds, labels) in batches {
//!     let result = accuracy(preds, labels);
//!     weighted_sum += result.weighted();  // value × n_samples
//!     total_n += result.n_samples();
//! }
//!
//! let final_acc = weighted_sum / total_n as f32;
//! println!("Final Accuracy = {:.4}", final_acc);
//! ```

pub mod classification;
pub mod regression;
pub mod traits;

#[cfg(test)]
mod tests;

use std::fmt;

// ============================================================================
// Metric Trait
// ============================================================================

/// 所有指标结果的统一接口
///
/// 所有指标函数返回的结构体都实现此 trait。
///
/// **注意**：普通用户无需导入此 trait，因为所有方法都已作为 inherent method 实现，
/// 可直接调用。此 trait 主要供需要泛型编程的高级用户使用。
///
/// ## 方法
///
/// | 方法 | 说明 |
/// |------|------|
/// | [`value()`](Metric::value) | 获取主指标值 |
/// | [`n_samples()`](Metric::n_samples) | 获取样本数 |
/// | [`percent()`](Metric::percent) | 百分比形式（value × 100） |
/// | [`weighted()`](Metric::weighted) | 加权值（批处理累加用） |
///
/// ## 泛型编程示例
///
/// ```rust
/// use only_torch::metrics::{accuracy, r2_score, Metric};
///
/// // 只有需要泛型时才导入 Metric trait
/// fn print_metric<M: Metric>(name: &str, metric: &M) {
///     println!("{}: {:.4} (n={})", name, metric.value(), metric.n_samples());
/// }
///
/// let acc = accuracy(&[0, 1, 1], &[0, 1, 0]);
/// let r2 = r2_score(&[1.0, 2.0], &[1.1, 2.0]);
///
/// print_metric("Accuracy", &acc);
/// print_metric("R²", &r2);
/// ```
pub trait Metric {
    /// 获取主指标值
    ///
    /// - 分类指标（Accuracy, Precision 等）：返回比例（0.0 ~ 1.0）
    /// - 回归指标（R², MSE 等）：返回数值
    fn value(&self) -> f32;

    /// 获取参与计算的样本数
    fn n_samples(&self) -> usize;

    /// 获取百分比形式（value × 100）
    ///
    /// 对于 0~1 范围的比例指标有意义。
    #[inline]
    fn percent(&self) -> f32 {
        self.value() * 100.0
    }

    /// 获取加权值（value × `n_samples`）
    ///
    /// 用于批处理场景的加权累加，最终除以总样本数得到全局指标值。
    #[inline]
    fn weighted(&self) -> f32 {
        self.value() * self.n_samples() as f32
    }
}

// ============================================================================
// 指标结果类型
// ============================================================================

/// 分类指标结果
///
/// 用于 Accuracy、Precision、Recall、F1 等分类指标。
///
/// ## 示例
///
/// ```rust
/// use only_torch::metrics::accuracy;
///
/// let result = accuracy(&[0, 1, 1, 0, 1], &[0, 1, 0, 0, 1]);
///
/// // 直接使用方法，无需导入 Metric trait
/// assert!((result.value() - 0.8).abs() < 1e-6);
/// assert_eq!(result.n_samples(), 5);
/// assert!((result.percent() - 80.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassificationMetric {
    metric_value: f32,
    sample_count: usize,
}

impl ClassificationMetric {
    /// 创建新的分类指标结果
    #[inline]
    pub(crate) const fn new(value: f32, n_samples: usize) -> Self {
        Self {
            metric_value: value,
            sample_count: n_samples,
        }
    }

    // ========== Inherent methods（用户无需导入 Metric trait）==========

    /// 获取主指标值（0.0 ~ 1.0）
    #[inline]
    pub const fn value(&self) -> f32 {
        self.metric_value
    }

    /// 获取样本数
    #[inline]
    pub const fn n_samples(&self) -> usize {
        self.sample_count
    }

    /// 获取百分比形式（0.0 ~ 100.0）
    #[inline]
    pub fn percent(&self) -> f32 {
        self.metric_value * 100.0
    }

    /// 获取加权值（value × `n_samples），用于批处理累加`
    #[inline]
    pub fn weighted(&self) -> f32 {
        self.metric_value * self.sample_count as f32
    }
}

impl Metric for ClassificationMetric {
    #[inline]
    fn value(&self) -> f32 {
        self.value()
    }

    #[inline]
    fn n_samples(&self) -> usize {
        self.n_samples()
    }

    #[inline]
    fn percent(&self) -> f32 {
        self.percent()
    }

    #[inline]
    fn weighted(&self) -> f32 {
        self.weighted()
    }
}

impl fmt::Display for ClassificationMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}% (n={})", self.percent(), self.sample_count)
    }
}

/// 回归指标结果
///
/// 用于 R²、MSE、MAE 等回归指标。
///
/// ## 示例
///
/// ```rust
/// use only_torch::metrics::r2_score;
///
/// let result = r2_score(&[2.5, 0.0, 2.0, 8.0], &[3.0, -0.5, 2.0, 7.0]);
///
/// // 直接使用方法，无需导入 Metric trait
/// assert!(result.value() > 0.9);
/// assert_eq!(result.n_samples(), 4);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegressionMetric {
    metric_value: f32,
    sample_count: usize,
}

impl RegressionMetric {
    /// 创建新的回归指标结果
    #[inline]
    pub(crate) const fn new(value: f32, n_samples: usize) -> Self {
        Self {
            metric_value: value,
            sample_count: n_samples,
        }
    }

    // ========== Inherent methods（用户无需导入 Metric trait）==========

    /// 获取主指标值
    #[inline]
    pub const fn value(&self) -> f32 {
        self.metric_value
    }

    /// 获取样本数
    #[inline]
    pub const fn n_samples(&self) -> usize {
        self.sample_count
    }

    /// 获取百分比形式（value × 100）
    #[inline]
    pub fn percent(&self) -> f32 {
        self.metric_value * 100.0
    }

    /// 获取加权值（value × `n_samples），用于批处理累加`
    #[inline]
    pub fn weighted(&self) -> f32 {
        self.metric_value * self.sample_count as f32
    }
}

impl Metric for RegressionMetric {
    #[inline]
    fn value(&self) -> f32 {
        self.value()
    }

    #[inline]
    fn n_samples(&self) -> usize {
        self.n_samples()
    }

    #[inline]
    fn percent(&self) -> f32 {
        self.percent()
    }

    #[inline]
    fn weighted(&self) -> f32 {
        self.weighted()
    }
}

impl fmt::Display for RegressionMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4} (n={})", self.metric_value, self.sample_count)
    }
}

/// 多标签分类指标结果
///
/// 用于多标签分类任务，提供**总体准确率**和**每个标签的准确率**。
///
/// ## 设计理念
///
/// - 实现 [`Metric`] trait，保持与其他指标的 API 一致性
/// - 内部复用 [`ClassificationMetric`]，不创造新的基础类型
/// - 一次计算同时得到总体和分项统计，避免重复遍历
///
/// ## 示例
///
/// ```rust
/// use only_torch::metrics::multilabel_loose_accuracy;
/// use only_torch::tensor::Tensor;
///
/// let preds = Tensor::new(&[0.9, 0.1, 0.8, 0.2], &[2, 2]);
/// let actuals = Tensor::new(&[1.0, 0.0, 1.0, 1.0], &[2, 2]);
///
/// let result = multilabel_loose_accuracy(&preds, &actuals, 0.5);
///
/// // 总体准确率
/// println!("总体: {:.1}%", result.percent());
///
/// // 各标签准确率
/// for (i, label_metric) in result.per_label().iter().enumerate() {
///     println!("  标签{}: {:.1}%", i, label_metric.percent());
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MultiLabelMetric {
    /// 总体准确率（所有标签）
    overall: ClassificationMetric,
    /// 每个标签的准确率
    per_label: Vec<ClassificationMetric>,
    /// 样本数（非标签数）
    num_samples: usize,
}

impl MultiLabelMetric {
    /// 创建新的多标签指标结果
    #[inline]
    pub(crate) fn new(
        overall: ClassificationMetric,
        per_label: Vec<ClassificationMetric>,
        num_samples: usize,
    ) -> Self {
        Self {
            overall,
            per_label,
            num_samples,
        }
    }

    // ========== Inherent methods（用户无需导入 Metric trait）==========

    /// 获取总体准确率（0.0 ~ 1.0）
    #[inline]
    pub fn value(&self) -> f32 {
        self.overall.value()
    }

    /// 获取总标签数（batch × num_labels）
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.overall.n_samples()
    }

    /// 获取总体百分比（0.0 ~ 100.0）
    #[inline]
    pub fn percent(&self) -> f32 {
        self.overall.percent()
    }

    /// 获取总体加权值（value × n_samples）
    #[inline]
    pub fn weighted(&self) -> f32 {
        self.overall.weighted()
    }

    // ========== 多标签特有方法 ==========

    /// 获取每个标签的准确率
    ///
    /// 返回一个 slice，长度等于标签数。
    /// 每个元素是该标签的 [`ClassificationMetric`]。
    #[inline]
    pub fn per_label(&self) -> &[ClassificationMetric] {
        &self.per_label
    }

    /// 获取标签数
    #[inline]
    pub fn num_labels(&self) -> usize {
        self.per_label.len()
    }

    /// 获取样本数（非标签数）
    ///
    /// 注意：`n_samples()` 返回的是总标签数（batch × num_labels），
    /// 而 `num_samples()` 返回的是样本数（batch）。
    #[inline]
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }
}

impl Metric for MultiLabelMetric {
    #[inline]
    fn value(&self) -> f32 {
        self.value()
    }

    #[inline]
    fn n_samples(&self) -> usize {
        self.n_samples()
    }

    #[inline]
    fn percent(&self) -> f32 {
        self.percent()
    }

    #[inline]
    fn weighted(&self) -> f32 {
        self.weighted()
    }
}

impl fmt::Display for MultiLabelMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.2}% (n={}, labels={})",
            self.percent(),
            self.num_samples,
            self.num_labels()
        )
    }
}

// ============================================================================
// 导出
// ============================================================================

// 导出 Trait
pub use traits::{IntoClassLabels, IntoFloatValues};

// 导出常用函数
pub use classification::{
    accuracy, confusion_matrix, f1_score, multilabel_loose_accuracy, multilabel_strict_accuracy,
    precision, recall,
};
pub use regression::r2_score;

// 注：MultiLabelMetric 已在本模块定义并自动导出
