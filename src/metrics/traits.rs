//! # 指标输入类型转换 Trait
//!
//! 提供统一的类型转换接口，让指标函数能接受多种输入类型。

use crate::tensor::Tensor;

/// 可转换为分类标签序列的类型
///
/// 用于分类指标（Accuracy、Precision、Recall、F1 等）的输入。
///
/// ## 支持的类型
///
/// - `&[usize]` / `Vec<usize>` - 直接使用
/// - `&[i32]` / `Vec<i32>` - 类型转换
/// - `Tensor` - 自动处理：
///   - `[batch]` 形状 → 直接取值作为类别索引
///   - `[batch, num_classes]` 形状 → 自动 argmax
///
/// ## 示例
///
/// ```rust
/// use only_torch::metrics::accuracy;
/// use only_torch::tensor::Tensor;
///
/// // 直接传 slice
/// let acc = accuracy(&[0, 1, 1], &[0, 1, 0]);
///
/// // 传 Tensor（自动 argmax）
/// let logits = Tensor::new(&[3, 2], &[0.1, 0.9, 0.8, 0.2, 0.3, 0.7]);
/// let labels = Tensor::new(&[3, 2], &[0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
/// let acc = accuracy(&logits, &labels);
/// ```
pub trait IntoClassLabels {
    /// 转换为类别标签向量
    fn to_class_labels(&self) -> Vec<usize>;
}

/// 可转换为浮点数值序列的类型
///
/// 用于回归指标（R²、MSE、MAE 等）的输入。
///
/// ## 支持的类型
///
/// - `&[f32]` / `Vec<f32>` - 直接使用
/// - `Tensor` - 自动处理：
///   - `[batch]` 形状 → 直接取值
///   - `[batch, 1]` 形状 → 展平取值
///
/// ## 示例
///
/// ```rust
/// use only_torch::metrics::r2_score;
/// use only_torch::tensor::Tensor;
///
/// // 直接传 slice
/// let r2 = r2_score(&[1.0, 2.0, 3.0], &[1.1, 2.0, 2.9]);
///
/// // 传 Tensor
/// let preds = Tensor::new(&[3, 1], &[1.0, 2.0, 3.0]);
/// let actuals = Tensor::new(&[3, 1], &[1.1, 2.0, 2.9]);
/// let r2 = r2_score(&preds, &actuals);
/// ```
pub trait IntoFloatValues {
    /// 转换为浮点数值向量
    fn to_float_values(&self) -> Vec<f32>;
}

// ============================================================================
// IntoClassLabels 实现
// ============================================================================

// 为 [usize] slice 实现
impl IntoClassLabels for [usize] {
    fn to_class_labels(&self) -> Vec<usize> {
        self.to_vec()
    }
}

// 为 Vec<usize> 实现
impl IntoClassLabels for Vec<usize> {
    fn to_class_labels(&self) -> Vec<usize> {
        self.clone()
    }
}

// 为 [i32] slice 实现
impl IntoClassLabels for [i32] {
    fn to_class_labels(&self) -> Vec<usize> {
        self.iter().map(|&x| x as usize).collect()
    }
}

// 为 Vec<i32> 实现
impl IntoClassLabels for Vec<i32> {
    fn to_class_labels(&self) -> Vec<usize> {
        self.iter().map(|&x| x as usize).collect()
    }
}

// 为 [u32] slice 实现
impl IntoClassLabels for [u32] {
    fn to_class_labels(&self) -> Vec<usize> {
        self.iter().map(|&x| x as usize).collect()
    }
}

// 为 Vec<u32> 实现
impl IntoClassLabels for Vec<u32> {
    fn to_class_labels(&self) -> Vec<usize> {
        self.iter().map(|&x| x as usize).collect()
    }
}

// 为 [i64] slice 实现
impl IntoClassLabels for [i64] {
    fn to_class_labels(&self) -> Vec<usize> {
        self.iter().map(|&x| x as usize).collect()
    }
}

// 为 Vec<i64> 实现
impl IntoClassLabels for Vec<i64> {
    fn to_class_labels(&self) -> Vec<usize> {
        self.iter().map(|&x| x as usize).collect()
    }
}

// 为 [u8] slice 实现
impl IntoClassLabels for [u8] {
    fn to_class_labels(&self) -> Vec<usize> {
        self.iter().map(|&x| x as usize).collect()
    }
}

// 为 Vec<u8> 实现
impl IntoClassLabels for Vec<u8> {
    fn to_class_labels(&self) -> Vec<usize> {
        self.iter().map(|&x| x as usize).collect()
    }
}

// ============================================================================
// 为固定大小数组实现（支持 &[1, 2, 3] 这种字面量写法）
// ============================================================================

impl<const N: usize> IntoClassLabels for [usize; N] {
    fn to_class_labels(&self) -> Vec<usize> {
        self.to_vec()
    }
}

impl<const N: usize> IntoClassLabels for [i32; N] {
    fn to_class_labels(&self) -> Vec<usize> {
        self.iter().map(|&x| x as usize).collect()
    }
}

impl<const N: usize> IntoClassLabels for [u32; N] {
    fn to_class_labels(&self) -> Vec<usize> {
        self.iter().map(|&x| x as usize).collect()
    }
}

impl<const N: usize> IntoClassLabels for [i64; N] {
    fn to_class_labels(&self) -> Vec<usize> {
        self.iter().map(|&x| x as usize).collect()
    }
}

impl<const N: usize> IntoClassLabels for [u8; N] {
    fn to_class_labels(&self) -> Vec<usize> {
        self.iter().map(|&x| x as usize).collect()
    }
}

// 为 Tensor 实现（核心：自动 argmax）
impl IntoClassLabels for Tensor {
    fn to_class_labels(&self) -> Vec<usize> {
        let shape = self.shape();
        match shape.len() {
            // [batch] 形状：直接作为类别索引
            1 => (0..shape[0]).map(|i| self[[i]] as usize).collect(),
            // [batch, num_classes] 形状：自动 argmax
            2 => {
                let argmax = self.argmax(1);
                (0..shape[0]).map(|i| argmax[[i]] as usize).collect()
            }
            _ => panic!(
                "IntoClassLabels: 不支持的 Tensor 形状 {:?}，期望 [batch] 或 [batch, num_classes]",
                shape
            ),
        }
    }
}

// ============================================================================
// IntoFloatValues 实现
// ============================================================================

// 为 [f32] slice 实现
impl IntoFloatValues for [f32] {
    fn to_float_values(&self) -> Vec<f32> {
        self.to_vec()
    }
}

// 为 Vec<f32> 实现
impl IntoFloatValues for Vec<f32> {
    fn to_float_values(&self) -> Vec<f32> {
        self.clone()
    }
}

// 为 [f64] slice 实现（转换精度）
impl IntoFloatValues for [f64] {
    fn to_float_values(&self) -> Vec<f32> {
        self.iter().map(|&x| x as f32).collect()
    }
}

// 为 Vec<f64> 实现
impl IntoFloatValues for Vec<f64> {
    fn to_float_values(&self) -> Vec<f32> {
        self.iter().map(|&x| x as f32).collect()
    }
}

// ============================================================================
// 为固定大小数组实现（支持 &[1.0, 2.0, 3.0] 这种字面量写法）
// ============================================================================

impl<const N: usize> IntoFloatValues for [f32; N] {
    fn to_float_values(&self) -> Vec<f32> {
        self.to_vec()
    }
}

impl<const N: usize> IntoFloatValues for [f64; N] {
    fn to_float_values(&self) -> Vec<f32> {
        self.iter().map(|&x| x as f32).collect()
    }
}

// 为 Tensor 实现
impl IntoFloatValues for Tensor {
    fn to_float_values(&self) -> Vec<f32> {
        let shape = self.shape();
        let n = shape[0];
        match shape.len() {
            // [batch] 形状：直接取值
            1 => (0..n).map(|i| self[[i]]).collect(),
            // [batch, 1] 形状：展平取值
            2 if shape[1] == 1 => (0..n).map(|i| self[[i, 0]]).collect(),
            // [batch, features] 形状：仅取第一列（回归通常只有一个输出）
            2 => {
                // 如果 features > 1，发出警告但仍取第一列
                if shape[1] > 1 {
                    eprintln!(
                        "警告: IntoFloatValues 收到 {:?} 形状的 Tensor，仅取第一列",
                        shape
                    );
                }
                (0..n).map(|i| self[[i, 0]]).collect()
            }
            _ => panic!(
                "IntoFloatValues: 不支持的 Tensor 形状 {:?}，期望 [batch] 或 [batch, 1]",
                shape
            ),
        }
    }
}
