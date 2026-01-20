use crate::errors::TensorError;
use crate::tensor::Tensor;
use std::ops::AddAssign;

impl AddAssign for Tensor {
    fn add_assign(&mut self, other: Self) {
        self.add_assign(&other);
    }
}

/// 张量的 += 操作，支持 NumPy 风格广播
///
/// # 广播规则
/// - 支持广播，但**广播后的结果形状必须与左操作数形状相同**
/// - 例如：`[3,4] += [1,4]` 成功（广播后仍是 [3,4]）
/// - 例如：`[3] += [1,3]` 失败（广播后变成 [1,3] ≠ [3]）
///
/// # Panics
/// 如果广播后形状与左操作数不同
impl<'a> AddAssign<&'a Self> for Tensor {
    fn add_assign(&mut self, other: &'a Self) {
        // 检查就地广播兼容性
        assert!(
            self.can_assign_broadcast_from(other),
            "{}",
            TensorError::IncompatibleShape
        );
        // 使用 ndarray 原生广播
        self.data += &other.data;
    }
}

impl AddAssign<f32> for Tensor {
    fn add_assign(&mut self, scalar: f32) {
        self.data += scalar;
    }
}

impl AddAssign<f32> for &mut Tensor {
    fn add_assign(&mut self, scalar: f32) {
        self.data += scalar;
    }
}
