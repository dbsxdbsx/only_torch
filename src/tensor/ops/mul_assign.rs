use crate::tensor::Tensor;
use std::ops::MulAssign;

impl MulAssign for Tensor {
    fn mul_assign(&mut self, other: Self) {
        // 使用`Mul` trait的`mul`方法来执行乘法，并更新当前张量
        *self = self.clone() * other;
    }
}

impl<'a> MulAssign<&'a Self> for Tensor {
    fn mul_assign(&mut self, other: &'a Self) {
        // 使用`Mul` trait的`mul`方法来执行乘法，并更新当前张量
        *self = self.clone() * other;
    }
}

impl MulAssign<f32> for Tensor {
    fn mul_assign(&mut self, scalar: f32) {
        // 使用`Mul` trait的`mul`方法来执行乘法，并更新当前张量
        *self = self.clone() * scalar;
    }
}

impl<'a> MulAssign<f32> for &'a mut Tensor {
    fn mul_assign(&mut self, scalar: f32) {
        // 使用`Mul` trait的`mul`方法来执行乘法，并更新当前张量
        **self = (*self).clone() * scalar;
    }
}
