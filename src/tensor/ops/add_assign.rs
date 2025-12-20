use crate::tensor::Tensor;
use std::ops::AddAssign;

impl AddAssign for Tensor {
    fn add_assign(&mut self, other: Self) {
        // 使用`Add` trait的`add`方法来执行加法，并更新当前张量
        *self = self.clone() + other;
    }
}

impl<'a> AddAssign<&'a Self> for Tensor {
    fn add_assign(&mut self, other: &'a Self) {
        // 使用`Add` trait的`add`方法来执行加法，并更新当前张量
        *self = self.clone() + other;
    }
}

impl AddAssign<f32> for Tensor {
    fn add_assign(&mut self, scalar: f32) {
        // 使用`Add` trait的`add`方法来执行加法，并更新当前张量
        *self = self.clone() + scalar;
    }
}

impl AddAssign<f32> for &mut Tensor {
    fn add_assign(&mut self, scalar: f32) {
        // 使用`Add` trait的`add`方法来执行加法，并更新当前张量
        **self = (*self).clone() + scalar;
    }
}
