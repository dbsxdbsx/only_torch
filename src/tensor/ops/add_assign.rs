use crate::tensor::Tensor;
use std::ops::AddAssign;

impl AddAssign for Tensor {
    fn add_assign(&mut self, other: Tensor) {
        // 使用`Add` trait的`add`方法来执行加法，并更新当前张量
        *self = self.clone() + other;
    }
}

impl<'a> AddAssign<&'a Tensor> for Tensor {
    fn add_assign(&mut self, other: &'a Tensor) {
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

impl<'a> AddAssign<f32> for &'a mut Tensor {
    fn add_assign(&mut self, scalar: f32) {
        // 使用`Add` trait的`add`方法来执行加法，并更新当前张量
        **self = (*self).clone() + scalar;
    }
}
