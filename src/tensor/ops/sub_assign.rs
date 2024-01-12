use crate::tensor::Tensor;
use std::ops::SubAssign;

impl SubAssign for Tensor {
    fn sub_assign(&mut self, other: Tensor) {
        // 使用`Sub` trait的`sub`方法来执行减法，并更新当前张量
        *self = self.clone() - other;
    }
}

impl<'a> SubAssign<&'a Tensor> for Tensor {
    fn sub_assign(&mut self, other: &'a Tensor) {
        // 使用`Sub` trait的`sub`方法来执行减法，并更新当前张量
        *self = self.clone() - other;
    }
}

impl SubAssign<f32> for Tensor {
    fn sub_assign(&mut self, scalar: f32) {
        // 使用`Sub` trait的`sub`方法来执行减法，并更新当前张量
        *self = self.clone() - scalar;
    }
}

impl<'a> SubAssign<f32> for &'a mut Tensor {
    fn sub_assign(&mut self, scalar: f32) {
        // 使用`Sub` trait的`sub`方法来执行减法，并更新当前张量
        **self = (*self).clone() - scalar;
    }
}
