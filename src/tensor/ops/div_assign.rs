use crate::tensor::Tensor;
use std::ops::DivAssign;

impl DivAssign for Tensor {
    fn div_assign(&mut self, other: Self) {
        // 使用`Div` trait的`div`方法来执行除法，并更新当前张量
        *self = self.clone() / other;
    }
}

impl<'a> DivAssign<&'a Self> for Tensor {
    fn div_assign(&mut self, other: &'a Self) {
        // 使用`Div` trait的`div`方法来执行除法，并更新当前张量
        *self = self.clone() / other;
    }
}

impl DivAssign<f32> for Tensor {
    fn div_assign(&mut self, scalar: f32) {
        // 使用`Div` trait的`div`方法来执行除法，并更新当前张量
        *self = self.clone() / scalar;
    }
}

impl<'a> DivAssign<f32> for &'a mut Tensor {
    fn div_assign(&mut self, scalar: f32) {
        // 使用`Div` trait的`div`方法来执行除法，并更新当前张量
        **self = (*self).clone() / scalar;
    }
}
