use crate::errors::{Operator, TensorError};
use crate::tensor::Tensor;
use std::ops::MulAssign;

impl MulAssign for Tensor {
    fn mul_assign(&mut self, other: Self) {
        self.mul_assign(&other);
    }
}

impl<'a> MulAssign<&'a Self> for Tensor {
    fn mul_assign(&mut self, other: &'a Self) {
        if self.is_same_shape(other) {
            self.data *= &other.data;
        } else {
            panic!(
                "{}",
                TensorError::OperatorError {
                    operator: Operator::Mul,
                    tensor1_shape: self.shape().to_vec(),
                    tensor2_shape: other.shape().to_vec(),
                }
            )
        }
    }
}

impl MulAssign<f32> for Tensor {
    fn mul_assign(&mut self, scalar: f32) {
        self.data *= scalar;
    }
}

impl MulAssign<f32> for &mut Tensor {
    fn mul_assign(&mut self, scalar: f32) {
        self.data *= scalar;
    }
}
