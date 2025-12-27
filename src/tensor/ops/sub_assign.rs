use crate::errors::{Operator, TensorError};
use crate::tensor::Tensor;
use std::ops::SubAssign;

impl SubAssign for Tensor {
    fn sub_assign(&mut self, other: Self) {
        self.sub_assign(&other);
    }
}

impl<'a> SubAssign<&'a Self> for Tensor {
    fn sub_assign(&mut self, other: &'a Self) {
        if self.is_same_shape(other) {
            self.data -= &other.data;
        } else {
            panic!(
                "{}",
                TensorError::OperatorError {
                    operator: Operator::Sub,
                    tensor1_shape: self.shape().to_vec(),
                    tensor2_shape: other.shape().to_vec(),
                }
            )
        }
    }
}

impl SubAssign<f32> for Tensor {
    fn sub_assign(&mut self, scalar: f32) {
        self.data -= scalar;
    }
}

impl SubAssign<f32> for &mut Tensor {
    fn sub_assign(&mut self, scalar: f32) {
        self.data -= scalar;
    }
}
