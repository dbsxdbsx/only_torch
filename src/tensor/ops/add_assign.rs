use crate::errors::{Operator, TensorError};
use crate::tensor::Tensor;
use std::ops::AddAssign;

impl AddAssign for Tensor {
    fn add_assign(&mut self, other: Self) {
        self.add_assign(&other);
    }
}

impl<'a> AddAssign<&'a Self> for Tensor {
    fn add_assign(&mut self, other: &'a Self) {
        if self.is_same_shape(other) {
            self.data += &other.data;
        } else {
            panic!(
                "{}",
                TensorError::OperatorError {
                    operator: Operator::Add,
                    tensor1_shape: self.shape().to_vec(),
                    tensor2_shape: other.shape().to_vec(),
                }
            )
        }
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
