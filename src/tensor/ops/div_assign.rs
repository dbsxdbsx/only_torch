use crate::errors::{Operator, TensorError};
use crate::tensor::Tensor;
use std::ops::DivAssign;

impl DivAssign for Tensor {
    fn div_assign(&mut self, other: Self) {
        self.div_assign(&other);
    }
}

impl<'a> DivAssign<&'a Self> for Tensor {
    fn div_assign(&mut self, other: &'a Self) {
        assert!(
            !other.has_zero_value(),
            "{}",
            TensorError::DivByZeroElement
        );
        if self.is_same_shape(other) {
            self.data /= &other.data;
        } else {
            panic!(
                "{}",
                TensorError::OperatorError {
                    operator: Operator::Div,
                    tensor1_shape: self.shape().to_vec(),
                    tensor2_shape: other.shape().to_vec(),
                }
            )
        }
    }
}

impl DivAssign<f32> for Tensor {
    fn div_assign(&mut self, scalar: f32) {
        assert!(!(scalar == 0.), "{}", TensorError::DivByZero);
        self.data /= scalar;
    }
}

impl DivAssign<f32> for &mut Tensor {
    fn div_assign(&mut self, scalar: f32) {
        assert!(!(scalar == 0.), "{}", TensorError::DivByZero);
        self.data /= scalar;
    }
}
