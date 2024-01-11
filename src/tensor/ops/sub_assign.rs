use crate::errors::{Operator, TensorError};
use crate::tensor::Tensor;
use std::ops::SubAssign;

impl SubAssign for Tensor {
    fn sub_assign(&mut self, other: Tensor) {
        // 检查是否可以执行减法操作
        if self.is_same_shape(&other) {
            self.data -= &other.data;
        } else if other.is_scalar() {
            self.data -= other.number().unwrap();
        } else {
            panic!(
                "{}",
                TensorError::OperatorError {
                    operator: Operator::SubAssign,
                    tensor1_shape: self.shape().to_vec(),
                    tensor2_shape: other.shape().to_vec(),
                }
            )
        }
    }
}

impl<'a> SubAssign<&'a Tensor> for Tensor {
    fn sub_assign(&mut self, other: &'a Tensor) {
        // 检查是否可以执行减法操作
        if self.is_same_shape(other) {
            self.data -= &other.data;
        } else if other.is_scalar() {
            self.data -= other.number().unwrap();
        } else {
            panic!(
                "{}",
                TensorError::OperatorError {
                    operator: Operator::SubAssign,
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

impl<'a> SubAssign<f32> for &'a mut Tensor {
    fn sub_assign(&mut self, scalar: f32) {
        self.data -= scalar;
    }
}
