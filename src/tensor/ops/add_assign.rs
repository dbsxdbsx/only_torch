use crate::errors::{Operator, TensorError};
use crate::tensor::Tensor;
use std::ops::AddAssign;

impl AddAssign for Tensor {
    fn add_assign(&mut self, other: Tensor) {
        // 检查是否可以执行加法操作
        if self.is_same_shape(&other) {
            self.data += &other.data;
        } else if other.is_scalar() {
            self.data += other.number().unwrap();
        } else {
            panic!(
                "{}",
                TensorError::OperatorError {
                    operator: Operator::AddAssign,
                    tensor1_shape: self.shape().to_vec(),
                    tensor2_shape: other.shape().to_vec(),
                }
            )
        }
    }
}

impl<'a> AddAssign<&'a Tensor> for Tensor {
    fn add_assign(&mut self, other: &'a Tensor) {
        // 检查是否可以执行加法操作
        if self.is_same_shape(other) {
            self.data += &other.data;
        } else if other.is_scalar() {
            self.data += other.number().unwrap();
        } else {
            panic!(
                "{}",
                TensorError::OperatorError {
                    operator: Operator::AddAssign,
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

impl<'a> AddAssign<f32> for &'a mut Tensor {
    fn add_assign(&mut self, scalar: f32) {
        self.data += scalar;
    }
}
