use crate::errors::{Operator, TensorError};
use crate::tensor::Tensor;
use std::ops::DivAssign;

impl DivAssign for Tensor {
    fn div_assign(&mut self, other: Tensor) {
        // 检查除数是否包含零值
        assert!(!other.has_zero_value(), "{}", TensorError::DivByZeroElement);
        // 检查是否可以执行除法操作
        if self.is_same_shape(&other) {
            self.data /= &other.data;
        } else if other.is_scalar() {
            let scalar = other.number().unwrap();
            if scalar == 0. {
                panic!("{}", TensorError::DivByZero);
            }
            self.data /= scalar;
        } else {
            panic!(
                "{}",
                TensorError::OperatorError {
                    operator: Operator::DivAssign,
                    tensor1_shape: self.shape().to_vec(),
                    tensor2_shape: other.shape().to_vec(),
                }
            )
        }
    }
}

impl<'a> DivAssign<&'a Tensor> for Tensor {
    fn div_assign(&mut self, other: &'a Tensor) {
        // 检查除数是否包含零值
        assert!(!other.has_zero_value(), "{}", TensorError::DivByZeroElement);
        // 检查是否可以执行除法操作
        if self.is_same_shape(other) {
            self.data /= &other.data;
        } else if other.is_scalar() {
            let scalar = other.number().unwrap();
            if scalar == 0. {
                panic!("{}", TensorError::DivByZero);
            }
            self.data /= scalar;
        } else {
            panic!(
                "{}",
                TensorError::OperatorError {
                    operator: Operator::DivAssign,
                    tensor1_shape: self.shape().to_vec(),
                    tensor2_shape: other.shape().to_vec(),
                }
            )
        }
    }
}

impl DivAssign<f32> for Tensor {
    fn div_assign(&mut self, scalar: f32) {
        if scalar == 0. {
            panic!("{}", TensorError::DivByZero);
        }
        self.data /= scalar;
    }
}

impl<'a> DivAssign<f32> for &'a mut Tensor {
    fn div_assign(&mut self, scalar: f32) {
        if scalar == 0. {
            panic!("{}", TensorError::DivByZero);
        }
        self.data /= scalar;
    }
}
