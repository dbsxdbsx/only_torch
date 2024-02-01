use crate::errors::{Operator, TensorError};
use crate::tensor::Tensor;
use std::ops::Sub;

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓f32 -（不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl Sub<Tensor> for f32 {
    type Output = Tensor;

    fn sub(self, tensor: Tensor) -> Tensor {
        Tensor {
            data: self - &tensor.data,
        }
    }
}
impl<'a> Sub<&'a Tensor> for f32 {
    type Output = Tensor;

    fn sub(self, tensor: &'a Tensor) -> Tensor {
        Tensor {
            data: self - &tensor.data,
        }
    }
}
//*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑f32 -（不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 - f32↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl Sub<f32> for Tensor {
    type Output = Tensor;

    fn sub(self, scalar: f32) -> Tensor {
        Tensor {
            data: &self.data - scalar,
        }
    }
}
impl<'a> Sub<f32> for &'a Tensor {
    type Output = Tensor;

    fn sub(self, scalar: f32) -> Tensor {
        Tensor {
            data: &self.data - scalar,
        }
    }
}
//*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 - f32↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 - （不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Tensor {
        sub_within_tensors(&self, &other)
    }
}

impl<'a> Sub<&'a Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, other: &'a Tensor) -> Tensor {
        sub_within_tensors(&self, other)
    }
}

impl<'a> Sub<Tensor> for &'a Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Tensor {
        sub_within_tensors(self, &other)
    }
}

impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn sub(self, other: &'b Tensor) -> Tensor {
        sub_within_tensors(self, other)
    }
}
//*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 - （不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

fn sub_within_tensors(tensor_1: &Tensor, tensor_2: &Tensor) -> Tensor {
    let data = if tensor_1.is_scalar() && tensor_2.is_scalar() {
        return Tensor::new(
            &[tensor_1.number().unwrap() - tensor_2.number().unwrap()],
            &[1],
        );
    } else if tensor_1.is_same_shape(tensor_2) {
        &tensor_1.data - &tensor_2.data
    } else if tensor_1.is_scalar() {
        tensor_1.number().unwrap() - &tensor_2.data
    } else if tensor_2.is_scalar() {
        &tensor_1.data - tensor_2.number().unwrap()
    } else {
        panic!(
            "{}",
            TensorError::OperatorError {
                operator: Operator::Sub,
                tensor1_shape: tensor_1.shape().to_vec(),
                tensor2_shape: tensor_2.shape().to_vec(),
            }
        )
    };

    Tensor { data }
}
