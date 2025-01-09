/*
 * @Author       : 老董
 * @Date         : 2023-08-17 17:24:24
 * @LastEditors  : 老董
 * @LastEditTime : 2025-01-09 10:58:47
 * @Description  : 张量的减法，实现了两个张量“逐元素”（或张量与纯数）相减的运算，并返回一个新的张量。
 *                 该运算支持以下情况：
 *                 1. 其中一个操作数为纯数而另一个为张量：则返回的张量形状与该张量相同。
 *                 2. 两个操作数均为张量：需保证两个操作数的形状严格一致。
 *                 注意：这里的减法概念与线性代数中的矩阵减法类似，但适用于更高阶的张量。
 */

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
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑f32 -（不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 - f32↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl Sub<f32> for Tensor {
    type Output = Self;

    fn sub(self, scalar: f32) -> Self {
        Self {
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
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 - f32↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 -（不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl Sub for Tensor {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        sub_within_tensors(&self, &other)
    }
}

impl<'a> Sub<&'a Self> for Tensor {
    type Output = Self;

    fn sub(self, other: &'a Self) -> Self {
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
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 -（不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

fn sub_within_tensors(tensor_1: &Tensor, tensor_2: &Tensor) -> Tensor {
    if tensor_1.is_same_shape(tensor_2) {
        Tensor {
            data: &tensor_1.data - &tensor_2.data,
        }
    } else {
        panic!(
            "{}",
            TensorError::OperatorError {
                operator: Operator::Sub,
                tensor1_shape: tensor_1.shape().to_vec(),
                tensor2_shape: tensor_2.shape().to_vec(),
            }
        )
    }
}
