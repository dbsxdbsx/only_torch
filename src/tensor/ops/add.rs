/*
 * @Author       : 老董
 * @Date         : 2023-08-17 17:24:24
 * @LastEditors  : 老董
 * @LastEditTime : 2025-01-09 10:58:47
 * @Description  : 张量的加法，实现了两个张量“逐元素”（或张量与纯数）相加的运算，并返回一个新的张量。
 *                 该运算支持以下情况：
 *                 1. 其中一个操作数为纯数而另一个为张量：则返回的张量形状与该张量相同。
 *                 2. 两个操作数均为张量：需保证两个操作数的形状严格一致。
 *                 注意：这里的加法概念与线性代数中的矩阵加法类似，但适用于更高阶的张量。
 */

use crate::errors::{Operator, TensorError};
use crate::tensor::Tensor;
use std::ops::Add;

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓f32 +（不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl Add<Tensor> for f32 {
    type Output = Tensor;

    fn add(self, tensor: Tensor) -> Tensor {
        Tensor {
            data: self + &tensor.data,
        }
    }
}
impl<'a> Add<&'a Tensor> for f32 {
    type Output = Tensor;

    fn add(self, tensor: &'a Tensor) -> Tensor {
        Tensor {
            data: self + &tensor.data,
        }
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑f32 +（不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 + f32↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl Add<f32> for Tensor {
    type Output = Self;

    fn add(self, scalar: f32) -> Self {
        Self {
            data: &self.data + scalar,
        }
    }
}
impl<'a> Add<f32> for &'a Tensor {
    type Output = Tensor;

    fn add(self, scalar: f32) -> Tensor {
        Tensor {
            data: &self.data + scalar,
        }
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 + f32↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 +（不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl Add for Tensor {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        add_within_tensors(&self, &other)
    }
}

impl<'a> Add<&'a Self> for Tensor {
    type Output = Self;

    fn add(self, other: &'a Self) -> Self {
        add_within_tensors(&self, other)
    }
}

impl<'a> Add<Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Tensor {
        add_within_tensors(self, &other)
    }
}

impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, other: &'b Tensor) -> Tensor {
        add_within_tensors(self, other)
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 +（不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

fn add_within_tensors(tensor_1: &Tensor, tensor_2: &Tensor) -> Tensor {
    if tensor_1.is_same_shape(tensor_2) {
        Tensor {
            data: &tensor_1.data + &tensor_2.data,
        }
    } else {
        panic!(
            "{}",
            TensorError::OperatorError {
                operator: Operator::Add,
                tensor1_shape: tensor_1.shape().to_vec(),
                tensor2_shape: tensor_2.shape().to_vec(),
            }
        )
    }
}
