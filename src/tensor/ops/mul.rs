/*
 * @Author       : 老董
 * @Date         : 2023-08-17 17:24:24
 * @LastEditors  : 老董
 * @LastEditTime : 2025-01-09 11:08:24
 * @Description  : 张量的乘法，实现了两个张量“逐元素”（或张量与纯数）相乘的运算，并返回一个新的张量。
 *                 该运算支持以下情况：
 *                 1. 其中一个操作数为纯数而另一个为张量：则返回的张量形状与该张量相同。
 *                 2. 两个操作数均为张量：需保证两个操作数的形状严格一致。
 *                 注意：这里的乘法概念与线性代数中的矩阵乘法（点积/点乘、叉积/叉乘）有所不同，在这里更类似于哈达玛积（Hadamard product）与数乘的结合。
 *                 参考：https://www.jianshu.com/p/9165e3264ced
 */

use crate::errors::{Operator, TensorError};
use crate::tensor::Tensor;
use std::ops::Mul;

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓f32 *（不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl Mul<Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, tensor: Tensor) -> Tensor {
        Tensor {
            data: self * &tensor.data,
        }
    }
}
impl<'a> Mul<&'a Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, tensor: &'a Tensor) -> Tensor {
        Tensor {
            data: self * &tensor.data,
        }
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑f32 *（不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 * f32↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl Mul<f32> for Tensor {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self {
        Self {
            data: &self.data * scalar,
        }
    }
}
impl<'a> Mul<f32> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, scalar: f32) -> Tensor {
        Tensor {
            data: &self.data * scalar,
        }
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 * f32↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 *（不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl Mul for Tensor {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        mul_within_tensors(&self, &other)
    }
}

impl<'a> Mul<&'a Self> for Tensor {
    type Output = Self;

    fn mul(self, other: &'a Self) -> Self {
        mul_within_tensors(&self, other)
    }
}

impl<'a> Mul<Tensor> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Tensor {
        mul_within_tensors(self, &other)
    }
}

impl<'a, 'b> Mul<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, other: &'b Tensor) -> Tensor {
        mul_within_tensors(self, other)
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 *（不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

fn mul_within_tensors(tensor_1: &Tensor, tensor_2: &Tensor) -> Tensor {
    if tensor_1.is_same_shape(tensor_2) {
        Tensor {
            data: &tensor_1.data * &tensor_2.data,
        }
    } else {
        panic!(
            "{}",
            TensorError::OperatorError {
                operator: Operator::Mul,
                tensor1_shape: tensor_1.shape().to_vec(),
                tensor2_shape: tensor_2.shape().to_vec(),
            }
        )
    }
}
