/*
 * @Author       : 老董
 * @Date         : 2023-08-17 17:24:24
 * @LastEditors  : 老董
 * @LastEditTime : 2025-01-09 11:36:25
 * @Description  : 张量的除法，实现了两个张量"逐元素"（或张量与纯数）相除的运算，并返回一个新的张量。
 *                 该运算支持以下情况：
 *                 1. 其中一个操作数为纯数而另一个为张量：则返回的张量形状与该张量相同。
 *                 2. 两个操作数均为张量：需保证两个操作数的形状严格一致。
 *                 3. 无论是情况1还是2，作为除数的操作数（第二个操作数）都不能为0或包含0元素。
 *                 注意：这里的除法概念与线性代数中的矩阵除法有所不同，在这里更类似于哈达玛除法（Hadamard division）与数除的结合。
 */

use crate::errors::{Operator, TensorError};
use crate::tensor::Tensor;
use std::ops::Div;

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓f32 /（不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl Div<Tensor> for f32 {
    type Output = Tensor;

    fn div(self, tensor: Tensor) -> Tensor {
        assert!(
            !tensor.has_zero_value(),
            "{}",
            TensorError::DivByZeroElement
        );
        Tensor {
            data: self / &tensor.data,
        }
    }
}
impl<'a> Div<&'a Tensor> for f32 {
    type Output = Tensor;

    fn div(self, tensor: &'a Tensor) -> Tensor {
        assert!(
            !tensor.has_zero_value(),
            "{}",
            TensorError::DivByZeroElement
        );
        Tensor {
            data: self / &tensor.data,
        }
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑f32 /（不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 / f32↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl Div<f32> for Tensor {
    type Output = Self;

    fn div(self, scalar: f32) -> Self {
        assert!(!(scalar == 0.), "{}", TensorError::DivByZero);
        Self {
            data: &self.data / scalar,
        }
    }
}
impl<'a> Div<f32> for &'a Tensor {
    type Output = Tensor;

    fn div(self, scalar: f32) -> Tensor {
        assert!(!(scalar == 0.), "{}", TensorError::DivByZero);
        Tensor {
            data: &self.data / scalar,
        }
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 / f32↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 /（不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl Div for Tensor {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        div_within_tensors(&self, &other)
    }
}

impl<'a> Div<&'a Self> for Tensor {
    type Output = Self;

    fn div(self, other: &'a Self) -> Self {
        div_within_tensors(&self, other)
    }
}

impl<'a> Div<Tensor> for &'a Tensor {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Tensor {
        div_within_tensors(self, &other)
    }
}

impl<'a, 'b> Div<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn div(self, other: &'b Tensor) -> Tensor {
        div_within_tensors(self, other)
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 /（不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

fn div_within_tensors(tensor_1: &Tensor, tensor_2: &Tensor) -> Tensor {
    assert!(
        !tensor_2.has_zero_value(),
        "{}",
        TensorError::DivByZeroElement
    );

    if tensor_1.is_same_shape(tensor_2) {
        Tensor {
            data: &tensor_1.data / &tensor_2.data,
        }
    } else {
        panic!(
            "{}",
            TensorError::OperatorError {
                operator: Operator::Div,
                tensor1_shape: tensor_1.shape().to_vec(),
                tensor2_shape: tensor_2.shape().to_vec(),
            }
        )
    }
}
