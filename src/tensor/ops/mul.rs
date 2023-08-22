/*
 * @Author       : 老董
 * @Date         : 2023-08-17 17:24:24
 * @LastEditors  : 老董
 * @LastEditTime : 2023-08-22 12:09:59
 * @Description  : 张量的乘法，实现了张量与标量的乘法以及两个张量“逐元素”相乘的运算，并返回一个新的张量。
 *                 乘法运算支持以下情况：
 *                 1. 若两个张量的形状严格一致, 则相乘后的张量形状不变；
 *                 2. 若其中一个张量为标量或纯数---统称为一阶张量。
 *                  2.1 若两个都是一阶张量，则相乘后返回一个标量，其形状为[1];
 *                  2.2 若其中一个是二阶以上的张量，则相乘后的形状为该张量的形状；
 *                 注意：这里的乘法概念与线性代数中的矩阵乘法（点积、叉积）不同，在这里其更类似于哈达玛积（Hadamard product）与数乘的结合。
 *                 参考：https://www.jianshu.com/p/9165e3264ced
 */

use crate::tensor::Tensor;
use std::ops::Mul;

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓f32 +（不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
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
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑f32 +（不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 * f32↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
impl Mul<f32> for Tensor {
    type Output = Tensor;

    fn mul(self, scalar: f32) -> Tensor {
        Tensor {
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
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 * f32↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 * （不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Tensor {
        mul_within_tensors(&self, &other)
    }
}

impl<'a> Mul<&'a Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, other: &'a Tensor) -> Tensor {
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
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 * （不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

fn mul_within_tensors(tensor_1: &Tensor, tensor_2: &Tensor) -> Tensor {
    let data = if tensor_1.is_scalar() && tensor_2.is_scalar() {
        return Tensor::new(
            &[tensor_1.to_number().unwrap() * tensor_2.to_number().unwrap()],
            &[1],
        );
    } else if tensor_1.is_same_shape(tensor_2) {
        &tensor_1.data * &tensor_2.data
    } else if tensor_1.is_scalar() {
        tensor_1.to_number().unwrap() * &tensor_2.data
    } else if tensor_2.is_scalar() {
        &tensor_1.data * tensor_2.to_number().unwrap()
    } else {
        panic!(
             "形状不一致且两个张量没有一个是标量，故无法相乘：第一个张量的形状为{:?}，第二个张量的形状为{:?}",
             tensor_1.shape(),
             tensor_2.shape()
         )
    };

    Tensor { data }
}
