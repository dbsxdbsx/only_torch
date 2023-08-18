use crate::tensor::Tensor;
use ndarray::Zip;
use std::cmp::PartialEq;

impl From<f32> for Tensor {
    /// 实现 From<f32> trait 用于将`f32`类型转换为形状为`[1]`的张量
    fn from(scalar: f32) -> Self {
        Tensor::new(&[scalar], &[1])
    }
}

pub trait DotSum<Rhs = Self> {
    type Output;
    fn dot_sum(self, rhs: Rhs) -> Self::Output;
}

impl DotSum<Tensor> for f32 {
    type Output = Tensor;
    fn dot_sum(self, rhs: Tensor) -> Self::Output {
        rhs.dot_sum(self)
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Tensor {
    /// 对张量中的所有元素求和并返回一个形状为[1]的标量。
    pub fn sum(&self) -> Tensor {
        let mut value = 0.0;
        Zip::from(&self.data).for_each(|a| value += a);
        Tensor::from(value)
    }

    /// 对两个张量(或其中一个是标量或纯数)进行逐元素相乘，然后对结果求和，并返回一个形状为[1]的标量。
    /// 这里`dot_sum`（点积和）的概念拓展自线性代数中向量内积的概念，但适用性更广---
    /// 这里只需保证两个张量的形状严格一致，或其中一个张量为标量即可运算
    /// 参考：https://www.jianshu.com/p/9165e3264ced
    pub fn dot_sum<T: Into<Tensor>>(&self, other: T) -> Tensor {
        let other = other.into();
        if !self.is_same_shape(&other) && !self.is_scalar() && !other.is_scalar() {
            panic!(
                "形状不一致且两个张量没有一个是标量，故无法进行点积和：第一个张量的形状为{:?}，第二个张量的形状为{:?}",
                self.shape(),
                other.shape()
            );
        }

        let product_tensor = self.clone() * other;
        product_tensor.sum()
    }
}
