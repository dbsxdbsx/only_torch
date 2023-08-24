use crate::tensor::Tensor;
use ndarray::{Array, Axis, Zip};
use rand::seq::SliceRandom;
use rand::thread_rng;

impl From<f32> for Tensor {
    /// 实现 From<f32> trait 用于将`f32`类型转换为形状为`[1]`的张量
    fn from(scalar: f32) -> Tensor {
        Tensor::new(&[scalar], &[1])
    }
}

pub trait DotSum<Rhs = Tensor> {
    type Output;
    fn dot_sum(self, rhs: Rhs) -> Self::Output;
}

impl DotSum<Tensor> for f32 {
    type Output = Tensor;
    fn dot_sum(self, rhs: Tensor) -> Self::Output {
        rhs.dot_sum(self)
    }
}

impl Tensor {
    /// 对张量中的所有元素求和并返回一个形状为[1]的标量。
    pub fn sum(&self) -> Tensor {
        let mut value = 0.;
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

    /// 计算张量的均值
    pub fn mean(&self) -> f32 {
        self.data.mean().unwrap()
    }

    /// 计算张量的标准差
    pub fn std_dev(&self) -> f32 {
        self.data.std_axis(ndarray::Axis(0), 0.).mean().unwrap()
    }

    /// 返回一个形状和`self`相同的张量，其中的元素按从小到大的顺序排列
    pub fn order(&self) -> Tensor {
        let flat_data = self.data.clone().into_shape(self.data.len()).unwrap();
        let mut flat_data = flat_data.into_raw_vec();
        flat_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ordered_data = Array::from_shape_vec(self.data.shape(), flat_data).unwrap();
        let ordered_data = ordered_data.into_shape(self.data.shape()).unwrap();
        Tensor { data: ordered_data }
    }

    /// 打乱张量中的元素顺序
    pub fn shuffle(&self, axis: Option<usize>) -> Tensor {
        let mut shuffled_data = self.data.clone();
        let mut rng = thread_rng();

        match axis {
            Some(axis) => {
                let axis = Axis(axis);
                let mut chunks: Vec<_> = shuffled_data
                    .axis_iter(axis)
                    .map(|c| c.to_owned())
                    .collect();
                chunks.shuffle(&mut rng);
                let mut new_data = Array::zeros(shuffled_data.raw_dim());
                for (i, chunk) in chunks.into_iter().enumerate() {
                    let mut slice = new_data.index_axis_mut(axis, i);
                    slice.assign(&chunk);
                }
                shuffled_data = new_data;
            }
            None => {
                let mut flat_data = shuffled_data.into_shape(self.data.len()).unwrap();
                flat_data.as_slice_mut().unwrap().shuffle(&mut rng);
                shuffled_data = flat_data.into_shape(self.data.shape()).unwrap();
            }
        }

        Tensor {
            data: shuffled_data,
        }
    }
}
