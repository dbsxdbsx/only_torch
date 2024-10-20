use crate::errors::{Operator, TensorError};

use crate::tensor::Tensor;
use ndarray::{Array, Axis, Zip};
use rand::seq::SliceRandom;
use rand::thread_rng;

impl From<f32> for Tensor {
    /// 实现 From<f32> trait 用于将`f32`类型转换为形状为`[1]`的张量
    fn from(scalar: f32) -> Self {
        Self::new(&[scalar], &[1])
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
    pub fn sum(&self) -> Self {
        let mut value = 0.;
        Zip::from(&self.data).for_each(|a| value += a);
        Self::from(value)
    }

    /// 对两个张量(或其中一个是标量或纯数)进行逐元素相乘，然后对结果求和，并返回一个形状为[1]的标量。
    /// 这里`dot_sum`（点积和）的概念拓展自线性代数中向量内积的概念，但适用性更广---
    /// 这里只需保证两个张量的形状严格一致，或其中一个张量为标量即可运算
    /// 参考：https://www.jianshu.com/p/9165e3264ced
    pub fn dot_sum<T: Into<Self>>(&self, other: T) -> Self {
        let other = other.into();
        assert!(
            !(!self.is_same_shape(&other) && !self.is_scalar() && !other.is_scalar()),
            "{}",
            TensorError::OperatorError {
                operator: Operator::DotSum,
                tensor1_shape: self.shape().to_vec(),
                tensor2_shape: other.shape().to_vec(),
            }
        );

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

    /// 不改变形状情况下，将张量的元素按从小到大的顺序排列，并将其返回（不影响原张量）
    pub fn order(&self) -> Self {
        let flat_data = self.data.view().into_shape(self.data.len()).unwrap();
        let mut sorted_data = flat_data.as_slice().unwrap().to_owned();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ordered_data = Array::from_shape_vec(self.data.shape(), sorted_data).unwrap();
        Self { data: ordered_data }
    }

    /// 不改变形状情况下，将张量的元素按从小到大的顺序排列（影响原张量）
    pub fn order_mut(&mut self) {
        let flat_len = self.data.len();
        let mut flat_data = self.data.view_mut().into_shape(flat_len).unwrap();
        let flat_data_slice = flat_data.as_slice_mut().unwrap();
        flat_data_slice.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.data = flat_data
            .to_owned()
            .into_shape(self.data.shape().to_owned())
            .unwrap();
    }

    /// 打乱张量中的元素顺序，并将其返回（不影响原张量）
    pub fn shuffle(&self, axis: Option<usize>) -> Self {
        let mut shuffled_data = self.data.clone();
        let mut rng = thread_rng();

        if let Some(axis) = axis {
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
        } else {
            let mut flat_data = shuffled_data.into_shape(self.data.len()).unwrap();
            flat_data.as_slice_mut().unwrap().shuffle(&mut rng);
            shuffled_data = flat_data.into_shape(self.data.shape()).unwrap();
        }

        Self {
            data: shuffled_data,
        }
    }

    /// 打乱张量中的元素顺序（影响原张量）
    pub fn shuffle_mut(&mut self, axis: Option<usize>) {
        let mut rng = thread_rng();

        if let Some(axis) = axis {
            let axis = Axis(axis);
            let mut chunks = self
                .data
                .axis_iter(axis)
                .map(|c| c.to_owned())
                .collect::<Vec<_>>();
            chunks.shuffle(&mut rng);
            for (i, chunk) in chunks.into_iter().enumerate() {
                let mut slice = self.data.index_axis_mut(axis, i);
                slice.assign(&chunk);
            }
        } else {
            let flat_len = self.data.len();
            let mut flat_data = self.data.view_mut().into_shape(flat_len).unwrap();
            let flat_data_slice = flat_data.as_slice_mut().unwrap();
            flat_data_slice.shuffle(&mut rng);
            self.data = flat_data
                .to_owned()
                .into_shape(self.data.shape().to_owned())
                .unwrap();
        }
    }
}
