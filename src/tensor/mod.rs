use ndarray::{Array, IxDyn};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;

use crate::errors::{ComparisonOperator, TensorError};

mod ops {
    pub mod add;
    pub mod div;
    pub mod eq;
    pub mod mul;
    pub mod others;
    pub mod sub;
}

mod index;
mod print;
mod shape;

#[cfg(test)]
pub mod tests;
/// 定义张量的结构体。其可以是标量、向量、矩阵或更高维度的数组。
/// 注：只要通Tensor初始化的都是张量（即使标量也是张量）；
/// 而通常意义上的数字（类型为usize、i32、f64等）就只是纯数（number），在这里不被认为是张量。
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Array<f32, IxDyn>,
}

impl Tensor {
    /// 创建一个张量，若为标量，`shape`可以是[]、[1]、[1,1]、[1,1,1]...
    /// 若为向量，`shape`可以是[n]、[1,n]、[n,1]；
    /// 若为矩阵，`shape`可以是[n,m]；
    /// 若为更高维度的数组，`shape`可以是[c,n,m,...]；
    /// 注：除了`data`长度为1且shape为`[]`的情况（标量），`data`的长度必须和`shape`中所有元素的乘积相等。
    pub fn new(data: &[f32], shape: &[usize]) -> Tensor {
        let data = Array::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap();
        Tensor { data }
    }

    /// 创建一个随机张量，其值在[min, max]的闭区间，若为标量，`shape`可以是[]、[1]、[1,1]、[1,1,1]...
    /// 若为向量，`shape`可以是[n]、[1,n]、[n,1]；
    /// 若为矩阵，`shape`可以是[n,m]；
    /// 若为更高维度的数组，`shape`可以是[c,n,m,...]；
    /// 注：除了`data`长度为1且shape为`[]`的情况（标量），`data`的长度必须和`shape`中所有元素的乘积相等。
    pub fn new_random(min: f32, max: f32, shape: &[usize]) -> Tensor {
        let mut rng = rand::thread_rng();
        let data = (0..shape.iter().product::<usize>())
            .map(|_| Uniform::from(min..=max).sample(&mut rng))
            .collect::<Vec<_>>();
        Tensor::new(&data, shape)
    }

    /// 创建一个含`n`个对角元素的单位矩阵。
    /// n必须大于等于2，否则会panic。
    pub fn new_eye(n: usize) -> Tensor {
        assert!(
            n >= 2,
            "{}",
            TensorError::ValueMustSatisfyComparison {
                value_name: "n".to_string(),
                operator: ComparisonOperator::GreaterOrEqual,
                threshold: 2,
            }
        );
        let data = Array::eye(n);
        let shape = vec![n, n];
        let data = Array::from_shape_vec(IxDyn(&shape), data.into_raw_vec()).unwrap();
        Tensor { data }
    }

    /// 创建一个服从正态分布的随机张量，其值在指定的均值和标准差范围内。
    /// 若为标量，shape可以是[]、[1,1]、[1,1,1]...；
    /// 若为向量，shape可以是[n]、[1,n]、[n,1]；
    /// 若为矩阵，shape可以是[n,m]；
    /// 若为更高维度的数组，shape可以是[c,n,m,...]。
    pub fn new_normal(mean: f32, std_dev: f32, shape: &[usize]) -> Tensor {
        let mut rng = rand::thread_rng();
        let data_len = shape.iter().product::<usize>();
        let mut data = Vec::with_capacity(data_len);

        while data.len() < data_len {
            let u1: f32 = rng.gen();
            let u2: f32 = rng.gen();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            let z0 = mean + std_dev * r * theta.cos();
            let z1 = mean + std_dev * r * theta.sin();

            if z0.is_finite() {
                data.push(z0);
            }
            if data.len() < data_len && z1.is_finite() {
                data.push(z1);
            }
        }

        Tensor::new(&data, shape)
    }
}

// 私有方法
impl Tensor {
    fn has_zero_value(&self) -> bool {
        self.data.iter().any(|&x| x == 0.)
    }

    fn generate_index_array(&self, shape: &[usize]) -> Vec<usize> {
        shape.iter().map(|_| 0).collect()
    }
}
