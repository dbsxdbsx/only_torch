use ndarray::{Array, IxDyn};
use rand::distributions::{Distribution, Uniform};

mod ops {
    pub mod add;
    pub mod div;
    pub mod eq;
    pub mod mul;
    pub mod others;
    pub mod sub;
}
mod print;

#[cfg(test)]
mod tests;

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
        assert!(n >= 2, "n必须大于等于2");
        let data = Array::eye(n);
        let shape = vec![n, n];
        let data = Array::from_shape_vec(IxDyn(&shape), data.into_raw_vec()).unwrap();
        Tensor { data }
    }

    /// 若为向量，`shape`可以是[n]、[1,n]、[n,1]；
    /// 若为矩阵，`shape`可以是[n,m]；
    /// 若为更高维度的数组，`shape`可以是[c,n,m,...]。
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    // TODO：reshape,如将单位矩阵reshape为向量，或者reshape到高阶张量

    /// 判断两个张量的形状是否严格一致。如：形状为 [1, 4]，[1, 4]和[4]是不一致的，会返回false
    pub fn is_same_shape(&self, other: &Tensor) -> bool {
        self.shape() == other.shape()
    }

    /// 判断张量是否为标量
    pub fn is_scalar(&self) -> bool {
        self.shape().is_empty() || self.shape().iter().all(|x| *x == 1)
    }

    pub fn to_number(&self) -> Option<f32> {
        if self.is_scalar() {
            let shape = self.shape();
            let index_array = self.generate_index_array(shape);
            Some(self.data[&index_array[..]])
        } else {
            None
        }
    }

    fn generate_index_array(&self, shape: &[usize]) -> Vec<usize> {
        shape.iter().map(|_| 0).collect()
    }
}
