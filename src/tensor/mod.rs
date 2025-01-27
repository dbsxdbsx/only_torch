use ndarray::{Array, ArrayD, ArrayViewD, IxDyn};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;

use serde::{Deserialize, Serialize};

use crate::errors::{ComparisonOperator, TensorError};
mod ops {
    pub mod add;
    pub mod add_assign;
    pub mod div;
    pub mod div_assign;
    pub mod eq;
    pub mod mat_mul;
    pub mod mul;
    pub mod mul_assign;
    pub mod others;
    pub mod sub;
    pub mod sub_assign;
}

pub mod filter;
pub mod image;
pub mod index;
pub mod print;
pub mod property;
pub mod save_load;
pub mod slice;

#[cfg(test)]
mod tests;
/// 定义张量的结构体。其可以是标量、向量、矩阵或更高维度的数组。
/// 注：通过Tensor初始化的都是张量（即使标量也是张量）；
/// 而通常意义上的数字（类型为usize、i32、f64等）就只是纯数（number），在这里不被认为是张量。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    data: ArrayD<f32>,
}

impl Default for Tensor {
    fn default() -> Self {
        Self::uninited(&[])
    }
}

impl Tensor {
    /// 创建一个空（未初始化）的张量--该张量的所有元素值为NaN，请务必之后赋予每个元素具体数值后再使用（虽然不会报错）。
    /// 若为标量，`shape`可以是[]、[1]、[1,1]、[1,1,1]...
    /// 若为向量，`shape`可以是[n]、[1,n]、[n,1]；
    /// 若为矩阵，`shape`可以是[n,m]；
    /// 若为更高维度的数组，`shape`可以是[c,n,m,...]；
    pub fn uninited(shape: &[usize]) -> Self {
        let data = Array::from_elem(IxDyn(shape), f32::NAN);
        Self { data }
    }

    /// 创建一个张量，
    /// 若为标量，`shape`可以是[]、[1]、[1,1]、[1,1,1]...
    /// 若为向量，`shape`可以是[n]、[1,n]、[n,1]；
    /// 若为矩阵，`shape`可以是[n,m]；
    /// 若为更高维度的数组，`shape`可以是[c,n,m,...]；
    /// 注：除了`data`长度为1且shape为`[]`的情况（标量），`data`的长度必须和`shape`中所有元素的乘积相等。
    pub fn new(data: &[f32], shape: &[usize]) -> Self {
        let data = Array::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap();
        Self { data }
    }

    /// 创建一个随机张量，其值在[min, max]的闭区间，若为标量，`shape`可以是[]、[1]、[1,1]、[1,1,1]...
    /// 若为向量，`shape`可以是[n]、[1,n]、[n,1]；
    /// 若为矩阵，`shape`可以是[n,m]；
    /// 若为更高维度的数组，`shape`可以是[c,n,m,...]；
    /// 注：除了`data`长度为1且shape为`[]`的情况（标量），`data`的长度必须和`shape`中所有元素的乘积相等。
    pub fn random(min: f32, max: f32, shape: &[usize]) -> Self {
        let mut rng = rand::thread_rng();
        let data = (0..shape.iter().product::<usize>())
            .map(|_| Uniform::from(min..=max).sample(&mut rng))
            .collect::<Vec<_>>();
        Self::new(&data, shape)
    }

    /// 创建一个全0的张量。
    /// 若为标量，`shape`可以是[]、[1]、[1,1]、[1,1,1]...
    /// 若为向量，`shape`可以是[n]、[1,n]、[n,1]；
    /// 若为矩阵，`shape`可以是[n,m]；
    /// 若为更高维度的数组，`shape`可以是[c,n,m,...]；
    pub fn zeros(shape: &[usize]) -> Self {
        let data = Array::zeros(IxDyn(shape));
        Self { data }
    }

    /// 创建一个含`n`个对角元素的单位矩阵。
    /// 注：n必须大于0，否则会panic。
    pub fn eyes(n: usize) -> Self {
        assert!(
            n > 0,
            "{}",
            TensorError::ValueMustSatisfyComparison {
                value_name: "n".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: 0,
            }
        );
        let data = Array::eye(n);
        let shape = vec![n, n];
        let data = Array::from_shape_vec(IxDyn(&shape), data.into_raw_vec()).unwrap();
        Self { data }
    }

    /// 创建一个服从正态分布的随机张量，其值在指定的均值和标准差范围内。
    /// 若为标量，shape可以是[]、[1,1]、[1,1,1]...；
    /// 若为向量，shape可以是[n]、[1,n]、[n,1]；
    /// 若为矩阵，shape可以是[n,m]；
    /// 若为更高维度的数组，shape可以是[c,n,m,...]。
    pub fn normal(mean: f32, std_dev: f32, shape: &[usize]) -> Self {
        let mut rng = rand::thread_rng();
        let data_len = shape.iter().product::<usize>();
        let mut data = Vec::with_capacity(data_len);

        while data.len() < data_len {
            let u1: f32 = rng.gen();
            let u2: f32 = rng.gen();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            let z0 = (std_dev * r).mul_add(theta.cos(), mean);
            let z1 = (std_dev * r).mul_add(theta.sin(), mean);

            if z0.is_finite() {
                data.push(z0);
            }
            if data.len() < data_len && z1.is_finite() {
                data.push(z1);
            }
        }

        Self::new(&data, shape)
    }

    /// 创建一个包含指定范围内等间隔值的一维张量。
    ///
    /// * `start` - 范围的起始值（包含）
    /// * `end` - 范围的结束值（不包含）
    /// * `step` - 相邻元素之间的间隔（默认为1.0）
    ///
    /// 返回一个新的一维张量。
    pub fn arange(start: f32, end: f32, step: Option<f32>) -> Self {
        let step = step.unwrap_or(if start <= end { 1.0 } else { -1.0 });
        assert!(step != 0.0, "步长不能为零");
        assert!((end - start) * step > 0.0, "步长的符号必须与范围的方向一致");

        let num_elements = ((end - start) / step).abs().ceil() as usize;
        let data: Vec<f32> = (0..num_elements)
            .map(|i| (i as f32).mul_add(step, start))
            .collect();

        Self::new(&data, &[num_elements])
    }

    /// 从快照创建一个新张量
    pub fn from_view(view: ArrayViewD<'_, f32>) -> Self {
        Self {
            data: view.to_owned(),
        }
    }
}
