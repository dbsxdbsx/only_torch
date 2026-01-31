/*
 * @Author       : 老董
 * @Date         : 2023-10-21 03:22:26
 * @LastEditors  : 老董
 * @LastEditTime : 2025-01-13 21:00:38
 * @Description  : 本模块包含一些常用的张量非四则运算的常用方法，包含转置及一些会改变形状的方法
 */

use std::collections::HashSet;

use crate::errors::{Operator, TensorError};

use crate::tensor::Tensor;
use ndarray::{Array, Axis, Dimension, IxDyn, Zip};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::thread_rng;

impl From<f32> for Tensor {
    /// 实现 From<f32> trait 用于将`f32`类型转换为形状为`[1,1]`的张量
    fn from(scalar: f32) -> Self {
        Self::new(&[scalar], &[1, 1])
    }
}

#[allow(dead_code)]
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

// 为了同时支持Tensor和f32，定义一个trait
pub trait TensorType {
    fn is_tensor() -> bool;
}

// 为 Tensor 实现这个 trait
impl TensorType for Tensor {
    fn is_tensor() -> bool {
        true
    }
}

// 为 f32实现这个 trait
impl TensorType for f32 {
    fn is_tensor() -> bool {
        false
    }
}

// 为Tensor引用类型实现这个 trait
impl TensorType for &Tensor {
    fn is_tensor() -> bool {
        true
    }
}

// 为f32引用类型实现这个 trait
impl TensorType for &f32 {
    fn is_tensor() -> bool {
        false
    }
}

// 为Tensor引用类型实现 Into<Tensor> trait
impl<'a> From<&'a Self> for Tensor {
    fn from(tensor: &'a Self) -> Self {
        tensor.clone()
    }
}

// 为f32引用类型实现 Into<Tensor> trait
impl<'a> From<&'a f32> for Tensor {
    fn from(scalar: &'a f32) -> Self {
        Self::new(&[*scalar], &[1, 1])
    }
}

impl Tensor {
    /// 对张量中的所有元素求和并返回一个形状为[1,1]的张量。
    pub fn sum(&self) -> Self {
        let mut value = 0.;
        Zip::from(&self.data).for_each(|a| value += a);
        Self::new(&[value], &[1, 1])
    }

    /// 沿指定轴求和，保持维度数不变（keepdims=true）
    ///
    /// # 参数
    /// - `axis`: 要求和的轴索引
    ///
    /// # 示例
    /// ```ignore
    /// let t = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    /// let result = t.sum_axis_keepdims(0);  // [2, 3] → [1, 3]
    /// assert_eq!(result.shape(), &[1, 3]);
    /// assert_eq!(result.data_as_slice(), &[4., 6., 8.]);
    /// ```
    pub fn sum_axis_keepdims(&self, axis: usize) -> Self {
        use ndarray::Axis;

        assert!(
            axis < self.dimension(),
            "sum_axis_keepdims: axis {} 超出维度范围 {}",
            axis,
            self.dimension()
        );

        // 沿指定轴求和
        let summed = self.data.sum_axis(Axis(axis));

        // 构建新形状（在求和的轴位置插入 1）
        let mut new_shape: Vec<usize> = summed.shape().to_vec();
        new_shape.insert(axis, 1);

        // 创建新张量
        Self::new(summed.as_slice().unwrap(), &new_shape)
    }

    /// 对两个操作数进行逐元素相乘后求和，返回形状为[1,1]的张量。
    ///
    /// 计算步骤：
    /// 1. 如果输入是纯数，将其广播到self的每个元素
    /// 2. 如果输入是张量，检查形状是否一致
    /// 3. 对两个张量进行逐元素相乘
    /// 4. 对乘积结果中的所有元素求和
    /// 5. 返回形状为[1,1]的结果张量
    ///
    /// # Panics
    ///
    /// 当输入是张量且形状与self不一致时会触发panic
    pub fn dot_sum<T>(&self, other: T) -> Self
    where
        T: Into<Self> + TensorType,
    {
        let product_tensor = if T::is_tensor() {
            // 如果输入是张量类型，直接检查形状
            let other = other.into();
            assert!(
                self.is_same_shape(&other),
                "{}",
                TensorError::OperatorError {
                    operator: Operator::DotSum,
                    tensor1_shape: self.shape().to_vec(),
                    tensor2_shape: other.shape().to_vec(),
                }
            );
            self * other
        } else {
            // 如果输入是纯数（f32），直接将其广播到self的每个元素
            let scalar = other.into(); // 转换为形状为[1,1]的张量
            let scalar_value = scalar.get_data_number().unwrap(); // 获取标量值
            self * scalar_value // 直接用标量值进行乘法运算
        };

        product_tensor.sum()
    }

    /// 计算张量所有元素的均值，返回形状为 [1, 1] 的张量。
    ///
    /// # 示例
    /// ```
    /// use only_torch::Tensor;
    /// let t = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    /// let result = t.mean();
    /// assert_eq!(result.shape(), &[1, 1]);
    /// assert_eq!(result[[0, 0]], 2.5);
    /// ```
    pub fn mean(&self) -> Self {
        let mean_value = self.data.mean().unwrap();
        Self::new(&[mean_value], &[1, 1])
    }

    /// 沿指定轴求均值，保持维度数不变（keepdims=true）
    ///
    /// # 参数
    /// - `axis`: 要求均值的轴索引
    ///
    /// # 示例
    /// ```
    /// use only_torch::Tensor;
    /// let t = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    /// let result = t.mean_axis_keepdims(0);  // [2, 3] → [1, 3]
    /// assert_eq!(result.shape(), &[1, 3]);
    /// assert_eq!(result.data_as_slice(), &[2.5, 3.5, 4.5]);
    /// ```
    pub fn mean_axis_keepdims(&self, axis: usize) -> Self {
        use ndarray::Axis;

        assert!(
            axis < self.dimension(),
            "mean_axis_keepdims: axis {} 超出维度范围 {}",
            axis,
            self.dimension()
        );

        // 沿指定轴求均值
        let meaned = self.data.mean_axis(Axis(axis)).unwrap();

        // 构建新形状（在求均值的轴位置插入 1）
        let mut new_shape: Vec<usize> = meaned.shape().to_vec();
        new_shape.insert(axis, 1);

        // 创建新张量
        Self::new(meaned.as_slice().unwrap(), &new_shape)
    }

    /// 计算张量的标准差
    pub fn std_dev(&self) -> f32 {
        self.data.std_axis(ndarray::Axis(0), 0.).mean().unwrap()
    }

    /// 计算张量每个元素的平方根
    pub fn sqrt(&self) -> Self {
        let sqrt_data = self.data.mapv(f32::sqrt);
        Self { data: sqrt_data }
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
    ///
    /// * `dim` - 可选的维度参数，指定沿哪个维度打乱；若为 None 则打乱所有元素
    pub fn shuffle(&self, dim: Option<usize>) -> Self {
        let mut shuffled_data = self.data.clone();
        let mut rng = thread_rng();

        if let Some(dim) = dim {
            let axis = Axis(dim);
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
    ///
    /// * `dim` - 可选的维度参数，指定沿哪个维度打乱；若为 None 则打乱所有元素
    pub fn shuffle_mut(&mut self, dim: Option<usize>) {
        let mut rng = thread_rng();

        if let Some(dim) = dim {
            let axis = Axis(dim);
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

    /// 使用指定种子打乱张量中的元素顺序（影响原张量，确保可重复性）
    ///
    /// * `dim` - 可选的维度参数，指定沿哪个维度打乱；若为 None 则打乱所有元素
    /// * `seed` - 随机数生成器的种子
    pub fn shuffle_mut_seeded(&mut self, dim: Option<usize>, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);

        if let Some(dim) = dim {
            let axis = Axis(dim);
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

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓reshape↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    pub fn reshape(&self, shape: &[usize]) -> Self {
        let total_elements: usize = self.data.len();
        let new_total_elements: usize = shape.iter().product();
        assert!(
            total_elements == new_total_elements,
            "{}",
            TensorError::IncompatibleShape
        );
        Self {
            data: self.data.clone().into_shape(shape).unwrap(),
        }
    }

    pub fn reshape_mut(&mut self, shape: &[usize]) {
        let total_elements: usize = self.data.len();
        let new_total_elements: usize = shape.iter().product();
        assert!(
            total_elements == new_total_elements,
            "{}",
            TensorError::IncompatibleShape
        );
        self.data = self.data.clone().into_shape(shape).unwrap();
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑reshape↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓stack(concat)↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 沿指定轴堆叠/拼接多个张量
    ///
    /// 统一实现 `PyTorch` 的 `torch.stack` 和 `torch.cat` 功能。
    ///
    /// # 参数
    /// - `tensors`: 要堆叠/拼接的张量切片
    /// - `axis`: 操作的轴
    /// - `new_dim`: 是否插入新维度
    ///   - `true`: 类似 `torch.stack`，在 `axis` 位置插入新维度，所有张量形状必须相同
    ///   - `false`: 类似 `torch.cat`，沿 `axis` 轴拼接，该轴可以不同但其他维度必须相同
    ///
    /// # 示例
    /// ```ignore
    /// // torch.stack 风格：插入新维度
    /// let a = Tensor::new(&[1.0, 2.0], &[2]);      // [2]
    /// let b = Tensor::new(&[3.0, 4.0], &[2]);      // [2]
    /// let stacked = Tensor::stack(&[&a, &b], 0, true);  // [2, 2]
    ///
    /// // torch.cat 风格：沿现有维度拼接
    /// let x = Tensor::new(&[1.0, 2.0], &[1, 2]);   // [1, 2]
    /// let y = Tensor::new(&[3.0, 4.0, 5.0], &[1, 3]); // [1, 3]
    /// let concat = Tensor::stack(&[&x, &y], 1, false); // [1, 5]
    /// ```
    pub fn stack(tensors: &[&Self], axis: usize, new_dim: bool) -> Self {
        assert!(!tensors.is_empty(), "{}", TensorError::EmptyList);

        let all_scalars = tensors.iter().all(|t| t.is_scalar());
        let first = tensors[0];
        let first_shape = first.shape();
        let ndim = first_shape.len();

        // 标量特殊处理
        if all_scalars {
            let data: Vec<f32> = tensors
                .iter()
                .flat_map(|t| t.data.as_slice().unwrap())
                .copied()
                .collect();
            return if new_dim {
                Self::new(&data, &[tensors.len(), 1])
            } else {
                Self::new(&data, &[tensors.len()])
            };
        }

        if new_dim {
            // torch.stack 模式：在 axis 位置插入新维度
            assert!(
                axis <= ndim,
                "stack: axis {axis} 超出张量维度 {ndim}（new_dim=true 时 axis 可以等于 ndim）"
            );

            // 所有张量形状必须完全相同
            for (i, t) in tensors.iter().enumerate().skip(1) {
                assert!(
                    t.shape() == first_shape,
                    "stack (new_dim=true): 张量 {} 的形状 {:?} 与第一个张量的形状 {:?} 不一致",
                    i,
                    t.shape(),
                    first_shape
                );
            }

            // 使用 ndarray::stack
            let views: Vec<_> = tensors.iter().map(|t| t.data.view()).collect();
            let stacked = ndarray::stack(Axis(axis), &views).expect("stack: ndarray stack 失败");
            Self { data: stacked }.into_contiguous()
        } else {
            // torch.cat 模式：沿现有 axis 拼接
            assert!(axis < ndim, "stack: axis {axis} 超出张量维度 {ndim}");

            // 检查除 axis 外的维度是否一致
            for (i, t) in tensors.iter().enumerate().skip(1) {
                let t_shape = t.shape();
                assert!(
                    t_shape.len() == ndim,
                    "stack (new_dim=false): 张量 {} 的维度 {} 与第一个张量的维度 {} 不一致",
                    i,
                    t_shape.len(),
                    ndim
                );

                for d in 0..ndim {
                    if d != axis {
                        assert!(
                            t_shape[d] == first_shape[d],
                            "stack (new_dim=false): 张量 {} 在维度 {} 的大小 {} 与第一个张量的 {} 不一致",
                            i,
                            d,
                            t_shape[d],
                            first_shape[d]
                        );
                    }
                }
            }

            // 使用 ndarray::concatenate
            let views: Vec<_> = tensors.iter().map(|t| t.data.view()).collect();
            let concatenated =
                ndarray::concatenate(Axis(axis), &views).expect("stack: ndarray concatenate 失败");
            Self { data: concatenated }.into_contiguous()
        }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑stack(concat)↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓split↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 沿指定轴分割张量
    ///
    /// 这是 `stack(..., new_dim=false)`（即 concat 模式）的逆操作。
    /// 注意：此方法不会减少维度，如需减少维度请使用 `unbind`（尚未实现）。
    ///
    /// # 参数
    /// - `axis`: 分割的轴
    /// - `sizes`: 各部分在 axis 维度的大小，之和必须等于该轴的长度
    ///
    /// # 返回
    /// 分割后的张量列表，每个张量的维度数与原张量相同
    ///
    /// # 示例
    /// ```ignore
    /// let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5]);
    /// let parts = t.split(1, &[2, 3]);  // [[1, 2], [3, 4, 5]]，形状分别为 [1, 2] 和 [1, 3]
    /// ```
    pub fn split(&self, axis: usize, sizes: &[usize]) -> Vec<Self> {
        let ndim = self.dimension();
        assert!(axis < ndim, "split: axis {axis} 超出张量维度 {ndim}");

        let total: usize = sizes.iter().sum();
        assert!(
            total == self.shape()[axis],
            "split: sizes 之和 {} 不等于轴 {} 的大小 {}",
            total,
            axis,
            self.shape()[axis]
        );

        let mut result = Vec::with_capacity(sizes.len());
        let mut start = 0;

        for &size in sizes {
            // 使用 slice_axis 获取切片
            let slice = self
                .data
                .slice_axis(Axis(axis), ndarray::Slice::from(start..start + size));
            result.push(Self {
                data: slice.to_owned(),
            });
            start += size;
        }

        result
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑split↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓(un)squeeze↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    pub fn squeeze(&self) -> Self {
        let mut new_shape = Vec::new();
        for dim in self.data.shape() {
            if *dim > 1 {
                new_shape.push(*dim);
            }
        }
        let squeezed_data = self.data.clone().into_shape(new_shape).unwrap();
        Self {
            data: squeezed_data,
        }
    }

    pub fn squeeze_mut(&mut self) {
        let mut new_shape = Vec::new();
        for dim in self.data.shape() {
            if *dim > 1 {
                new_shape.push(*dim);
            }
        }
        self.data = self
            .data
            .view_mut()
            .into_shape(new_shape)
            .unwrap()
            .to_owned();
    }

    /// 在指定维度上增加一个维度。
    ///
    /// * `dim` - 要增加维度的索引。若`dim`为正数或零，则从头开始计数；
    /// 若`dim`为负数，则从末尾开始计数。例如，-1表示在最后一个维度后增加。
    ///
    /// # 示例
    ///
    /// ```
    /// use only_torch::Tensor;
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    /// let unsqueezed = tensor.unsqueeze(0); // 在最前面增加一个维度
    /// assert_eq!(unsqueezed.shape(), &[1, 3]);
    ///
    /// let unsqueezed_last = tensor.unsqueeze(-1); // 在最后面增加一个维度
    /// assert_eq!(unsqueezed_last.shape(), &[3, 1]);
    /// ```
    pub fn unsqueeze(&self, dim: i8) -> Self {
        let dim = if dim < 0 {
            self.dimension() as i8 + dim + 1
        } else {
            dim
        };
        assert!(
            dim >= 0 && dim as usize <= self.dimension(),
            "维度超出范围。"
        );

        let mut new_shape = self.data.shape().to_vec();
        new_shape.insert(dim as usize, 1);
        self.reshape(&new_shape)
    }

    /// 就地在指定维度上增加一个维度。
    ///
    /// * `dim` - 要增加维度的索引。若`dim`为正数或零，则从头开始计数；
    /// 若`dim`为负数，则从末尾开始计数。例如，-1表示在最后一个维度增加。
    /// 若`dim`超出了当前维度的范围，将会触发panic。
    ///
    /// # 示例
    ///
    /// ```
    /// use only_torch::Tensor;
    ///
    /// let mut tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    /// tensor.unsqueeze_mut(0); // 在最前面增加一个维度
    /// assert_eq!(tensor.shape(), &[1, 3]);
    ///
    /// let mut tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    /// tensor.unsqueeze_mut(-1); // 在最后面增加一个维度
    /// assert_eq!(tensor.shape(), &[3, 1]);
    /// ```
    pub fn unsqueeze_mut(&mut self, dim: i8) {
        let dim = if dim < 0 {
            self.dimension() as i8 + dim + 1
        } else {
            dim
        };
        assert!(
            dim >= 0 && dim as usize <= self.dimension(),
            "维度超出范围。"
        );

        let mut new_shape = self.data.shape().to_vec();
        new_shape.insert(dim as usize, 1);
        self.reshape_mut(&new_shape);
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑(un)squeeze↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓permute↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 交换张量的两个（以上）维度，并将其返回（不影响原张量）
    pub fn permute(&self, axes: &[usize]) -> Self {
        assert!(axes.len() >= 2, "{}", TensorError::PermuteNeedAtLeast2Dims);
        // 检查axes中的所有元素必须是唯一且在[0, <张量维数>)范围内
        let unique_axes = axes.iter().copied().collect::<HashSet<_>>();
        assert!(
            !(unique_axes.len() != axes.len()
                || !unique_axes.iter().all(|&a| a < self.dimension())),
            "{}",
            TensorError::PermuteNeedUniqueAndInRange
        );

        let permuted_data = self.data.clone().permuted_axes(axes);
        Self {
            data: permuted_data,
        }
    }

    /// 交换张量的两个（以上）维度（影响原张量）
    pub fn permute_mut(&mut self, axes: &[usize]) {
        assert!(axes.len() >= 2, "{}", TensorError::PermuteNeedAtLeast2Dims);
        // 检查axes中的所有元素必须是唯一且在[0, <张量维数>)范围内
        let unique_axes = axes.iter().copied().collect::<HashSet<_>>();
        assert!(
            !(unique_axes.len() != axes.len()
                || !unique_axes.iter().all(|&a| a < self.dimension())),
            "{}",
            TensorError::PermuteNeedUniqueAndInRange
        );

        self.data = self.data.to_owned().permuted_axes(axes);
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑permute↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓transpose↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 张量的转置
    pub fn transpose(&self) -> Self {
        if self.dimension() <= 1 {
            self.clone()
        } else {
            let mut axes: Vec<usize> = (0..self.dimension()).collect();
            axes.swap(0, 1);
            self.permute(&axes)
        }
    }

    /// 张量的转置（影响原张量）
    pub fn transpose_mut(&mut self) {
        if self.dimension() > 1 {
            let mut axes: Vec<usize> = (0..self.dimension()).collect();
            axes.swap(0, 1);
            self.permute_mut(&axes);
        }
    }

    /// 交换指定的两个维度
    pub fn transpose_dims(&self, dim1: usize, dim2: usize) -> Self {
        assert!(
            dim1 < self.dimension() && dim2 < self.dimension(),
            "维度超出范围"
        );
        let mut axes: Vec<usize> = (0..self.dimension()).collect();
        axes.swap(dim1, dim2);
        self.permute(&axes)
    }

    /// 交换指定的两个维度（影响原张量）
    pub fn transpose_dims_mut(&mut self, dim1: usize, dim2: usize) {
        assert!(
            dim1 < self.dimension() && dim2 < self.dimension(),
            "维度超出范围"
        );
        let mut axes: Vec<usize> = (0..self.dimension()).collect();
        axes.swap(dim1, dim2);
        self.permute_mut(&axes);
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑transpose↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓flatten↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 将张量展平为1维张量，并返回新的张量（不影响原张量）
    pub fn flatten(&self) -> Self {
        let total_elements = self.data.len();
        Self {
            data: self.data.clone().into_shape(vec![total_elements]).unwrap(),
        }
    }

    /// 将张量展平为1维张量（影响原张量）
    pub fn flatten_mut(&mut self) {
        let total_elements = self.data.len();
        self.data = self.data.clone().into_shape(vec![total_elements]).unwrap();
    }

    /// 返回张量的1维展开视图，不复制数据
    /// NOTE：这个主要参考了numpy的ravel和pytorch的flatten
    pub fn flatten_view(&self) -> ndarray::ArrayView1<'_, f32> {
        self.data
            .view()
            .into_shape(ndarray::Dim(self.data.len()))
            .unwrap()
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑flatten↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓diag↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 返回一个新的张量。输入张量必须是1维或2维，否则会 panic。根据输入类型：
    /// - 若输入为标量，则返回同形状的标量
    /// - 若输入为向量，则返回以该向量为对角线的方阵
    /// - 若输入为方阵，则返回其对角线元素组成的1维向量
    /// - 若输入为非方阵，则panic
    /// 注意：对于仅含1个元素的1维或2维张量，为方便理解，可被视为标量而不是向量或方阵；
    /// 另外，不同于`numpy`的`diag`, 这里不支持诸如`[2,3]`这样的非标量、向量及方阵的情况
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// // 标量情况 (1维)
    /// let scalar = Tensor::new(&[1.0], &[1]);
    /// let diag = scalar.diag();
    /// assert_eq!(diag.shape(), &[1]);
    ///
    /// // 标量情况 (2维)
    /// let scalar = Tensor::new(&[1.0], &[1, 1]);
    /// let diag = scalar.diag();
    /// assert_eq!(diag.shape(), &[1, 1]);
    ///
    /// // 向量情况
    /// let vector = Tensor::new(&[1.0, 2.0], &[2]);
    /// let diag = vector.diag();
    /// assert_eq!(diag.shape(), &[2, 2]);
    ///
    /// // 方阵情况
    /// let matrix = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// let diag = matrix.diag();
    /// assert_eq!(diag.shape(), &[2]);
    /// ```
    pub fn diag(&self) -> Self {
        // 检查维度是否为1或2
        assert!(
            !(self.dimension() == 0 || self.dimension() > 2),
            "张量维度必须为1或2"
        );

        // 处理标量情况（size==1 时保持形状不变）
        if self.size() == 1 {
            return self.clone();
        }

        // 处理向量情况
        // 注意：向量 [n] -> 对角矩阵 [n, n]
        if self.is_vector() {
            let n = self.size();
            let mut diag_data = vec![0.0; n * n];
            let data_slice = self.data.as_slice().unwrap();
            for i in 0..n {
                diag_data[i * n + i] = data_slice[i];
            }
            return Self {
                data: Array::from_shape_vec(IxDyn(&[n, n]), diag_data).unwrap(),
            };
        }

        // 处理方阵情况
        // 注意：方阵 [n, n] -> 对角向量 [n]
        let shape = self.data.shape();
        assert!(
            !(shape.len() != 2 || shape[0] != shape[1]),
            "张量必须是标量、向量或方阵"
        );
        let diag_data = self.data.diag().to_owned();
        let diag_vector = Array::from_shape_vec(IxDyn(&[shape[0]]), diag_data.to_vec()).unwrap();
        Self { data: diag_vector }
    }

    /// 就地修改当前张量。输入张量必须是1维或2维，否则会 panic。根据输入类型：
    /// - 若输入为标量，则保持不变
    /// - 若输入为向量，则转换为以该向量为对角线的方阵
    /// - 若输入为方阵，则转换为其对角线元素组成的1维向量
    /// - 若输入为非方阵，则panic
    /// 注意：对于仅含1个元素的1维或2维张量，为方便理解，可被视为标量而不是向量或方阵；
    /// 另外，不同于`numpy`的`diag`, 这里不支持诸如`[2,3]`这样的非标量、向量及方阵的情况
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// // 标量情况 (1维)
    /// let mut scalar = Tensor::new(&[1.0], &[1]);
    /// scalar.diag_mut();
    /// assert_eq!(scalar.shape(), &[1]);
    ///
    /// // 标量情况 (2维)
    /// let mut scalar = Tensor::new(&[1.0], &[1, 1]);
    /// scalar.diag_mut();
    /// assert_eq!(scalar.shape(), &[1, 1]);
    ///
    /// // 向量情况
    /// let mut vector = Tensor::new(&[1.0, 2.0], &[2]);
    /// vector.diag_mut();
    /// assert_eq!(vector.shape(), &[2, 2]);
    ///
    /// // 方阵情况
    /// let mut matrix = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// matrix.diag_mut();
    /// assert_eq!(matrix.shape(), &[2]);
    /// ```
    pub fn diag_mut(&mut self) {
        // 检查维度是否为1或2
        assert!(
            !(self.dimension() == 0 || self.dimension() > 2),
            "张量维度必须为1或2"
        );

        // 处理标量情况（size==1 时保持形状不变）
        if self.size() == 1 {
            return;
        }

        // 处理向量情况
        // 注意：向量 [n] -> 对角矩阵 [n, n]
        if self.is_vector() {
            let n = self.size();
            let mut diag_data = vec![0.0; n * n];
            let data_slice = self.data.as_slice().unwrap();
            for i in 0..n {
                diag_data[i * n + i] = data_slice[i];
            }
            self.data = Array::from_shape_vec(IxDyn(&[n, n]), diag_data).unwrap();
            return;
        }

        // 处理方阵情况
        // 注意：方阵 [n, n] -> 对角向量 [n]
        let shape = self.data.shape();
        assert!(
            !(shape.len() != 2 || shape[0] != shape[1]),
            "张量必须是标量、向量或方阵"
        );
        let diag_data = self.data.diag().to_owned();
        let diag_vector = Array::from_shape_vec(IxDyn(&[shape[0]]), diag_data.to_vec()).unwrap();
        self.data = diag_vector;
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑diag↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓jacobi_diag↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 将张量转换为用于 Jacobian 计算的对角矩阵
    ///
    /// 专为神经网络反向传播设计：将逐元素操作的导数转换为对角 Jacobian 矩阵。
    /// 与 `diag()` 不同，本方法**始终返回 2D 矩阵**，确保与 `mat_mul` 兼容。
    ///
    /// 转换规则：
    /// - 任意形状 → 展平为 `[n]` → 对角矩阵 `[n, n]`
    /// - 特别地，`size=1` 时返回 `[1, 1]` 而非 `[1]`
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// // 标量导数 → [1, 1] 矩阵
    /// let scalar = Tensor::new(&[0.25], &[1]);
    /// let jacobi = scalar.jacobi_diag();
    /// assert_eq!(jacobi.shape(), &[1, 1]);
    ///
    /// // 向量导数 → 对角矩阵
    /// let vector = Tensor::new(&[0.1, 0.2, 0.3], &[3]);
    /// let jacobi = vector.jacobi_diag();
    /// assert_eq!(jacobi.shape(), &[3, 3]);
    /// ```
    pub fn jacobi_diag(&self) -> Self {
        let n = self.size();
        if n == 1 {
            // 标量情况：返回 [1, 1] 矩阵以兼容 mat_mul
            return Self::new(&[self.data.iter().next().copied().unwrap()], &[1, 1]);
        }
        // 一般情况：展平后构建对角矩阵
        self.flatten().diag()
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑jacobi_diag↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓tanh↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素应用双曲正切函数(tanh)
    ///
    /// tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[0.0, 1.0, -1.0], &[3]);
    /// let y = x.tanh();
    /// // y ≈ [0.0, 0.7616, -0.7616]
    /// ```
    pub fn tanh(&self) -> Self {
        let data = self.data.mapv(f32::tanh);
        Self { data }
    }

    /// 就地对张量的每个元素应用双曲正切函数(tanh)
    pub fn tanh_mut(&mut self) {
        self.data.mapv_inplace(f32::tanh);
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑tanh↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓sigmoid↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素应用 Sigmoid 函数
    ///
    /// sigmoid(x) = 1 / (1 + e^(-x))
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[0.0, 1.0, -1.0], &[3]);
    /// let y = x.sigmoid();
    /// // y ≈ [0.5, 0.7311, 0.2689]
    /// ```
    pub fn sigmoid(&self) -> Self {
        let data = self.data.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        Self { data }
    }

    /// 就地对张量的每个元素应用 Sigmoid 函数
    pub fn sigmoid_mut(&mut self) {
        self.data.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑sigmoid↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓exp↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素计算指数函数 e^x
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[0.0, 1.0, 2.0], &[3]);
    /// let y = x.exp();
    /// // y ≈ [1.0, 2.7183, 7.3891]
    /// ```
    pub fn exp(&self) -> Self {
        let data = self.data.mapv(f32::exp);
        Self { data }
    }

    /// 就地对张量的每个元素计算指数函数
    pub fn exp_mut(&mut self) {
        self.data.mapv_inplace(f32::exp);
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑exp↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ln↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素计算自然对数 ln(x)
    ///
    /// # 注意
    /// 对于 x <= 0 的元素，结果为 NaN 或 -inf
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[1.0, 2.7183, 7.3891], &[3]);
    /// let y = x.ln();
    /// // y ≈ [0.0, 1.0, 2.0]
    /// ```
    pub fn ln(&self) -> Self {
        let data = self.data.mapv(f32::ln);
        Self { data }
    }

    /// 就地对张量的每个元素计算自然对数
    pub fn ln_mut(&mut self) {
        self.data.mapv_inplace(f32::ln);
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ln↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓softmax↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 沿指定轴计算 softmax（数值稳定版本）
    ///
    /// softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
    ///
    /// 通过先减去最大值再计算 exp，避免数值溢出。
    ///
    /// # 参数
    /// - `axis`: 沿哪个轴计算 softmax
    ///
    /// # 返回
    /// 新张量，形状与输入相同，沿指定轴的元素和为 1
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
    /// let probs = x.softmax(1);  // 沿最后一维计算
    ///
    /// // 每行和为 1
    /// assert!((probs[[0, 0]] + probs[[0, 1]] + probs[[0, 2]] - 1.0).abs() < 1e-6);
    /// assert!((probs[[1, 0]] + probs[[1, 1]] + probs[[1, 2]] - 1.0).abs() < 1e-6);
    ///
    /// // softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652]
    /// assert!((probs[[0, 2]] - 0.6652).abs() < 0.001);
    /// ```
    pub fn softmax(&self, axis: usize) -> Self {
        assert!(
            axis < self.dimension(),
            "softmax: axis {} 超出维度范围 {}",
            axis,
            self.dimension()
        );

        // 数值稳定：先减去 max
        let max_vals = self.amax(axis); // 沿 axis 取最大值
        let max_broadcast = max_vals.unsqueeze(axis as i8); // 恢复维度以便广播

        // x - max(x)
        let shifted = self - &max_broadcast;

        // exp(x - max)
        let exp_vals = shifted.exp();

        // sum(exp)
        let sum_exp = exp_vals.sum_axis_keepdims(axis);

        // exp / sum
        &exp_vals / &sum_exp
    }

    /// 沿最后一维计算 softmax（数值稳定版本）
    ///
    /// 等价于 `self.softmax(self.dimension() - 1)`，这是最常用的情况。
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let logits = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    /// let probs = logits.softmax_last_dim();
    /// // probs ≈ [[0.0900, 0.2447, 0.6652]]
    /// ```
    pub fn softmax_last_dim(&self) -> Self {
        assert!(
            self.dimension() > 0,
            "softmax_last_dim: 张量维度必须大于 0"
        );
        self.softmax(self.dimension() - 1)
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑softmax↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓log_softmax↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 沿指定轴计算 log_softmax（数值稳定版本）
    ///
    /// `log_softmax(x) = log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))`
    ///
    /// 比直接计算 `softmax(x).ln()` 更数值稳定，避免 softmax 输出接近 0 时的精度问题。
    ///
    /// # 参数
    /// - `axis`: 计算 softmax 的轴
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let logits = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
    /// let log_probs = logits.log_softmax(1);
    ///
    /// // 检查形状
    /// assert_eq!(log_probs.shape(), &[2, 3]);
    ///
    /// // log_softmax 输出应该都是负数（因为 softmax 输出 < 1）
    /// assert!(log_probs[[0, 0]] < 0.0);
    /// assert!(log_probs[[0, 1]] < 0.0);
    /// assert!(log_probs[[0, 2]] < 0.0);
    ///
    /// // exp(log_softmax) 应该等于 softmax
    /// let probs = log_probs.exp();
    /// let sum = probs[[0, 0]] + probs[[0, 1]] + probs[[0, 2]];
    /// assert!((sum - 1.0).abs() < 1e-6);
    /// ```
    pub fn log_softmax(&self, axis: usize) -> Self {
        assert!(
            axis < self.dimension(),
            "log_softmax: axis {} 超出维度范围 {}",
            axis,
            self.dimension()
        );

        // 数值稳定：先减去 max
        let max_vals = self.amax(axis);
        let max_broadcast = max_vals.unsqueeze(axis as i8);

        // shifted = x - max(x)
        let shifted = self - &max_broadcast;

        // log_sum_exp = log(sum(exp(shifted)))
        let exp_vals = shifted.exp();
        let sum_exp = exp_vals.sum_axis_keepdims(axis);
        let log_sum_exp = sum_exp.ln();

        // log_softmax = shifted - log_sum_exp
        &shifted - &log_sum_exp
    }

    /// 沿最后一维计算 log_softmax（数值稳定版本）
    ///
    /// 等价于 `self.log_softmax(self.dimension() - 1)`，这是最常用的情况。
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let logits = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    /// let log_probs = logits.log_softmax_last_dim();
    ///
    /// // log_softmax([1,2,3]) ≈ [-2.407, -1.407, -0.407]
    /// assert!((log_probs[[0, 0]] - (-2.407)).abs() < 0.01);
    /// ```
    pub fn log_softmax_last_dim(&self) -> Self {
        assert!(
            self.dimension() > 0,
            "log_softmax_last_dim: 张量维度必须大于 0"
        );
        self.log_softmax(self.dimension() - 1)
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑log_softmax↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓max/min↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 返回张量中的最大值（标量）
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[1.0, 3.0, 2.0], &[3]);
    /// assert_eq!(x.max_value(), 3.0);
    /// ```
    pub fn max_value(&self) -> f32 {
        self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    /// 返回张量中的最小值（标量）
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[1.0, 3.0, 2.0], &[3]);
    /// assert_eq!(x.min_value(), 1.0);
    /// ```
    pub fn min_value(&self) -> f32 {
        self.data.iter().copied().fold(f32::INFINITY, f32::min)
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑max/min↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓minimum/maximum↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 逐元素取两个张量的最小值
    ///
    /// 类似 `PyTorch` 的 `torch.minimum(a, b)` 或 `NumPy` 的 `np.minimum(a, b)`。
    /// 支持 NumPy 风格的广播（broadcasting）。
    ///
    /// # 参数
    /// - `other`: 另一个张量
    ///
    /// # 返回
    /// 新张量，形状为广播后的形状，每个元素为对应位置两个输入的较小值
    ///
    /// # Panics
    /// 如果两个张量的形状不兼容（无法广播）
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// // 相同形状
    /// let a = Tensor::new(&[1.0, 4.0, 3.0], &[3]);
    /// let b = Tensor::new(&[2.0, 2.0, 5.0], &[3]);
    /// let result = a.minimum(&b);
    /// assert_eq!(result[[0]], 1.0);  // min(1, 2) = 1
    /// assert_eq!(result[[1]], 2.0);  // min(4, 2) = 2
    /// assert_eq!(result[[2]], 3.0);  // min(3, 5) = 3
    ///
    /// // 广播：标量与向量
    /// let a = Tensor::new(&[1.0, 4.0, 3.0], &[3]);
    /// let b = Tensor::new(&[2.0], &[1]);
    /// let result = a.minimum(&b);
    /// assert_eq!(result[[0]], 1.0);  // min(1, 2) = 1
    /// assert_eq!(result[[1]], 2.0);  // min(4, 2) = 2
    /// assert_eq!(result[[2]], 2.0);  // min(3, 2) = 2
    /// ```
    pub fn minimum(&self, other: &Tensor) -> Tensor {
        use crate::tensor::property::broadcast_shape;

        // 检查广播兼容性
        assert!(
            self.can_broadcast_with(other),
            "{}",
            TensorError::IncompatibleShape
        );

        // 计算广播后的形状
        let result_shape =
            broadcast_shape(self.shape(), other.shape()).expect("广播形状计算失败");

        // 使用 Zip 的 and_broadcast 实现逐元素最小值
        let result_data = Zip::from(self.data.broadcast(IxDyn(&result_shape)).unwrap())
            .and(other.data.broadcast(IxDyn(&result_shape)).unwrap())
            .map_collect(|&a, &b| a.min(b));

        Tensor { data: result_data }
    }

    /// 逐元素取两个张量的最大值
    ///
    /// 类似 `PyTorch` 的 `torch.maximum(a, b)` 或 `NumPy` 的 `np.maximum(a, b)`。
    /// 支持 NumPy 风格的广播（broadcasting）。
    ///
    /// # 参数
    /// - `other`: 另一个张量
    ///
    /// # 返回
    /// 新张量，形状为广播后的形状，每个元素为对应位置两个输入的较大值
    ///
    /// # Panics
    /// 如果两个张量的形状不兼容（无法广播）
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// // 相同形状
    /// let a = Tensor::new(&[1.0, 4.0, 3.0], &[3]);
    /// let b = Tensor::new(&[2.0, 2.0, 5.0], &[3]);
    /// let result = a.maximum(&b);
    /// assert_eq!(result[[0]], 2.0);  // max(1, 2) = 2
    /// assert_eq!(result[[1]], 4.0);  // max(4, 2) = 4
    /// assert_eq!(result[[2]], 5.0);  // max(3, 5) = 5
    ///
    /// // 广播：标量与向量
    /// let a = Tensor::new(&[1.0, 4.0, 3.0], &[3]);
    /// let b = Tensor::new(&[2.0], &[1]);
    /// let result = a.maximum(&b);
    /// assert_eq!(result[[0]], 2.0);  // max(1, 2) = 2
    /// assert_eq!(result[[1]], 4.0);  // max(4, 2) = 4
    /// assert_eq!(result[[2]], 3.0);  // max(3, 2) = 3
    /// ```
    pub fn maximum(&self, other: &Tensor) -> Tensor {
        use crate::tensor::property::broadcast_shape;

        // 检查广播兼容性
        assert!(
            self.can_broadcast_with(other),
            "{}",
            TensorError::IncompatibleShape
        );

        // 计算广播后的形状
        let result_shape =
            broadcast_shape(self.shape(), other.shape()).expect("广播形状计算失败");

        // 使用 Zip 实现逐元素最大值
        let result_data = Zip::from(self.data.broadcast(IxDyn(&result_shape)).unwrap())
            .and(other.data.broadcast(IxDyn(&result_shape)).unwrap())
            .map_collect(|&a, &b| a.max(b));

        Tensor { data: result_data }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑minimum/maximum↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓argmax/argmin↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 沿指定轴返回最大值的索引
    ///
    /// 类似 `PyTorch` 的 `tensor.argmax(dim=axis)`。
    ///
    /// # 参数
    /// - `axis`: 沿哪个轴查找最大值的索引
    ///
    /// # 返回
    /// 形状为原张量去掉 `axis` 维度后的张量，元素为 `usize` 类型的索引（以 `f32` 存储）
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// // 2D 张量
    /// let x = Tensor::new(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0], &[2, 3]);
    /// // [[1, 3, 2],
    /// //  [5, 4, 6]]
    ///
    /// let argmax_axis1 = x.argmax(1);  // 沿列方向找最大
    /// assert_eq!(argmax_axis1.shape(), &[2]);
    /// assert_eq!(argmax_axis1[[0]], 1.0);  // 第 0 行最大值在索引 1
    /// assert_eq!(argmax_axis1[[1]], 2.0);  // 第 1 行最大值在索引 2
    ///
    /// let argmax_axis0 = x.argmax(0);  // 沿行方向找最大
    /// assert_eq!(argmax_axis0.shape(), &[3]);
    /// assert_eq!(argmax_axis0[[0]], 1.0);  // 第 0 列最大值在行索引 1
    /// ```
    pub fn argmax(&self, axis: usize) -> Self {
        assert!(
            axis < self.dimension(),
            "argmax: axis {} 超出维度范围 {}",
            axis,
            self.dimension()
        );

        // 使用 map_axis 沿指定轴找 argmax
        // 注意：使用 fold 而非 max_by，确保在相等时返回第一个索引（与 PyTorch 行为一致）
        let argmax_array = self.data.map_axis(Axis(axis), |lane| {
            lane.iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |(max_idx, max_val), (idx, &val)| {
                    if val > max_val {
                        (idx, val)
                    } else {
                        (max_idx, max_val)
                    }
                })
                .0 as f32
        });

        Self {
            data: argmax_array.into_dyn(),
        }
    }

    /// 沿指定轴返回最小值的索引
    ///
    /// 类似 `PyTorch` 的 `tensor.argmin(dim=axis)`。
    ///
    /// # 参数
    /// - `axis`: 沿哪个轴查找最小值的索引
    ///
    /// # 返回
    /// 形状为原张量去掉 `axis` 维度后的张量，元素为 `usize` 类型的索引（以 `f32` 存储）
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0], &[2, 3]);
    /// let argmin_axis1 = x.argmin(1);
    /// assert_eq!(argmin_axis1[[0]], 0.0);  // 第 0 行最小值在索引 0
    /// assert_eq!(argmin_axis1[[1]], 1.0);  // 第 1 行最小值在索引 1
    /// ```
    pub fn argmin(&self, axis: usize) -> Self {
        assert!(
            axis < self.dimension(),
            "argmin: axis {} 超出维度范围 {}",
            axis,
            self.dimension()
        );

        // 使用 fold 而非 min_by，确保在相等时返回第一个索引（与 PyTorch 行为一致）
        let argmin_array = self.data.map_axis(Axis(axis), |lane| {
            lane.iter()
                .enumerate()
                .fold((0, f32::INFINITY), |(min_idx, min_val), (idx, &val)| {
                    if val < min_val {
                        (idx, val)
                    } else {
                        (min_idx, min_val)
                    }
                })
                .0 as f32
        });

        Self {
            data: argmin_array.into_dyn(),
        }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑argmax/argmin↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓amax/amin(axis)↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 沿指定轴返回最小值（只返回值，不返回索引）
    ///
    /// 对应 `PyTorch` 的 `tensor.amin(dim=axis)`。
    ///
    /// # 参数
    /// - `axis`: 沿哪个轴查找最小值
    ///
    /// # 返回
    /// 形状为原张量去掉 `axis` 维度后的张量，每个元素为沿该轴的最小值
    ///
    /// # Panics
    /// 如果 `axis` 超出维度范围
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// // 2D 张量
    /// let x = Tensor::new(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0], &[2, 3]);
    /// // [[1, 3, 2],
    /// //  [5, 4, 6]]
    ///
    /// let min_axis1 = x.amin(1);  // 沿列方向找最小
    /// assert_eq!(min_axis1.shape(), &[2]);
    /// assert_eq!(min_axis1[[0]], 1.0);  // 第 0 行最小值是 1
    /// assert_eq!(min_axis1[[1]], 4.0);  // 第 1 行最小值是 4
    ///
    /// let min_axis0 = x.amin(0);  // 沿行方向找最小
    /// assert_eq!(min_axis0.shape(), &[3]);
    /// assert_eq!(min_axis0[[0]], 1.0);  // 第 0 列最小值是 1
    /// assert_eq!(min_axis0[[1]], 3.0);  // 第 1 列最小值是 3
    /// assert_eq!(min_axis0[[2]], 2.0);  // 第 2 列最小值是 2
    /// ```
    pub fn amin(&self, axis: usize) -> Self {
        assert!(
            axis < self.dimension(),
            "amin: axis {} 超出维度范围 {}",
            axis,
            self.dimension()
        );

        // 使用 map_axis 沿指定轴找最小值
        let min_array = self.data.map_axis(Axis(axis), |lane| {
            lane.iter().copied().fold(f32::INFINITY, f32::min)
        });

        Self {
            data: min_array.into_dyn(),
        }
    }

    /// 沿指定轴返回最大值（只返回值，不返回索引）
    ///
    /// 对应 `PyTorch` 的 `tensor.amax(dim=axis)`。
    ///
    /// # 参数
    /// - `axis`: 沿哪个轴查找最大值
    ///
    /// # 返回
    /// 形状为原张量去掉 `axis` 维度后的张量，每个元素为沿该轴的最大值
    ///
    /// # Panics
    /// 如果 `axis` 超出维度范围
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// // 2D 张量
    /// let x = Tensor::new(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0], &[2, 3]);
    /// // [[1, 3, 2],
    /// //  [5, 4, 6]]
    ///
    /// let max_axis1 = x.amax(1);  // 沿列方向找最大
    /// assert_eq!(max_axis1.shape(), &[2]);
    /// assert_eq!(max_axis1[[0]], 3.0);  // 第 0 行最大值是 3
    /// assert_eq!(max_axis1[[1]], 6.0);  // 第 1 行最大值是 6
    ///
    /// let max_axis0 = x.amax(0);  // 沿行方向找最大
    /// assert_eq!(max_axis0.shape(), &[3]);
    /// assert_eq!(max_axis0[[0]], 5.0);  // 第 0 列最大值是 5
    /// assert_eq!(max_axis0[[1]], 4.0);  // 第 1 列最大值是 4
    /// assert_eq!(max_axis0[[2]], 6.0);  // 第 2 列最大值是 6
    /// ```
    pub fn amax(&self, axis: usize) -> Self {
        assert!(
            axis < self.dimension(),
            "amax: axis {} 超出维度范围 {}",
            axis,
            self.dimension()
        );

        // 使用 map_axis 沿指定轴找最大值
        let max_array = self.data.map_axis(Axis(axis), |lane| {
            lane.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        });

        Self {
            data: max_array.into_dyn(),
        }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑amax/amin(axis)↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓sign↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 返回张量每个元素的符号
    ///
    /// - 正数返回 1.0
    /// - 负数返回 -1.0
    /// - 零返回 0.0
    /// - NaN 返回 NaN
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    /// let y = x.sign();
    /// // y = [-1.0, -1.0, 0.0, 1.0, 1.0]
    /// ```
    pub fn sign(&self) -> Self {
        // 注：Rust 的 f32::signum() 对 0.0 返回 1.0，这与 PyTorch 行为不同
        // 这里显式处理零值，使其返回 0.0
        let data = self.data.mapv(|x| if x == 0.0 { 0.0 } else { x.signum() });
        Self { data }
    }

    /// 就地计算张量每个元素的符号
    pub fn sign_mut(&mut self) {
        self.data
            .mapv_inplace(|x| if x == 0.0 { 0.0 } else { x.signum() });
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑sign↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓abs↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 计算张量每个元素的绝对值
    ///
    /// - 正数返回原值
    /// - 负数返回其相反数
    /// - 零返回 0.0
    /// - NaN 返回 NaN
    /// - ±INFINITY 返回 INFINITY
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    /// let y = x.abs();
    /// // y = [2.0, 1.0, 0.0, 1.0, 2.0]
    /// ```
    pub fn abs(&self) -> Self {
        let data = self.data.mapv(f32::abs);
        Self { data }
    }

    /// 就地计算张量每个元素的绝对值
    pub fn abs_mut(&mut self) {
        self.data.mapv_inplace(f32::abs);
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑abs↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓gather↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 按索引张量从指定维度收集元素
    ///
    /// 类似 `PyTorch` 的 `torch.gather(input, dim, index)`。
    /// 对于 2D 张量 `input[M, N]` 和索引 `index[M, K]`（dim=1 时）：
    /// - `output[i, j] = input[i, index[i, j]]`
    ///
    /// # 参数
    /// - `dim`: 沿哪个维度进行 gather 操作
    /// - `index`: 索引张量（元素为 f32，会被转换为 usize）
    ///
    /// # 返回
    /// 与 `index` 形状相同的张量
    ///
    /// # Panics
    /// - 如果 `dim` 超出维度范围
    /// - 如果 `index` 和 `self` 的维度数不同
    /// - 如果 `index` 中除 `dim` 维度外的其他维度大小与 `self` 不匹配
    /// - 如果 `index` 中的索引值超出 `self` 在 `dim` 维度的范围
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// // SAC/DQN 场景：按动作索引选择 Q 值
    /// // Q 值：[[1.0, 2.0, 3.0],   (batch=2, action_dim=3)
    /// //        [4.0, 5.0, 6.0]]
    /// let q_values = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    ///
    /// // 动作索引：[[1],   (选第 0 行的索引 1)
    /// //           [2]]   (选第 1 行的索引 2)
    /// let actions = Tensor::new(&[1.0, 2.0], &[2, 1]);
    ///
    /// let selected = q_values.gather(1, &actions);
    /// // selected = [[2.0],   <- q_values[0, 1]
    /// //             [6.0]]   <- q_values[1, 2]
    /// assert_eq!(selected.shape(), &[2, 1]);
    /// assert_eq!(selected[[0, 0]], 2.0);
    /// assert_eq!(selected[[1, 0]], 6.0);
    /// ```
    pub fn gather(&self, dim: usize, index: &Tensor) -> Tensor {
        let self_shape = self.shape();
        let index_shape = index.shape();
        let ndim = self_shape.len();

        // 1. 验证 dim
        assert!(
            dim < ndim,
            "gather: dim {} 超出张量维度 {}",
            dim,
            ndim
        );

        // 2. 验证 index 维度数与 self 相同
        assert!(
            index_shape.len() == ndim,
            "gather: index 维度数 {} 必须与输入张量维度数 {} 相同",
            index_shape.len(),
            ndim
        );

        // 3. 验证除 dim 外的其他维度大小一致
        for d in 0..ndim {
            if d != dim {
                assert!(
                    index_shape[d] == self_shape[d],
                    "gather: 维度 {} 上 index 大小 {} 与输入张量大小 {} 不匹配",
                    d,
                    index_shape[d],
                    self_shape[d]
                );
            }
        }

        // 4. 计算输出大小并收集元素
        let output_size: usize = index_shape.iter().product();
        let mut output_data = Vec::with_capacity(output_size);

        // 使用 ndindex 遍历 index 张量的每个位置
        for idx in ndarray::indices(index_shape) {
            // 获取 index 中的值作为 gather 索引
            let gather_idx = index.data[&idx] as usize;

            // 验证索引范围
            assert!(
                gather_idx < self_shape[dim],
                "gather: 索引 {} 超出维度 {} 的范围 [0, {})",
                gather_idx,
                dim,
                self_shape[dim]
            );

            // 构建从 self 中取值的索引
            let mut self_idx: Vec<usize> = idx.as_array_view().to_vec();
            self_idx[dim] = gather_idx;

            // 取值
            output_data.push(self.data[IxDyn(&self_idx)]);
        }

        Tensor::new(&output_data, index_shape)
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑gather↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓soft_update↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 参数软更新（Soft Update）
    ///
    /// 用于强化学习中目标网络的平滑更新：
    /// `self = τ × source + (1 - τ) × self`
    ///
    /// # 参数
    /// - `source`: 源张量（如在线网络参数）
    /// - `tau`: 更新系数，通常取较小值（如 0.005）
    ///
    /// # Panics
    /// 如果 `self` 和 `source` 形状不匹配
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let mut target = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// let online = Tensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    ///
    /// target.soft_update(&online, 0.1);
    /// // target = 0.1 * [10, 20, 30, 40] + 0.9 * [1, 2, 3, 4]
    /// //        = [1.9, 3.8, 5.7, 7.6]
    /// ```
    pub fn soft_update(&mut self, source: &Tensor, tau: f32) {
        assert_eq!(
            self.shape(),
            source.shape(),
            "soft_update: 形状不匹配，self: {:?}, source: {:?}",
            self.shape(),
            source.shape()
        );

        // self = τ * source + (1 - τ) * self
        Zip::from(&mut self.data)
            .and(&source.data)
            .for_each(|target, &src| {
                *target = tau * src + (1.0 - tau) * *target;
            });
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑soft_update↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}
