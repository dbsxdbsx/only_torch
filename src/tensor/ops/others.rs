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
use ndarray::{Array, Axis, IxDyn, Zip};
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

    /// 计算张量的均值
    pub fn mean(&self) -> f32 {
        self.data.mean().unwrap()
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
    /// 将多个张量沿着首个维度堆叠起来，返回一个新的张量。
    /// * `tensors` - 一个包含多个张量的数组的引用。
    /// * `new_dim` - 布尔值，指示是否增加一个新的维度来堆叠。
    ///
    /// 当 `new_dim` 为 `true` 时，确保所有张量具有相同的形状。除非所有张量都是标量，则它们将堆叠为形状为 `[tensors.len(), 1]` 的张量。
    /// 当 `new_dim` 为 `false`，每个张量的第1个维度可以不同，但其余维度须相同。除非所有张量都是标量，则它们将堆叠为形状为 `[tensors.len()]` 的张量。
    /// 否则报错。
    pub fn stack(tensors: &[&Self], new_dim: bool) -> Self {
        assert!(!tensors.is_empty(), "{}", TensorError::EmptyList);

        let all_scalars = tensors.iter().all(|t| t.is_scalar());
        let first_shape = tensors[0].shape();

        let compatible_shapes = |t: &Self| {
            if all_scalars {
                true
            } else {
                let t_shape = t.shape();
                let skip = usize::from(!new_dim);
                t_shape.len() == first_shape.len()
                    && t_shape
                        .iter()
                        .skip(skip)
                        .zip(first_shape.iter().skip(skip))
                        .all(|(a, b)| a == b)
            }
        };

        assert!(
            tensors.iter().all(|t| compatible_shapes(t)),
            "{}",
            TensorError::InconsitentShape
        );

        let data = tensors
            .iter()
            .flat_map(|t| t.data.as_slice().unwrap())
            .copied()
            .collect::<Vec<_>>();

        let shape = match (new_dim, all_scalars) {
            (true, true) => vec![tensors.len(), 1],
            (true, false) => {
                let mut shape = tensors[0].shape().to_vec();
                shape.insert(0, tensors.len());
                shape
            }
            (false, true) => vec![tensors.len()],
            (false, false) => {
                let mut shape = tensors[0].shape().to_vec();
                shape[0] = tensors.iter().map(|t| t.shape()[0]).sum();
                shape
            }
        };

        Self::new(&data, &shape)
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑stack(concat)↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

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

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓argmax/argmin↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 沿指定轴返回最大值的索引
    ///
    /// 类似 PyTorch 的 `tensor.argmax(dim=axis)`。
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
    /// 类似 PyTorch 的 `tensor.argmin(dim=axis)`。
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
}
