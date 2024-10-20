use super::Tensor;
use crate::errors::TensorError;
use std::collections::HashSet;

impl Tensor {
    /// 若为向量，`shape`可以是[n]、[1,n]、[n,1]；
    /// 若为矩阵，`shape`可以是[n,m]；
    /// 若为更高维度的数组，`shape`可以是[c,n,m,...]。
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

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

    /// 张量的维（dim）数、阶（rank）数
    /// 即`shape()`的元素个数--如：形状为`[]`的标量阶数为0，向量阶数为1，矩阵阶数为2，以此类推
    pub fn dimension(&self) -> usize {
        self.data.ndim()
    }

    /// 判断两个张量的形状是否严格一致。如：形状为 [1, 4]，[1, 4]和[4]是不一致的，会返回false
    pub fn is_same_shape(&self, other: &Self) -> bool {
        self.shape() == other.shape()
    }

    /// 判断张量是否为标量
    pub fn is_scalar(&self) -> bool {
        self.shape().is_empty() || self.shape().iter().all(|x| *x == 1)
    }

    /// 转化为纯数（number）。若为标量，则返回Some(number)，否则返回None
    pub fn number(&self) -> Option<f32> {
        if self.is_scalar() {
            let shape = self.shape();
            let index_array = self.generate_index_array(shape);
            Some(self.data[&index_array[..]])
        } else {
            None
        }
    }

    /// 将多个张量沿着首个维度堆叠起来，返回一个新的张量。
    /// * `tensors` - 一个包含多个张量的数组的引用。
    /// * `new_dim` - 布尔值，指示是否增加一个新的维度来堆叠。
    ///
    /// 当 `new_dim` 为 `true` 时，确保所有张量具有相同的形状。除非所有张量都是标量，则它们将堆叠为形状为 `[tensors.len(), 1]` 的张量。
    /// 当 `new_dim` 为 `false`，确保所每个张量的第一个维度可以不同，但其余维度应相同。除非所有张量都是标量，则它们将堆叠为形状为 `[tensors.len()]` 的张量。
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
                let skip = if new_dim { 0 } else { 1 };
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
    /// ```ignore
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    /// let unsqueezed = tensor.unsqueeze(0); // 在最前面增加一个维度
    /// assert_eq!(unsqueezed.shape(), &[1, 3]);
    /// let unsqueezed_last = tensor.unsqueeze(-1); // 在最后面增加一个维度
    /// assert_eq!(unsqueezed.shape(), &[3, 1]);
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
    /// ```ignore
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
}
