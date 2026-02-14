/*
 * @Author       : 老董
 * @Date         : 2026-02-13
 * @Description  : 张量形状变换：reshape、stack、concat、split、squeeze、permute、transpose、flatten、diag 等
 */

use std::collections::HashSet;

use super::super::next_source_id;

use crate::errors::TensorError;
use crate::tensor::Tensor;
use ndarray::{Array, Axis, IxDyn};

impl Tensor {
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
            source_id: next_source_id(),
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

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓stack↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 沿新维度堆叠多个张量（类似 `torch.stack`）
    ///
    /// 在 `axis` 位置插入新维度，所有张量形状必须完全相同。
    ///
    /// # 参数
    /// - `tensors`: 要堆叠的张量切片
    /// - `axis`: 插入新维度的位置（0 到 ndim，包含 ndim）
    ///
    /// # 示例
    /// ```ignore
    /// let a = Tensor::new(&[1.0, 2.0], &[2]);      // [2]
    /// let b = Tensor::new(&[3.0, 4.0], &[2]);      // [2]
    /// let stacked = Tensor::stack(&[&a, &b], 0);    // [2, 2]
    /// ```
    pub fn stack(tensors: &[&Self], axis: usize) -> Self {
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
            return Self::new(&data, &[tensors.len(), 1]);
        }

        assert!(
            axis <= ndim,
            "stack: axis {axis} 超出张量维度 {ndim}（axis 可以等于 ndim）"
        );

        // 所有张量形状必须完全相同
        for (i, t) in tensors.iter().enumerate().skip(1) {
            assert!(
                t.shape() == first_shape,
                "stack: 张量 {} 的形状 {:?} 与第一个张量的形状 {:?} 不一致",
                i,
                t.shape(),
                first_shape
            );
        }

        // 使用 ndarray::stack
        let views: Vec<_> = tensors.iter().map(|t| t.data.view()).collect();
        let stacked = ndarray::stack(Axis(axis), &views).expect("stack: ndarray stack 失败");
        Self { data: stacked, source_id: next_source_id() }.into_contiguous()
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑stack↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓concat↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 沿现有维度拼接多个张量（类似 `torch.cat` / `tf.concat`）
    ///
    /// 沿 `axis` 轴拼接，该轴大小可以不同，但其他维度必须相同。
    ///
    /// # 参数
    /// - `tensors`: 要拼接的张量切片
    /// - `axis`: 拼接的轴（必须是已有维度）
    ///
    /// # 示例
    /// ```ignore
    /// let x = Tensor::new(&[1.0, 2.0], &[1, 2]);       // [1, 2]
    /// let y = Tensor::new(&[3.0, 4.0, 5.0], &[1, 3]);  // [1, 3]
    /// let result = Tensor::concat(&[&x, &y], 1);        // [1, 5]
    /// ```
    pub fn concat(tensors: &[&Self], axis: usize) -> Self {
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
            return Self::new(&data, &[tensors.len()]);
        }

        assert!(axis < ndim, "concat: axis {axis} 超出张量维度 {ndim}");

        // 检查除 axis 外的维度是否一致
        for (i, t) in tensors.iter().enumerate().skip(1) {
            let t_shape = t.shape();
            assert!(
                t_shape.len() == ndim,
                "concat: 张量 {} 的维度 {} 与第一个张量的维度 {} 不一致",
                i,
                t_shape.len(),
                ndim
            );

            for d in 0..ndim {
                if d != axis {
                    assert!(
                        t_shape[d] == first_shape[d],
                        "concat: 张量 {} 在维度 {} 的大小 {} 与第一个张量的 {} 不一致",
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
            ndarray::concatenate(Axis(axis), &views).expect("concat: ndarray concatenate 失败");
        Self { data: concatenated, source_id: next_source_id() }.into_contiguous()
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑concat↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓split↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 沿指定轴分割张量
    ///
    /// 这是 `Tensor::concat` 的逆操作。
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
                source_id: next_source_id(),
            });
            start += size;
        }

        result
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑split↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓narrow↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 沿指定轴取连续子范围（不降维）
    ///
    /// 等价于 PyTorch 的 `tensor.narrow(dim, start, length)`。
    ///
    /// # 参数
    /// - `axis`: 操作的轴
    /// - `start`: 起始索引
    /// - `length`: 取的长度
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    /// let n = t.narrow(1, 1, 2);  // 沿 axis=1 从 index 1 开始取 2 个
    /// assert_eq!(n.shape(), &[2, 2]);
    /// // n = [[2, 3], [5, 6]]
    /// ```
    pub fn narrow(&self, axis: usize, start: usize, length: usize) -> Self {
        let ndim = self.dimension();
        assert!(axis < ndim, "narrow: axis {axis} 超出张量维度 {ndim}");
        assert!(
            start + length <= self.shape()[axis],
            "narrow: start({start}) + length({length}) 超出轴 {axis} 的大小 {}",
            self.shape()[axis]
        );

        let slice = self
            .data
            .slice_axis(Axis(axis), ndarray::Slice::from(start..start + length));
        Self {
            data: slice.to_owned(),
            source_id: next_source_id(),
        }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑narrow↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

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
            source_id: next_source_id(),
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
            source_id: next_source_id(),
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
            source_id: next_source_id(),
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
                source_id: next_source_id(),
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
        Self { data: diag_vector, source_id: next_source_id() }
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
}
