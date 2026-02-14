/*
 * @Author       : 老董
 * @Date         : 2026-02-13
 * @Description  : 张量归约运算：求和、均值、极值、标准差等
 */

use super::super::next_source_id;
use crate::errors::{Operator, TensorError};
use crate::tensor::Tensor;
use ndarray::{Axis, Zip};

/// 用于 dot_sum 同时支持 Tensor 和 f32 的 trait
pub trait TensorType {
    fn is_tensor() -> bool;
}

impl TensorType for Tensor {
    fn is_tensor() -> bool {
        true
    }
}

impl TensorType for f32 {
    fn is_tensor() -> bool {
        false
    }
}

impl TensorType for &Tensor {
    fn is_tensor() -> bool {
        true
    }
}

impl TensorType for &f32 {
    fn is_tensor() -> bool {
        false
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

    /// 计算张量所有元素的总体方差（ddof=0），返回形状为 [1, 1] 的张量。
    ///
    /// 公式：`var = mean((x - mean(x))^2)`
    ///
    /// # 示例
    /// ```
    /// use only_torch::Tensor;
    /// let t = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    /// let var = t.variance();
    /// assert!((var[[0, 0]] - 2.9166667).abs() < 1e-5);
    /// ```
    pub fn variance(&self) -> Self {
        let mean_val = self.data.mean().unwrap();
        let n = self.data.len() as f32;
        let var = self.data.iter().map(|&x| (x - mean_val).powi(2)).sum::<f32>() / n;
        Self::new(&[var], &[1, 1])
    }

    /// 沿指定轴计算总体方差（ddof=0），保持维度数不变（keepdims=true）
    ///
    /// # 参数
    /// - `axis`: 要计算方差的轴索引
    ///
    /// # 示例
    /// ```
    /// use only_torch::Tensor;
    /// let t = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    /// let var0 = t.var_axis_keepdims(0);  // [2, 3] → [1, 3]
    /// assert_eq!(var0.shape(), &[1, 3]);
    /// // 每列方差：[(1-2.5)^2+(4-2.5)^2]/2 = 2.25
    /// assert!((var0.data_as_slice()[0] - 2.25).abs() < 1e-5);
    /// ```
    pub fn var_axis_keepdims(&self, axis: usize) -> Self {
        assert!(
            axis < self.dimension(),
            "var_axis_keepdims: axis {} 超出维度范围 {}",
            axis,
            self.dimension()
        );

        // ndarray 的 var_axis(Axis, ddof) 直接计算方差，ddof=0.0 即总体方差
        let var_array = self.data.var_axis(Axis(axis), 0.0);

        // 构建新形状（在方差的轴位置插入 1）
        let mut new_shape: Vec<usize> = var_array.shape().to_vec();
        new_shape.insert(axis, 1);

        Self::new(var_array.as_slice().unwrap(), &new_shape)
    }

    /// 计算张量的标准差
    pub fn std_dev(&self) -> f32 {
        self.data.std_axis(ndarray::Axis(0), 0.).mean().unwrap()
    }

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
            source_id: next_source_id(),
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
            source_id: next_source_id(),
        }
    }

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
            source_id: next_source_id(),
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
            source_id: next_source_id(),
        }
    }
}
