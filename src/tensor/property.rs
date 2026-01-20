/*
 * @Author       : 老董
 * @Date         : 2023-10-21 03:22:26
 * @Description  : 本类仅包含一些属性方法，不包含任何运算方法，所以不会需要用到mut
 * @LastEditors  : 老董
 * @LastEditTime : 2024-10-25 05:39:09
 */

use super::Tensor;
use ndarray::{ArrayViewD, ArrayViewMutD};

/// 计算两个形状广播后的输出形状（NumPy 风格）
///
/// # 广播规则
/// - 从右向左对齐维度
/// - 每个维度必须相等，或其中一个为 1
/// - 维度数不同时，较短的形状前面补 1
///
/// # 返回值
/// - `Some(shape)`: 广播后的形状
/// - `None`: 形状不兼容，无法广播
///
/// # 示例
/// ```ignore
/// assert_eq!(broadcast_shape(&[3, 4], &[4]), Some(vec![3, 4]));
/// assert_eq!(broadcast_shape(&[3, 1], &[1, 4]), Some(vec![3, 4]));
/// assert_eq!(broadcast_shape(&[3], &[4]), None);  // 不兼容
/// ```
pub fn broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Option<Vec<usize>> {
    let max_ndim = shape_a.len().max(shape_b.len());
    let mut result = vec![0; max_ndim];

    // 从右向左对齐并计算每个维度
    let iter_a = shape_a.iter().rev();
    let iter_b = shape_b.iter().rev();

    for (i, (d_a, d_b)) in iter_a
        .chain(std::iter::repeat(&1usize))
        .zip(iter_b.chain(std::iter::repeat(&1usize)))
        .take(max_ndim)
        .enumerate()
    {
        let idx = max_ndim - 1 - i;
        if d_a == d_b {
            result[idx] = *d_a;
        } else if *d_a == 1 {
            result[idx] = *d_b;
        } else if *d_b == 1 {
            result[idx] = *d_a;
        } else {
            // 维度不兼容
            return None;
        }
    }

    Some(result)
}

impl Tensor {
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓快照/view(_mut)↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    pub fn view(&self) -> ArrayViewD<'_, f32> {
        ArrayViewD::from_shape(self.shape(), self.data.as_slice().unwrap()).unwrap()
    }
    pub fn view_mut(&mut self) -> ArrayViewMutD<'_, f32> {
        let shape = self.shape().to_owned();
        let slice_mut = self.data.as_slice_mut();
        ArrayViewMutD::from_shape(shape, slice_mut.unwrap()).unwrap()
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑快照/view(_mut)↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /// 若为向量，`shape`可以是[n]、[1,n]、[n,1]；
    /// 若为矩阵，`shape`可以是[n,m]；
    /// 若为更高维度的数组，`shape`可以是[c,n,m,...]。
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// 张量的维数(ndim)，即`shape()`的长度
    /// 如：标量维数为0，向量维数为1，矩阵维数为2，以此类推
    /// NOTE: 这里用`dimension`是参照了PyTorch、NumPy等库的命名规范
    /// `但和MatrixSlow中的``dimension`不同（MatrixSlow/matrixslow/core/node.py#L106）
    /// 后者是张量中所有元素的数量，在本库中请使用`size()`方法来获取
    pub fn dimension(&self) -> usize {
        self.data.ndim()
    }

    /// 计算张量中元素的数量
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// 检查张量是否所有元素为`NaN`。
    /// 因为即使是仅含有1个`NaN`元素的也可能已经初始化，所以这里采用“所有”元素作为判断依据：
    /// 若所有元素为`NaN`，则判定为即未初始化，反之则已判定为已初始化。（单个元素的张量作为特例暂不特殊照顾）
    pub fn is_inited(&self) -> bool {
        !self.data.iter().all(|&x| x.is_nan())
    }

    /// 判断两个张量的形状是否严格一致。如：形状为 [1, 4]，[1, 4]和[4]是不一致的，会返回false
    pub fn is_same_shape(&self, other: &Self) -> bool {
        self.shape() == other.shape()
    }

    /// 判断两个张量是否可以广播（NumPy 风格）
    ///
    /// # 广播规则
    /// - 从右向左对齐维度
    /// - 每个维度必须相等，或其中一个为 1
    /// - 维度数不同时，较短的形状前面补 1
    ///
    /// # 示例
    /// - `[3, 4]` 和 `[4]` → 可广播 (结果 `[3, 4]`)
    /// - `[3, 4]` 和 `[3, 1]` → 可广播 (结果 `[3, 4]`)
    /// - `[3, 1]` 和 `[1, 4]` → 可广播 (结果 `[3, 4]`)
    /// - `[3]` 和 `[4]` → 不可广播
    pub fn can_broadcast_with(&self, other: &Self) -> bool {
        let shape1 = self.shape();
        let shape2 = other.shape();

        // 从右向左对齐比较
        let iter1 = shape1.iter().rev();
        let iter2 = shape2.iter().rev();

        for (d1, d2) in iter1.zip(iter2) {
            // 每个维度必须相等，或其中一个为 1
            if d1 != d2 && *d1 != 1 && *d2 != 1 {
                return false;
            }
        }
        true
    }

    /// 判断 other 是否可以广播到 self 的形状（用于 += 等就地操作）
    ///
    /// # 规则
    /// - other 可以广播到 self 的形状
    /// - 广播后的结果形状必须与 self 相同
    ///
    /// # 示例
    /// - `[3, 4]` 可接受 `[4]` → true (广播后仍是 `[3, 4]`)
    /// - `[3, 4]` 可接受 `[3, 1]` → true
    /// - `[]` 不可接受 `[3]` → false (会扩展形状)
    /// - `[3]` 不可接受 `[2, 3]` → false (会扩展形状)
    pub fn can_assign_broadcast_from(&self, other: &Self) -> bool {
        let self_shape = self.shape();
        let other_shape = other.shape();

        // other 的维度不能超过 self
        if other_shape.len() > self_shape.len() {
            return false;
        }

        // 从右向左对齐比较
        let iter_self = self_shape.iter().rev();
        let iter_other = other_shape.iter().rev();

        for (d_self, d_other) in iter_self.zip(iter_other) {
            // other 的维度必须为 1 或与 self 相等
            if d_other != d_self && *d_other != 1 {
                return false;
            }
        }

        // 检查 other 的多余维度（如果有）是否都是 1
        // 例如：self=[3,4], other=[1,1,4] 是可以的
        // 但由于 other_shape.len() <= self_shape.len() 已检查，这里不需要额外检查

        true
    }

    /// 将张量沿被广播的维度求和，收缩到目标形状
    ///
    /// 用于反向传播时，将梯度从广播后的形状求和回原始形状。
    ///
    /// # 参数
    /// - `target_shape`: 目标形状（通常是广播前的原始形状）
    ///
    /// # 示例
    /// ```ignore
    /// // Forward:  [32, 128] + [1, 128] → [32, 128]
    /// // Backward: grad [32, 128] → sum_to_shape → [1, 128]
    /// let grad = Tensor::new(&data, &[32, 128]);
    /// let result = grad.sum_to_shape(&[1, 128]);
    /// assert_eq!(result.shape(), &[1, 128]);
    /// ```
    ///
    /// # Panics
    /// 如果目标形状与当前形状不兼容（目标形状无法广播到当前形状）
    pub fn sum_to_shape(&self, target_shape: &[usize]) -> Tensor {
        let current_shape = self.shape();

        // 快速路径：形状相同，直接返回克隆
        if current_shape == target_shape {
            return self.clone();
        }

        // 将两个形状对齐到相同长度（左边补 1）
        let max_ndim = current_shape.len().max(target_shape.len());
        let mut padded_current: Vec<usize> = vec![1; max_ndim];
        let mut padded_target: Vec<usize> = vec![1; max_ndim];

        // 从右向左填充
        for (i, &d) in current_shape.iter().rev().enumerate() {
            padded_current[max_ndim - 1 - i] = d;
        }
        for (i, &d) in target_shape.iter().rev().enumerate() {
            padded_target[max_ndim - 1 - i] = d;
        }

        // 找出需要求和的维度（current > target 的维度）
        let mut axes_to_sum: Vec<usize> = Vec::new();
        for (i, (&cur, &tgt)) in padded_current.iter().zip(padded_target.iter()).enumerate() {
            if cur != tgt {
                // 验证：target 必须是 1（否则不是合法的广播关系）
                assert!(
                    tgt == 1,
                    "sum_to_shape: 目标形状 {:?} 与当前形状 {:?} 不兼容",
                    target_shape,
                    current_shape
                );
                axes_to_sum.push(i);
            }
        }

        // 如果没有需要求和的维度，只需要 reshape
        if axes_to_sum.is_empty() {
            return self.reshape(target_shape);
        }

        // 先 reshape 到 padded 形状（如果维度数不同）
        let working_tensor = if current_shape.len() < max_ndim {
            self.reshape(&padded_current)
        } else {
            self.clone()
        };

        // 沿需要求和的维度依次求和（从后向前，避免索引偏移）
        let mut result = working_tensor;
        for &axis in axes_to_sum.iter().rev() {
            result = result.sum_axis_keepdims(axis);
        }

        // 最后 reshape 到目标形状
        result.reshape(target_shape)
    }
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓判断张量是否为标量、向量、矩阵↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 判断张量是否为标量
    /// 判断标准：若形状为空或形状各维数乘积为1，则认为是标量
    pub fn is_scalar(&self) -> bool {
        self.shape().is_empty() || self.shape().iter().product::<usize>() == 1
    }

    /// 判断张量是否为向量
    /// 判断标准：若张量只有一个维度且该维度的大小大于1，或者张量有两个维度且其中一个维度为1，另一个维度大于1
    pub fn is_vector(&self) -> bool {
        let shape = self.shape();
        (shape.len() == 1 && shape[0] > 1)
            || (shape.len() == 2
                && ((shape[0] > 1 && shape[1] == 1) || (shape[1] > 1 && shape[0] == 1)))
    }

    /// 判断张量是否为矩阵
    /// 判断标准：若张量有两个维度且各维度大小均大于1，则认为是矩阵
    pub fn is_matrix(&self) -> bool {
        let shape = self.shape();
        shape.len() == 2 && shape.iter().all(|&x| x > 1)
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑判断张量是否为标量、向量、矩阵↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /// 判断张量中是否存在0值
    pub fn has_zero_value(&self) -> bool {
        self.data.iter().any(|&x| x == 0.)
    }

    /// 将张量转化为纯数（number）。若为标量，则返回Some(number)，否则返回None
    pub fn get_data_number(&self) -> Option<f32> {
        if self.is_scalar() {
            let shape = self.shape();
            let index_array = self.generate_index_array(shape);
            Some(self.data[&index_array[..]])
        } else {
            None
        }
    }

    /// 获取张量数据的连续内存切片（按行主序排列）
    ///
    /// 用于序列化、导出等需要直接访问底层数据的场景
    pub fn data_as_slice(&self) -> &[f32] {
        self.data.as_slice().expect("Tensor 数据应为连续内存布局")
    }
}
