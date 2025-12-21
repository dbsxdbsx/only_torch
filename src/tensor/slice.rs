use ndarray::{ArrayViewD, Slice};
use std::ops::{Range, RangeFull, RangeInclusive};

/// 表示不同类型的张量索引操作
#[derive(Clone, Debug)]
pub enum SliceInfo {
    /// 单个索引（例如：1）
    Single(usize),
    /// 带起始和结束的范围（例如：1..3）
    Range(Range<usize>),
    /// 完整范围（..）
    Full,
    /// 带步长的范围（例如：0..5 step 2）
    Step(Range<usize>, usize),
}

/// `用于将各种类型转换为SliceInfo的内部trait`
pub trait IntoSliceInfo {
    fn into_slice_info(&self) -> SliceInfo;
}

impl IntoSliceInfo for usize {
    fn into_slice_info(&self) -> SliceInfo {
        SliceInfo::Single(*self)
    }
}

impl IntoSliceInfo for Range<usize> {
    fn into_slice_info(&self) -> SliceInfo {
        SliceInfo::Range(self.clone())
    }
}

impl IntoSliceInfo for RangeFull {
    fn into_slice_info(&self) -> SliceInfo {
        SliceInfo::Full
    }
}

// 添加对 RangeInclusive 的支持
impl IntoSliceInfo for RangeInclusive<usize> {
    fn into_slice_info(&self) -> SliceInfo {
        // 将 RangeInclusive 转换为普通 Range
        // RangeInclusive 的结束值需要+1，因为它包含结束值
        SliceInfo::Range(*self.start()..*self.end() + 1)
    }
}

impl super::Tensor {
    /// 获取张量切片的视图，支持灵活的索引方式
    ///
    /// # 与 `NumPy` 的主要区别
    /// - 维度保持：本实现在使用单个索引时保持维度为1，而 `NumPy` 会自动压缩掉该维度
    ///   - 例如：对形状为 [2,3,1,4] 的张量，切片 [:, 0:2, 0, 1:3]
    ///   - 本实现输出形状为 [2,2,1,2]
    ///   - `NumPy` 输出形状为 [2,2,2]（自动压缩了第3维）
    ///
    /// # 参数
    /// * `indices` - 切片索引数组，支持以下索引类型：
    ///   - 单个索引：`&1` -> 选择特定位置
    ///   - 范围：`&(0..2)` -> 选择区间
    ///   - 完整范围：`&(..)` -> 选择全部
    ///   - 步进范围：通过 `SliceInfo::Step` 支持
    ///
    /// # 返回值
    /// 返回一个 `ArrayViewD`<f32> 类型的视图，保持原始数据的所有维度特性
    ///
    /// # 错误
    /// 以下情况会触发panic：
    /// - 空索引列表：`tensor.slice(&[])`
    /// - 维度不匹配：
    ///   - 索引数量少于张量维度：`4维张量.slice(&[&1, &2, &3])`
    ///   - 索引数量多于张量维度：`4维张量.slice(&[&1, &2, &3, &4, &5])`
    /// - 无效索引：
    ///   - 索引超出范围：`2维张量.slice(&[&5, &1])`
    ///   - 范围超出边界：`2维张量.slice(&[&(0..3), &1])`
    ///   - 空范围切片：`tensor.slice(&[&(0..0), &1])`
    ///
    pub fn slice_view(&self, indices: &[&dyn IntoSliceInfo]) -> ArrayViewD<'_, f32> {
        // 检查空索引列表
        assert!(!indices.is_empty(), "slice(_view)无法接受空索引");

        // 检查维度匹配
        assert!(
            indices.len() >= self.dimension(),
            "slice(_view)仅提供了{}个维度的索引，但目标张量是{}维",
            indices.len(),
            self.dimension()
        );
        assert!(
            indices.len() <= self.dimension(),
            "slice(_view)提供了{}个维度的索引，但目标张量只有{}维",
            indices.len(),
            self.dimension()
        );

        // 将输入的索引转换为SliceInfo类型
        let slice_infos: Vec<_> = indices.iter().map(|idx| idx.into_slice_info()).collect();

        // 检查每个维度的索引范围
        for (dim, info) in slice_infos.iter().enumerate() {
            let dim_size = self.shape()[dim];
            match info {
                SliceInfo::Single(i) => {
                    assert!(
                        *i < dim_size,
                        "slice(_view)无法接受某个维度的索引超出目标张量在该维度范围"
                    );
                }
                SliceInfo::Range(r) => {
                    assert!(
                        !(r.start >= dim_size || r.end > dim_size),
                        "slice(_view)无法接受某个维度的索引超出目标张量在该维度范围"
                    );
                    assert!(
                        r.start != r.end,
                        "slice(_view)无法接受某个维度为零数据范围的索引"
                    );
                }
                SliceInfo::Step(r, _) => {
                    assert!(
                        !(r.start >= dim_size || r.end > dim_size),
                        "slice(_view)无法接受某个维度的索引超出目标张量在该维度范围"
                    );
                }
                SliceInfo::Full => {} // 完整范围不需要检查
            }
        }

        // 根据不同的SliceInfo类型创建对应的Slice对象
        let slices: Vec<_> = slice_infos
            .iter()
            .map(|idx| match idx {
                SliceInfo::Single(i) => Slice::from(*i..*i + 1),
                SliceInfo::Range(r) => Slice::from(r.clone()),
                SliceInfo::Full => Slice::from(..),
                SliceInfo::Step(r, step) => Slice::from(r.clone()).step_by(*step as isize),
            })
            .collect();

        // 对张量数据的每个轴应用对应的切片操作
        self.data.slice_each_axis(|ax| slices[ax.axis.index()])
    }

    /// 获取张量切片，返回新的张量
    ///
    /// 这是 `slice_view` 的拥有所有权版本，会创建新的张量而不是视图。
    /// 继承了 `slice_view` 的所有切片特性，包括维度保持的行为。
    ///
    /// # 示例
    /// ```
    /// use only_torch::{Tensor, tensor_slice};
    ///
    /// let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// let slice = tensor.slice(&[&0, &(..)]); // 保持维度：形状为 [1, 2]
    /// ```
    ///
    /// # 参数与返回值
    /// * `indices` - 与 `slice_view` 相同的索引参数
    /// * 返回：包含切片数据的新张量
    pub fn slice(&self, indices: &[&dyn IntoSliceInfo]) -> Self {
        Self::from_view(self.slice_view(indices))
    }
}

/// 简化切片语法的宏
///
/// 通过为每个索引创建局部绑定来避免临时值生命周期问题。
/// 支持 1-6 维张量切片。
#[macro_export]
macro_rules! tensor_slice {
    // 1 维
    ($tensor:expr, $idx1:expr $(,)?) => {{
        use $crate::tensor::slice::IntoSliceInfo;
        let i1 = $idx1;
        $tensor.slice(&[&i1 as &dyn IntoSliceInfo])
    }};
    // 2 维
    ($tensor:expr, $idx1:expr, $idx2:expr $(,)?) => {{
        use $crate::tensor::slice::IntoSliceInfo;
        let i1 = $idx1;
        let i2 = $idx2;
        $tensor.slice(&[&i1 as &dyn IntoSliceInfo, &i2])
    }};
    // 3 维
    ($tensor:expr, $idx1:expr, $idx2:expr, $idx3:expr $(,)?) => {{
        use $crate::tensor::slice::IntoSliceInfo;
        let i1 = $idx1;
        let i2 = $idx2;
        let i3 = $idx3;
        $tensor.slice(&[&i1 as &dyn IntoSliceInfo, &i2, &i3])
    }};
    // 4 维
    ($tensor:expr, $idx1:expr, $idx2:expr, $idx3:expr, $idx4:expr $(,)?) => {{
        use $crate::tensor::slice::IntoSliceInfo;
        let i1 = $idx1;
        let i2 = $idx2;
        let i3 = $idx3;
        let i4 = $idx4;
        $tensor.slice(&[&i1 as &dyn IntoSliceInfo, &i2, &i3, &i4])
    }};
    // 5 维
    ($tensor:expr, $idx1:expr, $idx2:expr, $idx3:expr, $idx4:expr, $idx5:expr $(,)?) => {{
        use $crate::tensor::slice::IntoSliceInfo;
        let i1 = $idx1;
        let i2 = $idx2;
        let i3 = $idx3;
        let i4 = $idx4;
        let i5 = $idx5;
        $tensor.slice(&[&i1 as &dyn IntoSliceInfo, &i2, &i3, &i4, &i5])
    }};
    // 6 维
    ($tensor:expr, $idx1:expr, $idx2:expr, $idx3:expr, $idx4:expr, $idx5:expr, $idx6:expr $(,)?) => {{
        use $crate::tensor::slice::IntoSliceInfo;
        let i1 = $idx1;
        let i2 = $idx2;
        let i3 = $idx3;
        let i4 = $idx4;
        let i5 = $idx5;
        let i6 = $idx6;
        $tensor.slice(&[&i1 as &dyn IntoSliceInfo, &i2, &i3, &i4, &i5, &i6])
    }};
}

/// 视图版本的切片宏
///
/// 通过为每个索引创建局部绑定来避免临时值生命周期问题。
/// 支持 1-6 维张量切片。
#[macro_export]
macro_rules! tensor_slice_view {
    // 1 维
    ($tensor:expr, $idx1:expr $(,)?) => {{
        use $crate::tensor::slice::IntoSliceInfo;
        let i1 = $idx1;
        $tensor.slice_view(&[&i1 as &dyn IntoSliceInfo])
    }};
    // 2 维
    ($tensor:expr, $idx1:expr, $idx2:expr $(,)?) => {{
        use $crate::tensor::slice::IntoSliceInfo;
        let i1 = $idx1;
        let i2 = $idx2;
        $tensor.slice_view(&[&i1 as &dyn IntoSliceInfo, &i2])
    }};
    // 3 维
    ($tensor:expr, $idx1:expr, $idx2:expr, $idx3:expr $(,)?) => {{
        use $crate::tensor::slice::IntoSliceInfo;
        let i1 = $idx1;
        let i2 = $idx2;
        let i3 = $idx3;
        $tensor.slice_view(&[&i1 as &dyn IntoSliceInfo, &i2, &i3])
    }};
    // 4 维
    ($tensor:expr, $idx1:expr, $idx2:expr, $idx3:expr, $idx4:expr $(,)?) => {{
        use $crate::tensor::slice::IntoSliceInfo;
        let i1 = $idx1;
        let i2 = $idx2;
        let i3 = $idx3;
        let i4 = $idx4;
        $tensor.slice_view(&[&i1 as &dyn IntoSliceInfo, &i2, &i3, &i4])
    }};
    // 5 维
    ($tensor:expr, $idx1:expr, $idx2:expr, $idx3:expr, $idx4:expr, $idx5:expr $(,)?) => {{
        use $crate::tensor::slice::IntoSliceInfo;
        let i1 = $idx1;
        let i2 = $idx2;
        let i3 = $idx3;
        let i4 = $idx4;
        let i5 = $idx5;
        $tensor.slice_view(&[&i1 as &dyn IntoSliceInfo, &i2, &i3, &i4, &i5])
    }};
    // 6 维
    ($tensor:expr, $idx1:expr, $idx2:expr, $idx3:expr, $idx4:expr, $idx5:expr, $idx6:expr $(,)?) => {{
        use $crate::tensor::slice::IntoSliceInfo;
        let i1 = $idx1;
        let i2 = $idx2;
        let i3 = $idx3;
        let i4 = $idx4;
        let i5 = $idx5;
        let i6 = $idx6;
        $tensor.slice_view(&[&i1 as &dyn IntoSliceInfo, &i2, &i3, &i4, &i5, &i6])
    }};
}
