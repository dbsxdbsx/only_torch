use super::Tensor;
use ndarray::{s, Array, ArrayViewD, Axis, AxisDescription, IxDyn, Slice, SliceNextDim};

// 克隆式索引
impl Tensor {
    /// 使用给定的索引数组从张量中选取（多个）元素。
    /// * `indices` - 索引值的数组。
    ///
    /// 返回一个“克隆”的张量，其包含根据给定索引选取的（多个）元素。
    /// 若原始张量只有一个元素，则无论`indices`为何，均返回一个包含该元素的新张量，且形状为`&[]`。
    pub fn get(&self, indices: &[usize]) -> Tensor {
        if let Some(number) = self.number() {
            return Tensor::new(&[number], &[]);
        }
        let start: Vec<isize> = indices.iter().map(|&i| i as isize).collect();
        let end: Vec<isize> = indices.iter().map(|&i| (i + 1) as isize).collect();
        let step: Vec<isize> = vec![1; indices.len()];

        let t = Tensor {
            data: Self::slice_array(&self.data, &start, &end, &step),
        };
        t.squeeze() //将所有仅为1的维度优化掉
    }

    fn slice_array(
        array: &Array<f32, IxDyn>,
        start: &[isize],
        end: &[isize],
        step: &[isize],
    ) -> Array<f32, IxDyn> {
        let sliced = array.slice_each_axis(|axis: AxisDescription| {
            let axis_index = axis.axis.index();
            let start_index = start.get(axis_index).cloned().unwrap_or(0);
            let end_index = end
                .get(axis_index)
                .cloned()
                .unwrap_or(array.len_of(axis.axis) as isize);
            let step_size = step.get(axis_index).cloned().unwrap_or(1);
            Slice::new(start_index, Some(end_index), step_size)
        });

        sliced.to_owned()
    }
}

// 引用式索引
impl Tensor {
    /// 使用给定的“单个”索引从张量中选取某个元素的“可变引用”。
    /// * `indices` - 一个包含索引值的数组。
    pub fn index(&self, indices: &[usize]) -> &mut f32 {
        todo!()
    //     let view = self.view();
    //     &mut &view[indices]
    }

    // pub fn index(&self, indices: &[usize]) -> ArrayViewD<'_, f32> {
    //     let view = self.view();

    //     let start = indices.to_vec();
    //     let end = indices.iter().map(|&i| i + 1).collect::<Vec<_>>();

    //     view.slice(s![start, end])
    // }
}
