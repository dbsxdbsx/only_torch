use super::Tensor;
use ndarray::{Array, ArrayViewD, ArrayViewMutD, AxisDescription, IxDyn, Slice};

// 快照
impl Tensor {
    pub fn view(&self) -> ArrayViewD<'_, f32> {
        ArrayViewD::from_shape(self.shape(), self.data.as_slice().unwrap()).unwrap()
    }
    pub fn view_mut(&mut self) -> ArrayViewMutD<'_, f32> {
        let shape = self.shape().to_owned();
        let slice_mut = self.data.as_slice_mut();
        ArrayViewMutD::from_shape(shape, slice_mut.unwrap()).unwrap()
    }
}

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓index特性↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
// not ok
// use std::ops::{Index, IndexMut};
// use std::slice::SliceIndex;

// impl<D> Index<D> for Tensor
// where
//     D: Dimension + IntoDimension,
// {
//     type Output = f32;

//     fn index(&self, index: D) -> &Self::Output {
//         let dyn_index = index.into_dimension().into_dyn();
//         let flat_index = self.view()[&dyn_index];
//         &self.data[flat_index]
//     }
// }
// not ok
// impl<Idx> Index<Idx> for Tensor
// where
//     Idx: SliceIndex<[f32]>,
// {
//     type Output = f32;

//     fn index(&self, index: Idx) -> &Self::Output {
//         &self.data[index]
//     }
// }

// not ok
// impl<D> Index<D> for Tensor
// where
//     D: Dimension + IntoDimension,
// {
//     type Output = f32;
//     fn index(&self, index: D) -> &Self::Output {
//         self.view()[index.into_dimension().into_dyn()]
//     }
// }

// impl Index<IxDyn> for Tensor {
//     type Output = f32;
//     fn index(&self, index: IxDyn) -> &f32 {
//         &self.view()[index]
//     }
// }

// impl<'a> IndexMut<IxDyn> for Tensor {
//     fn index_mut(&'a mut self, index: IxDyn) -> &'a mut Self::Output {
//         let view_mut = self.view_mut();
//         &mut view_mut[index]
//     }
// }
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑index特性↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓克隆式索引（局部）张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
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
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑克隆式索引（局部）张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
