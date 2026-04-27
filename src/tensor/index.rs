use std::ops::{Index, IndexMut};

use super::{Tensor, next_source_id};
use ndarray::{Array, AxisDescription, IxDyn, Slice};

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓index特性↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
// 不可变index
impl Index<[usize; 0]> for Tensor {
    type Output = f32;
    fn index(&self, _index: [usize; 0]) -> &Self::Output {
        assert!(self.is_scalar());
        let shape = self.shape();
        let index_array = self.generate_index_array(shape);
        &self.data[&index_array[..]]
    }
}
impl Index<[usize; 1]> for Tensor {
    type Output = f32;
    fn index(&self, index: [usize; 1]) -> &Self::Output {
        let idx = IxDyn(&index);
        self.data.index(idx)
    }
}
impl Index<[usize; 2]> for Tensor {
    type Output = f32;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let idx = IxDyn(&index);
        self.data.index(idx)
    }
}
impl Index<[usize; 3]> for Tensor {
    type Output = f32;
    fn index(&self, index: [usize; 3]) -> &Self::Output {
        let idx = IxDyn(&index);
        self.data.index(idx)
    }
}
impl Index<[usize; 4]> for Tensor {
    type Output = f32;
    fn index(&self, index: [usize; 4]) -> &Self::Output {
        let idx = IxDyn(&index);
        self.data.index(idx)
    }
}

// 可变index
impl IndexMut<[usize; 0]> for Tensor {
    fn index_mut(&mut self, _index: [usize; 0]) -> &mut Self::Output {
        assert!(self.is_scalar());
        let shape = self.shape();
        let index_array = self.generate_index_array(shape);
        &mut self.data[&index_array[..]]
    }
}

impl IndexMut<[usize; 1]> for Tensor {
    fn index_mut(&mut self, index: [usize; 1]) -> &mut Self::Output {
        let idx = IxDyn(&index);
        self.data.index_mut(idx)
    }
}

impl IndexMut<[usize; 2]> for Tensor {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        let idx = IxDyn(&index);
        self.data.index_mut(idx)
    }
}

impl IndexMut<[usize; 3]> for Tensor {
    fn index_mut(&mut self, index: [usize; 3]) -> &mut Self::Output {
        let idx = IxDyn(&index);
        self.data.index_mut(idx)
    }
}

impl IndexMut<[usize; 4]> for Tensor {
    fn index_mut(&mut self, index: [usize; 4]) -> &mut Self::Output {
        let idx = IxDyn(&index);
        self.data.index_mut(idx)
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑index特性↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

impl Tensor {
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓克隆式索引（局部）张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 使用给定的索引数组从张量中选取（多个）元素。
    /// * `indices` - 索引值的数组。
    ///
    /// 返回一个“克隆”的张量，其包含根据给定索引选取的（多个）元素。
    /// 若原始张量只有1个元素，则无论`indices`为何，均返回一个包含该元素的新张量，且形状为`&[]`。
    pub fn get(&self, indices: &[usize]) -> Self {
        if let Some(number) = self.get_data_number() {
            return Self::new(&[number], &[]);
        }
        let start: Vec<isize> = indices.iter().map(|&i| i as isize).collect();
        let end: Vec<isize> = indices.iter().map(|&i| (i + 1) as isize).collect();
        let step: Vec<isize> = vec![1; indices.len()];

        let t = Self {
            data: Self::slice_array(&self.data, &start, &end, &step),
            source_id: next_source_id(),
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
            let start_index = start.get(axis_index).copied().unwrap_or(0);
            let end_index = end
                .get(axis_index)
                .copied()
                .unwrap_or(array.len_of(axis.axis) as isize);
            let step_size = step.get(axis_index).copied().unwrap_or(1);
            Slice::new(start_index, Some(end_index), step_size)
        });

        sliced.to_owned()
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑克隆式索引（局部）张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    pub(in crate::tensor) fn generate_index_array(&self, shape: &[usize]) -> Vec<usize> {
        shape.iter().map(|_| 0).collect()
    }
}
