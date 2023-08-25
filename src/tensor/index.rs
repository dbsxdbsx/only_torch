use ndarray::{Array, AxisDescription, IxDyn, Slice};

use super::Tensor;
impl Tensor {
    /// 使用给定的索引数组从张量中选取元素。
    /// * `idx` - 一个包含索引值的数组。
    ///
    /// 返回一个新的张量，包含根据给定索引选取的（多个）元素。
    /// 若原始张量只有一个元素，则无论`idx`为何，均返回一个包含该元素的新张量，且形状为`&[]`。
    pub fn get(&self, idx: &[usize]) -> Tensor {
        if let Some(number) = self.number() {
            return Tensor::new(&[number], &[]);
        }
        let start: Vec<isize> = idx.iter().map(|&i| i as isize).collect();
        let end: Vec<isize> = idx.iter().map(|&i| (i + 1) as isize).collect();
        let step: Vec<isize> = vec![1; idx.len()];

        let t = Tensor {
            data: slice_array(&self.data, &start, &end, &step),
        };
        t.squeeze() //将所有仅为1的维度优化掉
    }
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
