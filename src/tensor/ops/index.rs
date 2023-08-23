// impl<'a> Index<&[usize]> for Tensor<'a> {
//     type Output = ArrayView<'a, f32, IxDyn>;

//     fn index(&self, index: &[usize]) -> &Self::Output {
//         assert!(index.len() <= self.dims(), "索引维度过多");
//         let mut index_array = self.generate_index_array(self.shape());
//         for i in 0..index.len() {
//             index_array[i] = index[i];
//         }
//         let x = index_array[..]
//             .iter()
//             .map(|&x| SliceInfoElem::Index(x as isize))
//             .collect::<Vec<_>>();
//         let slice: SliceInfo<&Vec<SliceInfoElem>, ndarray::Dim<ndarray::IxDynImpl>, _> =
//             unsafe { SliceInfo::<_, IxDyn, _>::new(&x).unwrap() };

//         unsafe { std::mem::transmute(&self.data.slice(slice)) }
//     }
// }
// maybe workable
use ndarray::{s, Array, ArrayView, IxDyn, Slice, SliceInfo, SliceInfoElem};
use std::ops::Index;

use crate::tensor::Tensor;

impl<'a> Index<&[usize]> for Tensor<'a> {
    type Output = ArrayView<'a, f32, IxDyn>;

    fn index(&self, index: &[usize]) -> &Self::Output {
        assert!(index.len() <= self.dims(), "索引维度过多");
        let mut index_array = self.generate_index_array(self.shape());
        for i in 0..index.len() {
            index_array[i] = index[i];
        }
        let x = index_array[..]
            .iter()
            .map(|&x| SliceInfoElem::Index(x as isize))
            .collect::<Vec<_>>();
        let slice = unsafe { SliceInfo::<_, IxDyn, _>::new(&x).unwrap() };

        &self.view.slice(slice)
    }
}
// old
// impl Index<&[usize]> for Tensor {
//     type Output = Array<f32, IxDyn>;

//     fn index(&self, index: &[usize]) -> &Self::Output {
//         assert!(index.len() <= self.dims(), "索引维度过多");
//         let mut index_array = self.generate_index_array(self.shape());
//         for i in 0..index.len() {
//             index_array[i] = index[i];
//         }
//         // let slice = Slice::new(1,2,3);
//         let slice = unsafe {
//             SliceInfo::<_, IxDyn, _>::new(
//                 &index_array[..]
//                     .iter()
//                     .map(|&x| SliceInfoElem::Index(x as isize))
//                     .collect::<Vec<_>>(),
//             )
//             .unwrap()
//         };

//         self.data.slice(slice)
//     }
// }

// impl Index<&[usize]> for Tensor {
//     type Output = [f32];

//     fn index(&self, index: &[usize]) -> &Self::Output {
//         assert!(index.len() <= self.dims(), "索引维度过多");
//         let mut index_array = self.generate_index_array(self.shape());
//         for i in 0..index.len() {
//             index_array[i] = index[i];
//         }
//         &self.data[&index_array[..]]
//     }
// }
