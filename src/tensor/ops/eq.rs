use std::cmp::PartialEq;

use crate::tensor::Tensor;

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<'a> PartialEq<&'a Tensor> for Tensor {
    fn eq(&self, other: &&'a Tensor) -> bool {
        self.data == other.data
    }
}

impl<'a> PartialEq<Tensor> for &'a Tensor {
    fn eq(&self, other: &Tensor) -> bool {
        self.data == other.data
    }
}

// impl<'a, 'b> PartialEq<&'b Tensor> for &'a Tensor {
//     fn eq(&self, other: &'b Tensor) -> bool {
//         self.data == other.data
//     }
// }
