use std::cmp::PartialEq;

use crate::tensor::Tensor;

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓f32 ==（不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
impl PartialEq<Tensor> for f32 {
    fn eq(&self, other: &Tensor) -> bool {
        other.to_number().map_or(false, |x| x == *self)
    }
}

impl<'a> PartialEq<&'a Tensor> for f32 {
    fn eq(&self, other: &&'a Tensor) -> bool {
        other.to_number().map_or(false, |x| x == *self)
    }
}
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑f32 ==（不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 == f32↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
impl PartialEq<f32> for Tensor {
    fn eq(&self, other: &f32) -> bool {
        self.to_number().map_or(false, |x| x == *other)
    }
}

impl<'a> PartialEq<f32> for &'a Tensor {
    fn eq(&self, other: &f32) -> bool {
        self.to_number().map_or(false, |x| x == *other)
    }
}
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 == f32↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 == （不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
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
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 == （不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
