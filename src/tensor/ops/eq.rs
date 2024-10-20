use std::cmp::PartialEq;

use crate::tensor::Tensor;

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓f32 ==（不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl PartialEq<Tensor> for f32 {
    fn eq(&self, other: &Tensor) -> bool {
        other.number().map_or(false, |x| x == *self)
    }
}

impl<'a> PartialEq<&'a Tensor> for f32 {
    fn eq(&self, other: &&'a Tensor) -> bool {
        other.number().map_or(false, |x| x == *self)
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑f32 ==（不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 == f32↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl PartialEq<f32> for Tensor {
    fn eq(&self, other: &f32) -> bool {
        self.number().map_or(false, |x| x == *other)
    }
}

impl<'a> PartialEq<f32> for &'a Tensor {
    fn eq(&self, other: &f32) -> bool {
        self.number().map_or(false, |x| x == *other)
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 == f32↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 ==（不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<'a> PartialEq<&'a Self> for Tensor {
    fn eq(&self, other: &&'a Self) -> bool {
        self.data == other.data
    }
}

impl<'a> PartialEq<Tensor> for &'a Tensor {
    fn eq(&self, other: &Tensor) -> bool {
        self.data == other.data
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 ==（不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

use ndarray::ArrayViewD;
/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ArrayViewD ==（不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl PartialEq<Tensor> for ArrayViewD<'_, f32> {
    fn eq(&self, other: &Tensor) -> bool {
        *self == other.view()
    }
}
impl<'a> PartialEq<&'a Tensor> for ArrayViewD<'_, f32> {
    fn eq(&self, other: &&'a Tensor) -> bool {
        *self == other.view()
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ArrayViewD ==（不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 == ArrayViewD↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl PartialEq<ArrayViewD<'_, f32>> for Tensor {
    fn eq(&self, other: &ArrayViewD<'_, f32>) -> bool {
        self.view() == *other
    }
}
impl<'a> PartialEq<ArrayViewD<'_, f32>> for &'a Tensor {
    fn eq(&self, other: &ArrayViewD<'_, f32>) -> bool {
        self.view() == *other
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 == ArrayViewD↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

use ndarray::ArrayViewMutD;
/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ArrayViewMutD ==（不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl PartialEq<Tensor> for ArrayViewMutD<'_, f32> {
    fn eq(&self, other: &Tensor) -> bool {
        *self == other.view()
    }
}
impl<'a> PartialEq<&'a Tensor> for ArrayViewMutD<'_, f32> {
    fn eq(&self, other: &&'a Tensor) -> bool {
        *self == other.view()
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ArrayViewMutD ==（不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 == ArrayViewMutD↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl PartialEq<ArrayViewMutD<'_, f32>> for Tensor {
    fn eq(&self, other: &ArrayViewMutD<'_, f32>) -> bool {
        self.view() == *other
    }
}
impl<'a> PartialEq<ArrayViewMutD<'_, f32>> for &'a Tensor {
    fn eq(&self, other: &ArrayViewMutD<'_, f32>) -> bool {
        self.view() == *other
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 == ArrayViewMutD↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
