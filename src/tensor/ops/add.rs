use crate::tensor::Tensor;
use std::ops::Add;

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓f32 +（不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
impl Add<Tensor> for f32 {
    type Output = Tensor;

    fn add(self, tensor: Tensor) -> Tensor {
        Tensor {
            data: self + &tensor.data,
        }
    }
}
impl<'a> Add<&'a Tensor> for f32 {
    type Output = Tensor;

    fn add(self, tensor: &'a Tensor) -> Tensor {
        Tensor {
            data: self + &tensor.data,
        }
    }
}
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑f32 +（不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 + f32↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
impl Add<f32> for Tensor {
    type Output = Tensor;

    fn add(self, scalar: f32) -> Tensor {
        Tensor {
            data: &self.data + scalar,
        }
    }
}
impl<'a> Add<f32> for &'a Tensor {
    type Output = Tensor;

    fn add(self, scalar: f32) -> Tensor {
        Tensor {
            data: &self.data + scalar,
        }
    }
}
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 + f32↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 + （不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
impl Add for Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Tensor {
        add_within_tensors(&self, &other)
    }
}

impl<'a> Add<&'a Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, other: &'a Tensor) -> Tensor {
        add_within_tensors(&self, other)
    }
}

impl<'a> Add<Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Tensor {
        add_within_tensors(self, &other)
    }
}

impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, other: &'b Tensor) -> Tensor {
        add_within_tensors(self, other)
    }
}
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 + （不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

fn add_within_tensors(tensor_1: &Tensor, tensor_2: &Tensor) -> Tensor {
    let data = if tensor_1.is_scalar() && tensor_2.is_scalar() {
        return Tensor::new(
            &[tensor_1.number().unwrap() + tensor_2.number().unwrap()],
            &[1],
        );
    } else if tensor_1.is_same_shape(tensor_2) {
        &tensor_1.data + &tensor_2.data
    } else if tensor_1.is_scalar() {
        tensor_1.number().unwrap() + &tensor_2.data
    } else if tensor_2.is_scalar() {
        &tensor_1.data + tensor_2.number().unwrap()
    } else {
        panic!(
            "形状不一致且两个张量没有一个是标量，故无法相加：第一个张量的形状为{:?}，第二个张量的形状为{:?}",
            tensor_1.shape(),
            tensor_2.shape()
        )
    };

    Tensor { data }
}
