use crate::tensor::Tensor;
use std::ops::Div;

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓f32 +（不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
impl Div<Tensor> for f32 {
    type Output = Tensor;

    fn div(self, tensor: Tensor) -> Tensor {
        assert!(!tensor.has_zero_value(), "作为除数的张量中存在为零元素");
        Tensor {
            data: self / &tensor.data,
        }
    }
}
impl<'a> Div<&'a Tensor> for f32 {
    type Output = Tensor;

    fn div(self, tensor: &'a Tensor) -> Tensor {
        assert!(!tensor.has_zero_value(), "作为除数的张量中存在为零元素");
        Tensor {
            data: self / &tensor.data,
        }
    }
}
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑f32 +（不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 / f32↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
impl Div<f32> for Tensor {
    type Output = Tensor;

    fn div(self, scalar: f32) -> Tensor {
        if scalar == 0. {
            panic!("除数为零");
        }
        Tensor {
            data: &self.data / scalar,
        }
    }
}
impl<'a> Div<f32> for &'a Tensor {
    type Output = Tensor;

    fn div(self, scalar: f32) -> Tensor {
        if scalar == 0. {
            panic!("除数为零");
        }
        Tensor {
            data: &self.data / scalar,
        }
    }
}
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 / f32↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量 / （不）带引用的张量↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
impl Div for Tensor {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Tensor {
        div_within_tensors(&self, &other)
    }
}

impl<'a> Div<&'a Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, other: &'a Tensor) -> Tensor {
        div_within_tensors(&self, other)
    }
}

impl<'a> Div<Tensor> for &'a Tensor {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Tensor {
        div_within_tensors(self, &other)
    }
}

impl<'a, 'b> Div<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn div(self, other: &'b Tensor) -> Tensor {
        div_within_tensors(self, other)
    }
}
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量 / （不）带引用的张量↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

fn div_within_tensors(tensor_1: &Tensor, tensor_2: &Tensor) -> Tensor {
    assert!(!tensor_2.has_zero_value(), "作为除数的张量中存在为零元素");

    let data = if tensor_1.is_scalar() && tensor_2.is_scalar() {
        return Tensor::new(
            &[tensor_1.number().unwrap() / tensor_2.number().unwrap()],
            &[1],
        );
    } else if tensor_1.is_same_shape(tensor_2) {
        &tensor_1.data / &tensor_2.data
    } else if tensor_1.is_scalar() {
        tensor_1.number().unwrap() / &tensor_2.data
    } else if tensor_2.is_scalar() {
        &tensor_1.data / tensor_2.number().unwrap()
    } else {
        panic!(
            "形状不一致且两个张量没有一个是标量，故无法相除：第一个张量的形状为{:?}，第二个张量的形状为{:?}",
            tensor_1.shape(),
            tensor_2.shape()
        )
    };

    Tensor { data }
}
