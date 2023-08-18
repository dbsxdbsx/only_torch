use crate::tensor::Tensor;
use std::ops::Div;

impl Div<f32> for Tensor {
    type Output = Self;

    fn div(self, scalar: f32) -> Self {
        if scalar == 0.0 {
            panic!("除数为零");
        }
        Self {
            data: &self.data / scalar,
        }
    }
}

impl Div<Tensor> for f32 {
    type Output = Tensor;

    fn div(self, tensor: Tensor) -> Tensor {
        if tensor.data.iter().any(|&x| x == 0.0) {
            panic!("作为除数的张量中存在为零元素");
        }
        Tensor {
            data: self / &tensor.data,
        }
    }
}

impl Div for Tensor {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        if other.data.iter().any(|&x| x == 0.0) {
            panic!("作为除数的张量中存在为零元素");
        }

        if self.is_scalar() && other.is_scalar() {
            Tensor::new(
                &[self.to_number().unwrap() / other.to_number().unwrap()],
                &[1],
            )
        } else if self.is_same_shape(&other) {
            Self {
                data: &self.data / &other.data,
            }
        } else if self.is_scalar() {
            Self {
                data: self.to_number().unwrap() / &other.data,
            }
        } else if other.is_scalar() {
            Self {
                data: &self.data / other.to_number().unwrap(),
            }
        } else {
            panic!(
                "形状不一致且两个张量没有一个是标量，故无法相除：第一个张量的形状为{:?}，第二个张量的形状为{:?}",
                self.shape(),
                other.shape()
            )
        }
    }
}
