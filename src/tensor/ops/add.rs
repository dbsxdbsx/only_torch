use crate::tensor::Tensor;
use std::ops::Add;

impl Add<f32> for Tensor {
    type Output = Self;

    fn add(self, scalar: f32) -> Self {
        Self {
            data: &self.data + scalar,
        }
    }
}

impl Add<Tensor> for f32 {
    type Output = Tensor;

    fn add(self, tensor: Tensor) -> Tensor {
        Tensor {
            data: &tensor.data + self,
        }
    }
}

impl Add for Tensor {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let data = if self.is_scalar() && other.is_scalar() {
            return Tensor::new(
                &[self.to_number().unwrap() + other.to_number().unwrap()],
                &[1],
            );
        } else if self.is_same_shape(&other) {
            self.data + other.data
        } else if self.is_scalar() {
            self.to_number().unwrap() + other.data
        } else if other.is_scalar() {
            self.data + other.to_number().unwrap()
        } else {
            panic!(
                "形状不一致且两个张量没有一个是标量，故无法相加：第一个张量的形状为{:?}，第二个张量的形状为{:?}",
                self.shape(),
                other.shape()
            )
        };

        Self { data }
    }
}
