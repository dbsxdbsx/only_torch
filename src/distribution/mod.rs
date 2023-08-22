use crate::tensor::Tensor;
use rand::Rng;

#[cfg(test)]
mod tests;

pub struct Distribution;

impl Distribution {
    pub fn new_normal(mean: f32, std_dev: f32, shape: &[usize]) -> Tensor {
        let mut rng = rand::thread_rng();
        let data = (0..shape.iter().product::<usize>())
            .map(|_| {
                let u1: f32 = rng.gen();
                let u2: f32 = rng.gen();
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                mean + std_dev * z0
            })
            .collect::<Vec<_>>();
        Tensor::new(&data, shape)
    }
}
