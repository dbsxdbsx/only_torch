use crate::tensor::Tensor;
use image::{DynamicImage, GenericImageView};

pub trait TensorTraitForDynamicImage {
    fn to_tensor(&self) -> Result<Tensor, String>;
}

impl TensorTraitForDynamicImage for DynamicImage {
    fn to_tensor(&self) -> Result<Tensor, String> {
        let channels = match self.color() {
            image::ColorType::L8 => 1,
            image::ColorType::Rgb8 => 3,
            image::ColorType::Rgba8 => 4,
            _ => return Err(format!("不支持的图像类型:`{:?}`。", self.color())),
        };

        let (width, height) = self.dimensions();
        let tensor = Tensor::new(&[], &[height as usize, width as usize, channels]);

        for y in 0..height {
            for x in 0..width {
                let _pixel = self.get_pixel(x, y);
                // // TODO:
                // tensor[[y as usize, x as usize, 0]] = pixel[0] as f32;

                // if channels >= 3 {
                //     tensor[[y as usize, x as usize, 1]] = pixel[1] as f32;
                //     tensor[[y as usize, x as usize, 2]] = pixel[2] as f32;
                // }

                // if channels >= 4 {
                //     tensor[[y as usize, x as usize, 3]] = pixel[3] as f32;
                // }
            }
        }

        Ok(tensor)
    }
}
