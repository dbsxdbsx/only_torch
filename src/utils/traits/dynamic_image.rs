use crate::tensor::Tensor;
use image::{DynamicImage, GenericImageView};

pub trait TraitForDynamicImage {
    fn to_tensor(&self) -> Result<Tensor, String>;
}

impl TraitForDynamicImage for DynamicImage {
    /// 将Image库的`DynamicImage`格式转换为张量
    fn to_tensor(&self) -> Result<Tensor, String> {
        let channels = match self.color() {
            image::ColorType::L8 => 1,
            image::ColorType::Rgb8 => 3,
            image::ColorType::Rgba8 => 4,
            _ => return Err(format!("不支持的图像类型:`{:?}`。", self.color())),
        };
        let width = self.dimensions().0 as usize;
        let height = self.dimensions().1 as usize;
        let init_data = vec![0.0; width * height * channels];
        let mut tensor = Tensor::new(&init_data, &[height, width, channels]);

        for y in 0..height {
            for x in 0..width {
                let pixel = self.get_pixel(x as u32, y as u32);
                let mut view = tensor.view_mut();
                view[[y, x, 0]] = pixel[0] as f32;
                if channels >= 3 {
                    view[[y, x, 1]] = pixel[1] as f32;
                    view[[y, x, 2]] = pixel[2] as f32;
                }
                if channels >= 4 {
                    view[[y, x, 3]] = pixel[3] as f32;
                }
            }
        }

        Ok(tensor)
    }
}
